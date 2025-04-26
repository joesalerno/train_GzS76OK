import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import logging
import lightgbm as lgb  # Added for early stopping callback

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10] # Lags based on num_orders
ROLLING_WINDOWS = [3, 5, 10] # Rolling windows based on num_orders
# Other features (not directly dependent on recursive prediction)
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 50 # Number of Optuna trials
OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "submission_recursive"
SHAP_FILE_PREFIX = "shap_recursive"
N_SHAP_SAMPLES = 2000

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Data ---
logging.info("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    meal_info = pd.read_csv(MEAL_INFO_PATH)
    center_info = pd.read_csv(CENTER_INFO_PATH)
except FileNotFoundError as e:
    logging.error(f"Error loading data file: {e}. Ensure train.csv, test.csv, meal_info.csv, and fulfilment_center_info.csv are present.")
    raise

# --- Preprocessing ---
logging.info("Preprocessing data...")
def preprocess_data(df, meal_info, center_info):
    """Merges dataframes and sorts."""
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

df = preprocess_data(df, meal_info, center_info)
test = preprocess_data(test, meal_info, center_info)

# Add placeholder for num_orders in test for alignment
if 'num_orders' not in test.columns:
    test['num_orders'] = np.nan

# --- Feature Engineering ---
logging.info("Creating features...")
GROUP_COLS = ["center_id", "meal_id"]

def create_lag_rolling_features(df, target_col='num_orders', lag_weeks=LAG_WEEKS, rolling_windows=ROLLING_WINDOWS):
    """Creates lag and rolling window features for a given target column."""
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Lags
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)

    # Rolling features (use shift(1) to avoid data leakage)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)

    return df_out

def create_other_features(df):
    """Creates features not directly dependent on recursive prediction."""
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Price features
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan) # Avoid division by zero
    df_out["price_diff"] = group["checkout_price"].diff()

    # Rolling sums for promo/featured (use shift(1))
    for col in OTHER_ROLLING_SUM_COLS:
        shifted = group[col].shift(1)
        df_out[f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}"] = shifted.rolling(OTHER_ROLLING_SUM_WINDOW, min_periods=1).sum().reset_index(drop=True)

    # Time features
    df_out["weekofyear"] = df_out["week"] % 52

    return df_out

def create_interaction_features(df):
    """Creates interaction features."""
    df_out = df.copy()
    # Define interactions (ensure base features exist)
    interactions = {
        "price_diff_x_emailer": ("price_diff", "emailer_for_promotion"),
        "lag1_x_emailer": ("num_orders_lag_1", "emailer_for_promotion"),
        "price_diff_x_home": ("price_diff", "homepage_featured"),
        "lag1_x_home": ("num_orders_lag_1", "homepage_featured"),
    }
    for name, (feat1, feat2) in interactions.items():
         if feat1 in df_out.columns and feat2 in df_out.columns:
              df_out[name] = df_out[feat1] * df_out[feat2]
         else:
              logging.warning(f"Skipping interaction '{name}' because base feature(s) missing.")
              df_out[name] = 0 # Add column with default value if base features missing

    return df_out

def apply_feature_engineering(df, is_test=False):
    """Applies all feature engineering steps."""
    df_out = df.copy()
    if not is_test: # Only create historical features from actuals in training data
        df_out = create_lag_rolling_features(df_out)
    df_out = create_other_features(df_out)
    df_out = create_interaction_features(df_out) # Interactions rely on lags created above

    # Fill NaNs created by shifts/rolling/diffs - important after creating all features
    # Identify columns generated by lag/roll/diff logic
    lag_roll_diff_cols = [col for col in df_out.columns if
                           any(sub in col for sub in ["lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home"])]
    # Only fill NaNs in columns that actually exist in the dataframe
    cols_to_fill = [col for col in lag_roll_diff_cols if col in df_out.columns]
    df_out[cols_to_fill] = df_out[cols_to_fill].fillna(0)

    # Fill discount_pct NaNs (e.g., from base_price=0)
    if "discount_pct" in df_out.columns:
      df_out["discount_pct"] = df_out["discount_pct"].fillna(0)

    return df_out

# Apply initial feature engineering (excluding lags/rolling for now on combined data)
df_full = pd.concat([df, test], ignore_index=True) # Combine for consistent encoding
df_full = create_other_features(df_full) # Features not dependent on target lags

# --- One-hot encoding ---
logging.info("Applying one-hot encoding...")
cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in df_full.columns]
if cat_cols:
    df_full = pd.get_dummies(df_full, columns=cat_cols, dummy_na=False) # Avoid NaN columns from dummies

# Separate back into train/test after encoding
train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()

# Now create lag/rolling features for the training data (using actual num_orders)
train_df = create_lag_rolling_features(train_df)
# Interactions need lags, create them now for train_df
train_df = create_interaction_features(train_df)

# Fill NaNs generated in the lag/rolling/interaction steps for train_df
lag_roll_diff_cols = [col for col in train_df.columns if
                       any(sub in col for sub in ["lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home"])]
cols_to_fill = [col for col in lag_roll_diff_cols if col in train_df.columns]
train_df[cols_to_fill] = train_df[cols_to_fill].fillna(0)
if "discount_pct" in train_df.columns:
    train_df["discount_pct"] = train_df["discount_pct"].fillna(0)

# Drop rows in train_df where target is NA (if any, though unlikely from problem desc)
train_df = train_df.dropna(subset=['num_orders']).reset_index(drop=True)


# --- Define Features and Target ---
TARGET = "num_orders"
# Define potential features (check existence before adding)
FEATURES = ["week", "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion"]
FEATURES += [col for col in ["discount", "discount_pct", "price_diff", "weekofyear"] if col in train_df.columns]
# REMOVE lag features:
# FEATURES += [f"{TARGET}_lag_{lag}" for lag in LAG_WEEKS if f"{TARGET}_lag_{lag}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_mean_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_mean_{w}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_std_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_std_{w}" in train_df.columns]
FEATURES += [f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}" for col in OTHER_ROLLING_SUM_COLS if f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}" in train_df.columns]
# Remove lag-based interaction features as well
FEATURES += [col for col in train_df.columns if col.startswith("price_diff_x_")]
# Do NOT include lag1_x_emailer or lag1_x_home
FEATURES += [col for col in train_df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]

# Ensure only existing columns are included (robustness)
FEATURES = [f for f in FEATURES if f not in ("week", "weekofyear") and f in train_df.columns and f != TARGET and f !='id']
logging.info(f"Using {len(FEATURES)} features: {FEATURES}")


# --- Train/validation split ---
max_week = train_df["week"].max()
valid_df = train_df[train_df["week"] > max_week - VALIDATION_WEEKS].copy()
train_split_df = train_df[train_df["week"] <= max_week - VALIDATION_WEEKS].copy()

logging.info(f"Train split shape: {train_split_df.shape}, Validation shape: {valid_df.shape}")

# --- RMSLE Metric ---
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0) # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False # lower is better

# --- Model Training Function ---
def get_lgbm(params=None):
    """Initializes LGBMRegressor with default or provided params."""
    default_params = {
        'objective': 'regression_l1', # MAE objective often works well for RMSLE
        'metric': 'None', # Use custom metric
        'boosting_type': 'gbdt',
        'n_estimators': 2000, # Increase estimators, use early stopping
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    if params:
        default_params.update(params)
        # Ensure metric is None if custom metric is used during fit
        if 'eval_metric' in params and params['eval_metric'] == lgb_rmsle:
             default_params['metric'] = 'None'
    return LGBMRegressor(**default_params)


# --- Optuna Hyperparameter Tuning ---
logging.info("Starting Optuna hyperparameter tuning...")

# Use Optuna's SQLite storage for persistence (no joblib)
try:
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Loaded existing Optuna study from {OPTUNA_DB}")
except Exception:
    study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Created new Optuna study at {OPTUNA_DB}")

def objective(trial):
    """Optuna objective function."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
    }
    # Add fixed params
    params.update({
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric':'None', # Crucial when using feval
    })

    model = LGBMRegressor(**params)
    model.fit(
        train_split_df[FEATURES], train_split_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric=lgb_rmsle, # Use custom RMSLE metric
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmsle'), # Pruning based on validation RMSLE
                   lgb.early_stopping(100, verbose=False)] # Early stopping
    )
    preds = model.predict(valid_df[FEATURES])
    score = rmsle(valid_df[TARGET], preds)
    return score

# Run Optuna optimization
study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=1800) # Add a timeout (e.g., 30 minutes)

# No need to save with joblib, study is persisted in SQLite
logging.info(f"Optuna study saved to {OPTUNA_DB}")

best_params = study.best_params
logging.info(f"Best Optuna params: {best_params}")
logging.info(f"Best validation RMSLE: {study.best_value:.5f}")

# --- Final Model Training ---
logging.info("Training final model on full training data with best params...")
# Merge best params with fixed params for the final model
final_params = {
    'objective': 'regression_l1',
    'boosting_type': 'gbdt',
    'n_estimators': 3000, # Increase slightly for final training
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'metric': 'None'
}
final_params.update(best_params) # Best params from Optuna override defaults

final_model = LGBMRegressor(**final_params)

# Train on the entire training dataset (train_split_df + valid_df)
# No early stopping here, train for the specified number of estimators
final_model.fit(train_df[FEATURES], train_df[TARGET], eval_metric=lgb_rmsle)

# --- Recursive Prediction ---
logging.info("Starting recursive prediction on the test set...")
# Prepare the combined data history (training data + test structure)
# We need the structure of test_df but will fill num_orders recursively
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

test_weeks = sorted(test_df['week'].unique())

for week_num in test_weeks:
    logging.info(f"Predicting for week {week_num}...")
    # Identify rows for the current week to predict
    current_week_mask = history_df['week'] == week_num

    # 1. Create features for the current week using history *up to the previous week*
    #    Need to recalculate lag/rolling features based on potentially filled values from previous iterations
    history_df = create_lag_rolling_features(history_df, target_col='num_orders')
    #    Recalculate interactions based on updated lags
    history_df = create_interaction_features(history_df)
    #    Fill NaNs resulting from these calculations for the current week
    lag_roll_diff_cols = [col for col in history_df.columns if
                           any(sub in col for sub in ["lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home"])]
    cols_to_fill = [col for col in lag_roll_diff_cols if col in history_df.columns]

    # Fill NaNs carefully - only for the columns that need it, potentially just for the current week if needed
    # A global fillna(0) might be sufficient if handled correctly after feature creation
    history_df[cols_to_fill] = history_df[cols_to_fill].fillna(0)
    if "discount_pct" in history_df.columns:
        history_df["discount_pct"] = history_df["discount_pct"].fillna(0)

    # 2. Select features for the current week
    current_features = history_df.loc[current_week_mask, FEATURES]

    # Handle potential missing columns in test data after alignment (should not happen with proper alignment, but defensive)
    missing_cols = [col for col in FEATURES if col not in current_features.columns]
    if missing_cols:
        logging.warning(f"Missing columns during prediction for week {week_num}: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            current_features[col] = 0
    current_features = current_features[FEATURES] # Ensure correct order

    # 3. Predict for the current week
    current_preds = final_model.predict(current_features)
    current_preds = np.clip(current_preds, 0, None).round().astype(float) # Use float for potential later calculations

    # 4. Update the 'num_orders' in history_df for the current week with predictions
    #    This ensures the next iteration uses the predicted values to calculate lags/rolling features
    history_df.loc[current_week_mask, 'num_orders'] = current_preds

logging.info("Recursive prediction finished.")

# Extract final predictions for the original test set IDs
final_predictions_df = history_df.loc[history_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int) # Final conversion to int
final_predictions_df['id'] = final_predictions_df['id'].astype(int)

# --- Create Submission File ---
submission_path = f"{SUBMISSION_FILE_PREFIX}_optuna.csv"
final_predictions_df.to_csv(submission_path, index=False)
logging.info(f"Submission file saved to {submission_path}")

# --- SHAP Analysis ---
logging.info("Calculating SHAP values...")
try:
    # Sample data for SHAP to keep computation reasonable
    if len(train_df) > N_SHAP_SAMPLES:
        shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
    else:
        shap_sample = train_df.copy()

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample[FEATURES])

    # Save SHAP values and importance
    shap_values_df = pd.DataFrame(shap_values, columns=FEATURES)
    shap_values_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_values.csv", index=False)

    shap_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_feature_importances.csv", index=False)

    # Generate SHAP plots
    logging.info("Generating SHAP plots...")
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, shap_sample[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_summary.png")
    plt.close()

    # Importance Bar Plot (Top 20)
    plt.figure(figsize=(10, 8))
    shap_importance_df.head(20).plot(kind='barh', x='feature', y='mean_abs_shap', legend=False, figsize=(10, 8))
    plt.gca().invert_yaxis() # Display most important at the top
    plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
    plt.title('Top 20 SHAP Feature Importances (Recursive Optuna Model)')
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_top20_importance.png")
    plt.close()

    logging.info("SHAP analysis saved.")

except Exception as e:
    logging.error(f"Error during SHAP analysis: {e}")


# --- Plotting Example: Actual vs Predicted for Validation Set ---
logging.info("Generating validation plot...")
try:
    valid_preds = final_model.predict(valid_df[FEATURES])
    valid_preds = np.clip(valid_preds, 0, None)

    plt.figure(figsize=(15, 6))
    plt.scatter(valid_df[TARGET], valid_preds, alpha=0.5, s=10)
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2, label='Ideal')
    plt.xlabel("Actual Orders (Validation Set)")
    plt.ylabel("Predicted Orders (Validation Set)")
    plt.title(f"Actual vs. Predicted Orders (Validation Set) - RMSLE: {rmsle(valid_df[TARGET], valid_preds):.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_actual_vs_predicted.png")
    plt.close()
    logging.info("Validation plot saved.")

except Exception as e:
    logging.error(f"Error during plotting: {e}")

logging.info("Script finished.")
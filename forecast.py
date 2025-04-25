import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
# from sklearn.model_selection import train_test_split # Not used directly for time split
from sklearn.model_selection import TimeSeriesSplit # Added for potential future use
import optuna
import lightgbm as lgb # Explicit import needed for callback
import shap
import gc # Garbage collector

# --- Configuration ---
USE_TUNER = True  # Set to False to skip tuning
N_TRIALS = 50     # Number of Optuna trials
VALIDATION_WEEKS = 8 # How many recent weeks for validation
TARGET_TRANSFORM = True # Use log1p transform for the target

# File Paths
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"

# Feature Engineering Parameters
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
EWMA_SPANS = [3, 5, 10] # Added EWMA spans

SEED = 42
TARGET = "num_orders"

# --- Utility Functions ---
def reduce_mem_usage(df, verbose=True):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Only try to cast to int if there are no NaN or inf values
                if not df[col].isnull().any() and np.isfinite(df[col]).all():
                    if (df[col] == df[col].astype(np.int32)).all():
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        else:
                            df[col] = df[col].astype(np.int64)
                # Check float precision
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16) # Use float16 with caution
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # Skip object type columns for now, consider converting to 'category' if appropriate

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df

# --- Load Data ---
print("Loading data...")
train_df = pd.read_csv(DATA_PATH)
test_df = pd.read_csv(TEST_PATH)
meal_info = pd.read_csv(MEAL_INFO_PATH)
center_info = pd.read_csv(CENTER_INFO_PATH)

# Merge information
train_df = train_df.merge(meal_info, on="meal_id", how="left")
train_df = train_df.merge(center_info, on="center_id", how="left")
test_df = test_df.merge(meal_info, on="meal_id", how="left")
test_df = test_df.merge(center_info, on="center_id", how="left")

# Combine for efficient feature engineering
train_df['is_train'] = 1
test_df['is_train'] = 0
# Add placeholder for target in test set
test_df[TARGET] = np.nan
# Concatenate, ensuring test 'id' is preserved if needed later (it's the index in test_df)
combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

# Sort for time-based features
combined_df = combined_df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

print("Initial data loaded and combined.")
del train_df, test_df, meal_info, center_info # Free memory
gc.collect()

# --- Feature Engineering (Vectorized) ---
print("Starting feature engineering...")
def create_features_vectorized(df):
    df = df.copy()
    group_cols = ["center_id", "meal_id"]

    # Lags
    print("  Creating lag features...")
    for lag in LAG_WEEKS:
        df[f"num_orders_lag_{lag}"] = df.groupby(group_cols)[TARGET].shift(lag)
        # Optional: Fill initial NaNs if needed, e.g., with 0 or ffill after shift
        # df[f"num_orders_lag_{lag}"] = df.groupby(group_cols)[f"num_orders_lag_{lag}"].ffill()


    # Rolling Window Features (Mean, Std, EWMA) - shift first to avoid target leakage
    print("  Creating rolling window features...")
    shifted_orders = df.groupby(group_cols)[TARGET].shift(1) # Shift once for all rolling calculations
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted_orders.rolling(window, min_periods=1).mean().reset_index(0, drop=True) # min_periods=1 handles start
        df[f"rolling_std_{window}"] = shifted_orders.rolling(window, min_periods=2).std().reset_index(0, drop=True)   # min_periods=2 for std
        # Optional: Fill NaNs after rolling
        # df[f"rolling_mean_{window}"] = df.groupby(group_cols)[f"rolling_mean_{window}"].ffill()
        # df[f"rolling_std_{window}"] = df.groupby(group_cols)[f"rolling_std_{window}"].ffill().fillna(0) # Fill std NaNs with 0

    # EWMA Features
    print("  Creating EWMA features...")
    for span in EWMA_SPANS:
         df[f"ewma_{span}"] = shifted_orders.ewm(span=span, adjust=False).mean().reset_index(0, drop=True)
         # Optional: Fill NaNs
         # df[f"ewma_{span}"] = df.groupby(group_cols)[f"ewma_{span}"].ffill()


    # Price Features
    print("  Creating price features...")
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = (df["discount"] / df["base_price"]).fillna(0).replace([np.inf, -np.inf], 0) # Handle division by zero
    # Correct price difference (current vs lag 1)
    df['checkout_price_lag_1'] = df.groupby(group_cols)["checkout_price"].shift(1)
    df['price_diff'] = df['checkout_price'] - df['checkout_price_lag_1']
    # df['price_diff'] = df.groupby(group_cols)['price_diff'].ffill() # Optional fill

    # Promotion Rolling Sum
    print("  Creating promotion features...")
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted_promo = df.groupby(group_cols)[col].shift(1)
        df[f"{col}_rolling_sum_3"] = shifted_promo.rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        # df[f"{col}_rolling_sum_3"] = df.groupby(group_cols)[f"{col}_rolling_sum_3"].ffill() # Optional fill

    # Time Features
    print("  Creating time features...")
    df["weekofyear"] = df["week"] % 52
    df["month"] = (df["week"] // 4) % 13 # Approximate month

    # Interaction Feature Example (Optional)
    # df['price_x_promo'] = df['checkout_price'] * df['emailer_for_promotion']

    # Fill remaining NaNs strategically (e.g., with 0 or -1, or median/mean)
    # For lags/rolling, 0 might be reasonable if it means no prior data
    lag_roll_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'ewma' in col or 'diff' in col]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0) # Example: fill with 0

    return df

combined_df = create_features_vectorized(combined_df)
print("Feature engineering complete.")
gc.collect()

# --- Target Transformation ---
if TARGET_TRANSFORM:
    print(f"Applying log1p transformation to target '{TARGET}'...")
    # Apply only to training data rows where target is not NaN
    train_mask = combined_df['is_train'] == 1
    combined_df.loc[train_mask, TARGET] = np.log1p(combined_df.loc[train_mask, TARGET])
    # Note: NaNs in the test set target remain NaNs

# --- One-Hot Encode Categorical Features ---
print("Encoding categorical features...")
cat_cols = []
# Check which columns actually exist and are object/category type
potential_cat_cols = ["category", "cuisine", "center_type"]
for col in potential_cat_cols:
    if col in combined_df.columns and combined_df[col].dtype in ['object', 'category']:
        cat_cols.append(col)

if cat_cols:
    print(f"  One-hot encoding: {cat_cols}")
    combined_df = pd.get_dummies(combined_df, columns=cat_cols, dummy_na=False) # dummy_na=False is usually safer
else:
    print("  No categorical columns found for encoding.")

# --- Reduce Memory Usage ---
print("Reducing memory usage...")
combined_df = reduce_mem_usage(combined_df)
gc.collect()


# --- Define Features ---
FEATURES = [
    # Original IDs/Prices (can be useful)
    "center_id", "meal_id",
    "checkout_price", "base_price",
    # Promotions
    "homepage_featured", "emailer_for_promotion",
    # Price derived
    "discount", "discount_pct", "price_diff", 'checkout_price_lag_1',
    # Time
    "week", "weekofyear", "month",
    # Center/Meal Info (original) - Keep if not one-hot encoded above
    "op_area", "city_code", "region_code",
    # Lags
] + [f"num_orders_lag_{lag}" for lag in LAG_WEEKS] + \
  [ # Rolling
    f"rolling_mean_{w}" for w in ROLLING_WINDOWS
] + [
    f"rolling_std_{w}" for w in ROLLING_WINDOWS
] + [ # EWMA
    f"ewma_{span}" for span in EWMA_SPANS
] + [ # Promotion Rolling
    f"{col}_rolling_sum_3" for col in ["emailer_for_promotion", "homepage_featured"]
]
# Add one-hot encoded columns dynamically
ohe_cols = [col for col in combined_df.columns if any(col.startswith(prefix + "_") for prefix in potential_cat_cols)]
FEATURES += ohe_cols

# Ensure all selected features exist in the dataframe
FEATURES = [f for f in FEATURES if f in combined_df.columns and f != TARGET and f != 'is_train' and f != 'id']
# Remove features with zero variance if any were created
#nunique = combined_df[FEATURES].nunique()
#FEATURES = nunique[nunique > 1].index.tolist()

print(f"Using {len(FEATURES)} features.")
# print("Features:", FEATURES) # Uncomment to see the full list

# --- Split back into Train, Validation, Test ---
print("Splitting data...")
# Training data (excluding validation period)
max_train_week = combined_df[combined_df['is_train'] == 1]['week'].max()
validation_start_week = max_train_week - VALIDATION_WEEKS + 1

train_df = combined_df[(combined_df['is_train'] == 1) & (combined_df['week'] < validation_start_week)].reset_index(drop=True)
valid_df = combined_df[(combined_df['is_train'] == 1) & (combined_df['week'] >= validation_start_week)].reset_index(drop=True)
test_df = combined_df[combined_df['is_train'] == 0].reset_index(drop=True) # Contains original 'id' if needed

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {valid_df.shape}")
print(f"Test shape: {test_df.shape}")

del combined_df # Free memory
gc.collect()


# --- Model Training ---
def get_lgbm(params=None):
    """Initializes an LGBMRegressor with default or provided parameters."""
    default_params = dict(
        objective="regression_l1" if TARGET_TRANSFORM else "regression", # MAE objective (robust to outliers) if log-transformed
        metric="rmse", # LGBM will report RMSE, Optuna optimizes based on objective's return
        learning_rate=0.05, # Slightly lower default LR
        # num_leaves=31, # Tuned by Optuna
        # feature_fraction=0.9, # Tuned by Optuna
        # bagging_fraction=0.9, # Tuned by Optuna
        # bagging_freq=1, # Tuned by Optuna
        # min_child_samples=20, # Tuned by Optuna
        # lambda_l1=0.1, # Tuned by Optuna
        # lambda_l2=0.1, # Tuned by Optuna
        # max_depth=-1, # Default no limit, Tuned by Optuna
        n_estimators=2000, # High number, rely on early stopping
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    if params:
        default_params.update(params)
    return LGBMRegressor(**default_params)

if USE_TUNER:
    print(f"\nStarting Optuna hyperparameter tuning ({N_TRIALS} trials)...")
    # Define the objective function for Optuna
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100), # Wider range
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60), # Wider range
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True), # Wider range
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True), # Wider range
            'max_depth': trial.suggest_int('max_depth', 3, 12), # Allow deeper trees
            # 'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 1), # Optional
        }
        model = get_lgbm(params)

        # Use early stopping
        early_stopping_callback = lgb.early_stopping(100, verbose=False) # Increased patience

        model.fit(
            train_df[FEATURES], train_df[TARGET],
            eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
            eval_metric="rmse", # Evaluate using RMSE
            callbacks=[early_stopping_callback]
        )

        # Predict on validation set
        preds = model.predict(valid_df[FEATURES])

        # Calculate RMSE on the *transformed* target scale (effectively RMSLE if log1p was used)
        # Or, inverse transform both preds and actuals if you want RMSE on original scale
        if TARGET_TRANSFORM:
             # RMSE on log scale (RMSLE)
             score = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))
        else:
             # RMSE on original scale
             score = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))

        return score

    # Create and run the Optuna study
    study = optuna.create_study(direction="minimize", study_name="LGBM Demand Forecast")
    study.optimize(objective, n_trials=N_TRIALS, timeout=7200) # Add timeout (e.g., 2 hours)

    best_params = study.best_params
    print("\nOptuna finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE/RMSLE): {trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters found by Optuna
    print("\nTraining final model with best parameters...")
    final_model = get_lgbm(best_params)
    final_model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)] # Use early stopping
    )

else:
    # Train with default parameters if tuner is off
    print("\nTraining model with default parameters (no tuning)...")
    final_model = get_lgbm()
    final_model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=True)] # Show early stopping info
    )

print("Model training complete.")
gc.collect()

# --- SHAP Value Analysis ---
print("\nComputing SHAP values for the final model...")
# Use a sample of the validation or training data for faster SHAP calculation if needed
sample_df = valid_df.sample(min(1000, len(valid_df)), random_state=SEED) # Sample 1000 points
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(sample_df[FEATURES])

print("Displaying SHAP summary plots...")
# Bar plot (mean absolute SHAP value - global importance)
shap.summary_plot(shap_values, sample_df[FEATURES], plot_type="bar", show=True)
# Beeswarm plot (shows distribution and feature value impact)
shap.summary_plot(shap_values, sample_df[FEATURES], show=True)
print("SHAP analysis complete.")


# --- Predict on Test Set ---
print("\nPredicting on test set...")
test_predictions_raw = final_model.predict(test_df[FEATURES])

# Inverse transform if target was transformed
if TARGET_TRANSFORM:
    print("Applying inverse transform (expm1) to predictions...")
    test_predictions = np.expm1(test_predictions_raw)
else:
    test_predictions = test_predictions_raw

# Post-processing: Clip predictions (non-negative) and round
test_predictions = np.clip(test_predictions, 0, None).round().astype(int)

# --- Create Submission File ---
print("Creating submission file...")
submission_df = pd.DataFrame({'id': test_df['id'], TARGET: test_predictions})
submission_df['id'] = submission_df['id'].astype(int) # Ensure ID is integer

submission_df.to_csv("submission_v2.csv", index=False)
print("submission_v2.csv saved successfully.")

# --- Optional: Save Model ---
# import joblib
# joblib.dump(final_model, 'lgbm_demand_model_v2.pkl')
# print("Model saved to lgbm_demand_model_v2.pkl")

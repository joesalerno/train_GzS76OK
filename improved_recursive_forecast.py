# Improved Recursive Forecaster
# Based on findings from SHAP importances and recursive_hybrid_forecast.py
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import logging
import lightgbm as lgb

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8
OPTUNA_TRIALS = 50
OPTUNA_STUDY_NAME = "improved_recursive_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "submission_improved_recursive"
SHAP_FILE_PREFIX = "shap_improved_recursive"
N_SHAP_SAMPLES = 2000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Data ---
logging.info("Loading data...")
df = pd.read_csv(DATA_PATH)
test = pd.read_csv(TEST_PATH)
meal_info = pd.read_csv(MEAL_INFO_PATH)
center_info = pd.read_csv(CENTER_INFO_PATH)

def preprocess_data(df, meal_info, center_info):
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

df = preprocess_data(df, meal_info, center_info)
test = preprocess_data(test, meal_info, center_info)
if 'num_orders' not in test.columns:
    test['num_orders'] = np.nan

# --- Feature Engineering ---
GROUP_COLS = ["center_id", "meal_id"]

def create_lag_rolling_features(df, target_col='num_orders', lag_weeks=LAG_WEEKS, rolling_windows=ROLLING_WINDOWS):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)
    return df_out

def create_other_features(df):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = group["checkout_price"].diff()
    for col in OTHER_ROLLING_SUM_COLS:
        shifted = group[col].shift(1)
        df_out[f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}"] = shifted.rolling(OTHER_ROLLING_SUM_WINDOW, min_periods=1).sum().reset_index(drop=True)
    df_out["weekofyear"] = df_out["week"] % 52
    return df_out

def create_interaction_features(df):
    df_out = df.copy()
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
            df_out[name] = 0
    return df_out

def apply_feature_engineering(df, is_test=False):
    df_out = df.copy()
    if not is_test:
        df_out = create_lag_rolling_features(df_out)
    df_out = create_other_features(df_out)
    df_out = create_interaction_features(df_out)
    lag_roll_diff_cols = [col for col in df_out.columns if any(sub in col for sub in ["lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home"])]
    cols_to_fill = [col for col in lag_roll_diff_cols if col in df_out.columns]
    df_out[cols_to_fill] = df_out[cols_to_fill].fillna(0)
    if "discount_pct" in df_out.columns:
        df_out["discount_pct"] = df_out["discount_pct"].fillna(0)
    return df_out

# --- One-hot encoding ---
df_full = pd.concat([df, test], ignore_index=True)
df_full = create_other_features(df_full)
cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in df_full.columns]
if cat_cols:
    df_full = pd.get_dummies(df_full, columns=cat_cols, dummy_na=False)
train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()
train_df = create_lag_rolling_features(train_df)
train_df = create_interaction_features(train_df)
lag_roll_diff_cols = [col for col in train_df.columns if any(sub in col for sub in ["lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home"])]
cols_to_fill = [col for col in lag_roll_diff_cols if col in train_df.columns]
train_df[cols_to_fill] = train_df[cols_to_fill].fillna(0)
if "discount_pct" in train_df.columns:
    train_df["discount_pct"] = train_df["discount_pct"].fillna(0)
train_df = train_df.dropna(subset=['num_orders']).reset_index(drop=True)

# --- Feature Selection (SHAP-driven, robust to missing lags in test) ---
TARGET = "num_orders"
FEATURES = [
    "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion", "discount", "discount_pct", "price_diff", "weekofyear",
    "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"
]
FEATURES += [f"{TARGET}_lag_{lag}" for lag in LAG_WEEKS if f"{TARGET}_lag_{lag}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_mean_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_mean_{w}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_std_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_std_{w}" in train_df.columns]
FEATURES += [f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}" for col in OTHER_ROLLING_SUM_COLS if f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}" in train_df.columns]
FEATURES += [col for col in train_df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]
FEATURES = [f for f in FEATURES if f in train_df.columns and f != TARGET and f !='id']
logging.info(f"Using {len(FEATURES)} features: {FEATURES}")

# --- Train/validation split ---
max_week = train_df["week"].max()
valid_df = train_df[train_df["week"] > max_week - VALIDATION_WEEKS].copy()
train_split_df = train_df[train_df["week"] <= max_week - VALIDATION_WEEKS].copy()

def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    return 'rmsle', rmsle(y_true, y_pred), False

def get_lgbm(params=None):
    default_params = {
        'objective': 'regression_l1',
        'metric': 'None',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
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
        if 'eval_metric' in params and params['eval_metric'] == lgb_rmsle:
            default_params['metric'] = 'None'
    return LGBMRegressor(**default_params)

# --- Optuna Hyperparameter Tuning ---
logging.info("Starting Optuna hyperparameter tuning...")
try:
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Loaded existing Optuna study from {OPTUNA_DB}")
except Exception:
    study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Created new Optuna study at {OPTUNA_DB}")

def objective(trial):
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
    params.update({
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric':'None',
    })
    model = LGBMRegressor(**params)
    model.fit(
        train_split_df[FEATURES], train_split_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric=lgb_rmsle,
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmsle'),
                   lgb.early_stopping(100, verbose=False)]
    )
    preds = model.predict(valid_df[FEATURES])
    score = rmsle(valid_df[TARGET], preds)
    return score

study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=1800)
logging.info(f"Optuna study saved to {OPTUNA_DB}")
best_params = study.best_params
logging.info(f"Best Optuna params: {best_params}")
logging.info(f"Best validation RMSLE: {study.best_value:.5f}")

# --- Final Model Training ---
logging.info("Training final model on full training data with best params...")
final_params = {
    'objective': 'regression_l1',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'metric': 'None'
}
final_params.update(best_params)
final_model = LGBMRegressor(**final_params)
final_model.fit(train_df[FEATURES], train_df[TARGET], eval_metric=lgb_rmsle)

# --- Improved Recursive Prediction ---
logging.info("Starting improved recursive prediction on the test set...")
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test_weeks = sorted(test_df['week'].unique())
for week_num in test_weeks:
    logging.info(f"Predicting for week {week_num}...")
    current_week_mask = history_df['week'] == week_num
    # Smoother: Use rolling means for prediction features, less on raw lags
    history_df = create_lag_rolling_features(history_df, target_col='num_orders')
    history_df = create_interaction_features(history_df)
    lag_roll_diff_cols = [col for col in history_df.columns if any(sub in col for sub in ["lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home"])]
    cols_to_fill = [col for col in lag_roll_diff_cols if col in history_df.columns]
    history_df[cols_to_fill] = history_df[cols_to_fill].fillna(0)
    if "discount_pct" in history_df.columns:
        history_df["discount_pct"] = history_df["discount_pct"].fillna(0)
    current_features = history_df.loc[current_week_mask, FEATURES]
    missing_cols = [col for col in FEATURES if col not in current_features.columns]
    if missing_cols:
        for col in missing_cols:
            current_features[col] = 0
    current_features = current_features[FEATURES]
    # Blend: Use model prediction and rolling mean as a fallback
    model_preds = final_model.predict(current_features)
    rolling_mean_col = f"num_orders_rolling_mean_{ROLLING_WINDOWS[-1]}"
    if rolling_mean_col in current_features.columns:
        fallback = current_features[rolling_mean_col].values
        blended_preds = 0.85 * model_preds + 0.15 * fallback
    else:
        blended_preds = model_preds
    blended_preds = np.clip(blended_preds, 0, None).round().astype(float)
    history_df.loc[current_week_mask, 'num_orders'] = blended_preds
logging.info("Improved recursive prediction finished.")

final_predictions_df = history_df.loc[history_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int)
final_predictions_df['id'] = final_predictions_df['id'].astype(int)
submission_path = f"{SUBMISSION_FILE_PREFIX}_optuna.csv"
final_predictions_df.to_csv(submission_path, index=False)
logging.info(f"Submission file saved to {submission_path}")

# --- SHAP Analysis ---
logging.info("Calculating SHAP values...")
try:
    if len(train_df) > N_SHAP_SAMPLES:
        shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
    else:
        shap_sample = train_df.copy()
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample[FEATURES])
    shap_values_df = pd.DataFrame(shap_values, columns=FEATURES)
    shap_values_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_values.csv", index=False)
    shap_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_feature_importances.csv", index=False)
    plt.figure()
    shap.summary_plot(shap_values, shap_sample[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_summary.png")
    plt.close()
    plt.figure(figsize=(10, 8))
    shap_importance_df.head(20).plot(kind='barh', x='feature', y='mean_abs_shap', legend=False, figsize=(10, 8))
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
    plt.title('Top 20 SHAP Feature Importances (Improved Recursive Optuna Model)')
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_top20_importance.png")
    plt.close()
    logging.info("SHAP analysis saved.")
except Exception as e:
    logging.error(f"Error during SHAP analysis: {e}")

logging.info("Script finished.")

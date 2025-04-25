import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SEED = 42
TARGET = "num_orders"
TARGET_TRANSFORM = True
VALIDATION_WEEKS = 8
N_TRIALS = 40
USE_TUNER = True
OPTUNA_DB = "sqlite:///optuna_lgbm.db"

# --- Data Paths ---
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"

# --- Load Data ---
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
meal_info = pd.read_csv(MEAL_INFO_PATH)
center_info = pd.read_csv(CENTER_INFO_PATH)

# --- Merge Info ---
train = train.merge(meal_info, on="meal_id", how="left")
train = train.merge(center_info, on="center_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")
test = test.merge(center_info, on="center_id", how="left")

# --- Mark train/test ---
train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan
combined = pd.concat([train, test], ignore_index=True, sort=False)
combined = combined.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

# --- Feature Engineering ---
def cyclical_encode(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def create_features(df):
    df = df.copy()
    group = ["center_id", "meal_id"]
    # Lags
    for lag in [1, 2, 3]:
        df[f"num_orders_lag_{lag}"] = df.groupby(group)[TARGET].shift(lag)
    # Moving averages
    for span in [3, 5]:
        df[f"ewma_{span}"] = df.groupby(group)[TARGET].shift(1).ewm(span=span, adjust=False).mean().reset_index(0, drop=True)
    # Rolling mean/std
    for window in [3, 5]:
        shifted = df.groupby(group)[TARGET].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        df[f"rolling_std_{window}"] = shifted.rolling(window, min_periods=2).std().reset_index(0, drop=True)
    # Price features
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = (df["discount"] / df["base_price"]).fillna(0).replace([np.inf, -np.inf], 0)
    df['checkout_price_lag_1'] = df.groupby(group)["checkout_price"].shift(1)
    df['price_diff'] = df['checkout_price'] - df['checkout_price_lag_1']
    # Promotion rolling sum
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted_promo = df.groupby(group)[col].shift(1)
        df[f"{col}_rolling_sum_3"] = shifted_promo.rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    # Cyclical encoding for weekofyear
    df["weekofyear"] = df["week"] % 52
    df = cyclical_encode(df, "weekofyear", 52)
    # Fill NaNs
    lag_roll_cols = [col for col in df.columns if any(x in col for x in ["lag", "rolling", "ewma", "diff"])]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    return df

combined = create_features(combined)

# --- Target Transformation ---
if TARGET_TRANSFORM:
    train_mask = combined['is_train'] == 1
    combined.loc[train_mask, TARGET] = np.log1p(combined.loc[train_mask, TARGET])

# --- One-hot Encoding (pruned) ---
cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in combined.columns and combined[col].dtype == 'object']
if cat_cols:
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)

# --- Feature List (pruned) ---
FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", 'checkout_price_lag_1',
    "op_area", "city_code", "region_code",
    "weekofyear_sin", "weekofyear_cos"
] + [f"num_orders_lag_{lag}" for lag in [1,2,3]] \
  + [f"ewma_{span}" for span in [3,5]] \
  + [f"rolling_mean_{w}" for w in [3,5]] \
  + [f"rolling_std_{w}" for w in [3,5]] \
  + [f"{col}_rolling_sum_3" for col in ["emailer_for_promotion", "homepage_featured"]]
# Add one-hot columns for top categories only (based on previous SHAP)
ohe_cols = [col for col in combined.columns if any(col.startswith(prefix + "_") for prefix in ["cuisine_Indian", "category_Rice Bowl", "center_type_Type A"])]
FEATURES += ohe_cols
FEATURES = [f for f in FEATURES if f in combined.columns and f != TARGET and f != 'is_train' and f != 'id']

# Remove zero-variance features
for f in FEATURES.copy():
    if combined[f].nunique() <= 1:
        FEATURES.remove(f)

# --- Split Data ---
max_train_week = combined[combined['is_train'] == 1]['week'].max()
validation_start_week = max_train_week - VALIDATION_WEEKS + 1
train_df = combined[(combined['is_train'] == 1) & (combined['week'] < validation_start_week)].reset_index(drop=True)
valid_df = combined[(combined['is_train'] == 1) & (combined['week'] >= validation_start_week)].reset_index(drop=True)
test_df = combined[combined['is_train'] == 0].reset_index(drop=True)

# --- Model ---
def get_lgbm(params=None):
    default_params = dict(
        objective="regression_l1" if TARGET_TRANSFORM else "regression",
        metric="rmse",
        learning_rate=0.05,
        n_estimators=2000,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    if params:
        default_params.update(params)
    return LGBMRegressor(**default_params)

# --- Optuna Tuning (resume) ---
if USE_TUNER:
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
        }
        model = get_lgbm(params)
        model.fit(
            train_df[FEATURES], train_df[TARGET],
            eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        preds = model.predict(valid_df[FEATURES])
        score = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))
        return score
    study_name = "LGBM_Ultimate_Forecast"
    storage = OPTUNA_DB
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print("Resuming Optuna study...")
    except Exception:
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage)
        print("Created new Optuna study.")
    study.optimize(objective, n_trials=N_TRIALS, timeout=7200)
    best_params = study.best_params
    print("Best trial value:", study.best_trial.value)
    print("Best params:", best_params)
    # Save Optuna results to CSV
    df_trials = study.trials_dataframe()
    df_trials.to_csv("optuna_ultimate_trials.csv", index=False)
    final_model = get_lgbm(best_params)
    final_model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
else:
    final_model = get_lgbm()
    final_model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )

# --- Predict on Test Set ---
test_predictions_raw = final_model.predict(test_df[FEATURES])
if TARGET_TRANSFORM:
    test_predictions = np.expm1(test_predictions_raw)
else:
    test_predictions = test_predictions_raw
test_predictions = np.clip(test_predictions, 0, None).round().astype(int)

submission_df = pd.DataFrame({'id': test_df['id'], TARGET: test_predictions})
submission_df['id'] = submission_df['id'].astype(int)
submission_df.to_csv("submission_ultimate.csv", index=False)
print("submission_ultimate.csv saved.")

# --- SHAP Analysis ---
print("Calculating SHAP values...")
shap_sample = train_df[FEATURES].sample(n=min(2000, len(train_df)), random_state=SEED)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(shap_sample)
# Save SHAP values to CSV
shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df.to_csv("shap_ultimate_values.csv", index=False)
# Save SHAP feature importances to CSV
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv("shap_ultimate_feature_importances.csv", index=False)
# Save SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("shap_ultimate_summary.png")
plt.close()
print("SHAP analysis saved to shap_ultimate_values.csv, shap_ultimate_feature_importances.csv, shap_ultimate_summary.png.")

# --- Feature Importance Chart ---
plt.figure(figsize=(10, 6))
shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
plt.title('Top 20 SHAP Feature Importances')
plt.ylabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('shap_ultimate_top20.png')
plt.close()

# --- Validation Report ---
valid_preds = final_model.predict(valid_df[FEATURES])
if TARGET_TRANSFORM:
    valid_preds = np.expm1(valid_preds)
    valid_true = np.expm1(valid_df[TARGET])
else:
    valid_true = valid_df[TARGET]
rmse = np.sqrt(np.mean((valid_preds - valid_true) ** 2))
mae = np.mean(np.abs(valid_preds - valid_true))
report = pd.DataFrame({
    'RMSE': [rmse],
    'MAE': [mae],
    'Validation Weeks': [VALIDATION_WEEKS],
    'Train Rows': [len(train_df)],
    'Valid Rows': [len(valid_df)],
    'Features Used': [len(FEATURES)]
})
report.to_csv('ultimate_validation_report.csv', index=False)
print(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
print("Validation report saved to ultimate_validation_report.csv.")

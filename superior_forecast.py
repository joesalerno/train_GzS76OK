import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# --- Configuration ---
SEED = 42
TARGET = "num_orders"
VALIDATION_WEEKS = 8
N_TRIALS = 2
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
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
def create_features(df):
    df = df.copy()
    group = ["center_id", "meal_id"]
    # Lags
    for lag in LAG_WEEKS:
        df[f"num_orders_lag_{lag}"] = df.groupby(group)[TARGET].shift(lag)
    # Rolling means/stds
    for window in ROLLING_WINDOWS:
        shifted = df.groupby(group)[TARGET].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        df[f"rolling_std_{window}"] = shifted.rolling(window, min_periods=2).std().reset_index(0, drop=True)
    # Price features
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = (df["discount"] / df["base_price"]).fillna(0).replace([np.inf, -np.inf], 0)
    df["price_diff"] = df.groupby(group)["checkout_price"].diff().fillna(0)
    # Promotion rolling sum
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted = df.groupby(group)[col].shift(1)
        df[f"{col}_rolling_sum_3"] = shifted.rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    # Time features
    df["weekofyear"] = df["week"] % 52
    # --- New interaction features ---
    df["price_diff_x_email"] = df["price_diff"] * df["emailer_for_promotion"]
    df["lag1_x_email"] = df["num_orders_lag_1"] * df["emailer_for_promotion"]
    # --- Aggregate features ---
    df["meal_mean"] = df.groupby("meal_id")[TARGET].transform("mean")
    df["center_mean"] = df.groupby("center_id")[TARGET].transform("mean")
    # Fill NaNs for lags/rolling/price_diff
    lag_roll_cols = [col for col in df.columns if any(x in col for x in ["lag", "rolling", "diff"])]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    return df

combined = create_features(combined)

# --- One-hot Encoding ---
cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in combined.columns and combined[col].dtype == 'object']
if cat_cols:
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)

# --- Feature List ---
FEATURES = [
    "num_orders_lag_1", "num_orders_lag_2", "num_orders_lag_3", "num_orders_lag_5", "num_orders_lag_10",
    "rolling_mean_3", "rolling_mean_5", "rolling_mean_10",
    "rolling_std_3", "rolling_std_5", "rolling_std_10",
    "discount", "discount_pct", "price_diff",
    "emailer_for_promotion", "homepage_featured",
    "emailer_for_promotion_rolling_sum_3", "homepage_featured_rolling_sum_3",
    "weekofyear",
    "price_diff_x_email", "lag1_x_email",
    "meal_mean", "center_mean"
]
# Add one-hot columns
ohe_cols = [col for col in combined.columns if any(col.startswith(prefix + "_") for prefix in ["category", "cuisine", "center_type"])]
FEATURES += ohe_cols
FEATURES = [f for f in FEATURES if f in combined.columns]

# --- Split Data ---
max_train_week = combined[combined['is_train'] == 1]['week'].max()
validation_start_week = max_train_week - VALIDATION_WEEKS + 1
train_df = combined[(combined['is_train'] == 1) & (combined['week'] < validation_start_week)].reset_index(drop=True)
valid_df = combined[(combined['is_train'] == 1) & (combined['week'] >= validation_start_week)].reset_index(drop=True)
test_df = combined[combined['is_train'] == 0].reset_index(drop=True)

# --- Model ---
def get_lgbm(params=None):
    default_params = dict(
        objective="regression",
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
    study_name = "LGBM_Superior_Forecast"
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
    df_trials.to_csv("optuna_superior_trials.csv", index=False)
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

# --- Offset Correction ---
valid_preds = final_model.predict(valid_df[FEATURES])
valid_true = valid_df[TARGET]
offsets = np.linspace(-20, 20, 201)
rmse_scores = [np.sqrt(np.mean((valid_preds + o - valid_true) ** 2)) for o in offsets]
best_offset = offsets[np.argmin(rmse_scores)]
print(f"Best offset: {best_offset:.2f}, RMSE: {min(rmse_scores):.4f}")

# --- Row-wise Test Feature Creation ---
def make_test_features(test, train_hist):
    test_feat = test.copy()
    for lag in LAG_WEEKS:
        test_feat[f"num_orders_lag_{lag}"] = np.nan
    for window in ROLLING_WINDOWS:
        test_feat[f"rolling_mean_{window}"] = np.nan
        test_feat[f"rolling_std_{window}"] = np.nan
    test_feat["price_diff"] = np.nan
    for col in ["emailer_for_promotion", "homepage_featured"]:
        test_feat[f"{col}_rolling_sum_3"] = np.nan
    test_feat["weekofyear"] = test_feat["week"] % 52
    test_feat["price_diff_x_email"] = np.nan
    test_feat["lag1_x_email"] = np.nan
    test_feat["meal_mean"] = np.nan
    test_feat["center_mean"] = np.nan
    for idx, row in test_feat.iterrows():
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        for lag in LAG_WEEKS:
            test_feat.at[idx, f"num_orders_lag_{lag}"] = hist["num_orders"].iloc[-lag] if len(hist) >= lag else 0
        for window in ROLLING_WINDOWS:
            vals = hist["num_orders"].iloc[-window:] if len(hist) >= window else hist["num_orders"]
            test_feat.at[idx, f"rolling_mean_{window}"] = vals.mean() if len(vals) > 0 else 0
            test_feat.at[idx, f"rolling_std_{window}"] = vals.std() if len(vals) > 1 else 0
        test_feat.at[idx, "price_diff"] = row["checkout_price"] - hist["checkout_price"].iloc[-1] if len(hist) >= 1 else 0
        for col in ["emailer_for_promotion", "homepage_featured"]:
            vals = hist[col].iloc[-3:] if len(hist) >= 3 else hist[col]
            test_feat.at[idx, f"{col}_rolling_sum_3"] = vals.sum() if len(vals) > 0 else 0
        test_feat.at[idx, "price_diff_x_email"] = test_feat.at[idx, "price_diff"] * row["emailer_for_promotion"]
        test_feat.at[idx, "lag1_x_email"] = test_feat.at[idx, "num_orders_lag_1"] * row["emailer_for_promotion"]
        test_feat.at[idx, "meal_mean"] = hist["num_orders"].mean() if len(hist) > 0 else 0
        test_feat.at[idx, "center_mean"] = hist["num_orders"].mean() if len(hist) > 0 else 0
    return test_feat

# --- Predict on Test Set ---
test_feat = make_test_features(test_df, pd.concat([train_df, valid_df]))
test_feat, _ = test_feat.align(train_df, join="right", axis=1, fill_value=0)
test_predictions = final_model.predict(test_feat[FEATURES]) + best_offset
test_predictions = np.clip(test_predictions, 0, None).round().astype(int)

submission_df = pd.DataFrame({'id': test_feat['id'], TARGET: test_predictions})
submission_df['id'] = submission_df['id'].astype(int)
submission_df.to_csv("submission_superior.csv", index=False)
print("submission_superior.csv saved.")

# --- SHAP Analysis ---
print("Calculating SHAP values...")
shap_sample = train_df[FEATURES].sample(n=min(2000, len(train_df)), random_state=SEED)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(shap_sample)
# Save SHAP values to CSV
shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df.to_csv("shap_superior_values.csv", index=False)
# Save SHAP feature importances to CSV
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv("shap_superior_feature_importances.csv", index=False)
# Save SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("shap_superior_summary.png")
plt.close()
# Save SHAP top 20 bar plot
plt.figure(figsize=(10, 6))
shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
plt.title('Top 20 SHAP Feature Importances (Superior Model)')
plt.ylabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('shap_superior_top20.png')
plt.close()
print("SHAP analysis saved to shap_superior_values.csv, shap_superior_feature_importances.csv, shap_superior_summary.png, shap_superior_top20.png.")

# --- Validation Report ---
valid_preds = final_model.predict(valid_df[FEATURES]) + best_offset
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
report.to_csv('superior_validation_report.csv', index=False)
print(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
print("Validation report saved to superior_validation_report.csv.")

# --- Greedy Backward Feature Elimination ---
def greedy_feature_selection(train_df, valid_df, FEATURES, TARGET, get_lgbm, best_offset):
    import copy
    best_features = copy.deepcopy(FEATURES)
    best_rmse = None
    improving = True
    while improving and len(best_features) > 1:
        improving = False
        scores = []
        for f in best_features:
            trial_features = [x for x in best_features if x != f]
            model = get_lgbm()
            model.fit(
                train_df[trial_features], train_df[TARGET],
                eval_set=[(valid_df[trial_features], valid_df[TARGET])],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            preds = model.predict(valid_df[trial_features]) + best_offset
            rmse = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))
            scores.append((rmse, f))
        scores.sort()
        if best_rmse is None:
            # Initial RMSE with all features
            model = get_lgbm()
            model.fit(
                train_df[best_features], train_df[TARGET],
                eval_set=[(valid_df[best_features], valid_df[TARGET])],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            preds = model.predict(valid_df[best_features]) + best_offset
            best_rmse = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))
        # Try removing the feature that gives the best (lowest) RMSE
        best_candidate_rmse, feature_to_remove = scores[0]
        if best_candidate_rmse <= best_rmse:
            print(f"Removing feature: {feature_to_remove} (RMSE improved from {best_rmse:.4f} to {best_candidate_rmse:.4f})")
            best_features.remove(feature_to_remove)
            best_rmse = best_candidate_rmse
            improving = True
        else:
            print(f"No further improvement by removing any single feature. Stopping.")
    print(f"Best feature set: {best_features}")
    print(f"Best validation RMSE: {best_rmse:.4f}")
    return best_features, best_rmse

# --- Run feature selection and retrain final model ---
best_features, best_rmse = greedy_feature_selection(train_df, valid_df, FEATURES, TARGET, get_lgbm, best_offset)
final_model = get_lgbm()
final_model.fit(
    train_df[best_features], train_df[TARGET],
    eval_set=[(valid_df[best_features], valid_df[TARGET])],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# --- Predict on Test Set with best features ---
test_feat = make_test_features(test_df, pd.concat([train_df, valid_df]))
test_feat, _ = test_feat.align(train_df, join="right", axis=1, fill_value=0)
test_predictions = final_model.predict(test_feat[best_features]) + best_offset
test_predictions = np.clip(test_predictions, 0, None).round().astype(int)

submission_df = pd.DataFrame({'id': test_feat['id'], TARGET: test_predictions})
submission_df['id'] = submission_df['id'].astype(int)
submission_df.to_csv("submission_superior_greedy.csv", index=False)
print("submission_superior_greedy.csv saved.")

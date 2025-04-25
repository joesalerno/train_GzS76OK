import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1]
ROLLING_WINDOWS = [10]

# --- Load data ---
df = pd.read_csv(DATA_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test = pd.read_csv(TEST_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
meal_info = pd.read_csv(MEAL_INFO_PATH)
df = df.merge(meal_info, on="meal_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")
center_info = pd.read_csv(CENTER_INFO_PATH)
df = df.merge(center_info, on="center_id", how="left")
test = test.merge(center_info, on="center_id", how="left")

# --- Feature engineering (minimal + interactions) ---
def create_features(df):
    df = df.copy()
    group = ["center_id", "meal_id"]
    # Lags
    df["num_orders_lag_1"] = df.groupby(group)["num_orders"].shift(1)
    # Rolling mean
    shifted = df.groupby(group)["num_orders"].shift(1)
    df["rolling_mean_10"] = shifted.rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    # Price features
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = df["discount"] / df["base_price"]
    df["checkout_price_lag_1"] = df.groupby(group)["checkout_price"].shift(1)
    df["price_diff"] = df["checkout_price"] - df["checkout_price_lag_1"]
    # Promotion features
    df["emailer_for_promotion"] = df["emailer_for_promotion"]
    df["homepage_featured"] = df["homepage_featured"]
    # Time features
    df["weekofyear"] = df["week"] % 52
    # --- Interaction features ---
    df["price_diff_x_emailer"] = df["price_diff"] * df["emailer_for_promotion"]
    df["lag1_x_emailer"] = df["num_orders_lag_1"] * df["emailer_for_promotion"]
    df["price_diff_x_home"] = df["price_diff"] * df["homepage_featured"]
    df["lag1_x_home"] = df["num_orders_lag_1"] * df["homepage_featured"]
    # Fill NaNs
    lag_roll_cols = ["num_orders_lag_1", "rolling_mean_10", "price_diff", "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    return df

df = create_features(df)
df = df[df["num_orders"].notna()].reset_index(drop=True)

# --- Minimal one-hot encoding (keep only top categories/cuisines/center_types if present) ---
cat_cols = []
if "category" in df.columns:
    cat_cols.append("category")
if "cuisine" in df.columns:
    cat_cols.append("cuisine")
if "center_type" in df.columns:
    cat_cols.append("center_type")
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols)
    test = pd.get_dummies(test, columns=cat_cols)
df, test = df.align(test, join="left", axis=1, fill_value=0)

# --- Feature list (minimal + interactions) ---
FEATURES = [
    "num_orders_lag_1", "rolling_mean_10", "emailer_for_promotion", "homepage_featured", "price_diff",
    "discount_pct", "weekofyear", "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"
]
# Add one-hot columns if present
FEATURES += [col for col in df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET = "num_orders"

# --- Train/validation split (last 8 weeks for validation) ---
max_week = df["week"].max()
valid_df = df[df["week"] > max_week - 8].copy()
train_df = df[df["week"] <= max_week - 8].copy()

# --- RMSLE metric ---
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred.clip(0)) - np.log1p(y_true.clip(0)))))

def lgb_rmsle(y_true, y_pred):
    rmsle_score = rmsle(y_true, y_pred)
    return 'rmsle', rmsle_score, False

# --- Greedy backward feature elimination ---
def greedy_feature_selection(train_df, valid_df, FEATURES, TARGET, get_lgbm):
    best_features = FEATURES.copy()
    best_rmsle = float('inf')
    improved = True
    while improved and len(best_features) > 1:
        improved = False
        scores = []
        for f in best_features:
            trial_features = [x for x in best_features if x != f]
            model = get_lgbm()
            model.fit(
                train_df[trial_features], train_df[TARGET],
                eval_set=[(valid_df[trial_features], valid_df[TARGET])],
                eval_metric=lgb_rmsle
            )
            preds = model.predict(valid_df[trial_features])
            score = rmsle(valid_df[TARGET], preds)
            scores.append((score, f))
        scores.sort()
        if scores[0][0] < best_rmsle:
            best_rmsle = scores[0][0]
            best_features.remove(scores[0][1])
            improved = True
            print(f"Removed {scores[0][1]}, new best RMSLE: {best_rmsle:.5f}")
    return best_features, best_rmsle

# --- Model training function ---
def get_lgbm(params=None):
    default_params = dict(
        objective="regression",
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_child_samples=20,
        lambda_l1=0.1,
        lambda_l2=0.1,
        max_depth=5,
        random_state=SEED,
        verbose=-1
    )
    if params:
        default_params.update(params)
    return LGBMRegressor(**default_params)

# --- Run greedy feature selection ---
best_features, best_rmsle = greedy_feature_selection(train_df, valid_df, FEATURES, TARGET, get_lgbm)
print(f"Best feature subset: {best_features}\nBest validation RMSLE: {best_rmsle:.5f}")

# --- Retrain final model on best features ---
model = get_lgbm()
model.fit(
    train_df[best_features], train_df[TARGET],
    eval_set=[(valid_df[best_features], valid_df[TARGET])],
    eval_metric=lgb_rmsle
)
preds_valid = model.predict(valid_df[best_features])
val_rmsle = rmsle(valid_df[TARGET], preds_valid)
print(f"Final validation RMSLE with selected features: {val_rmsle:.5f}")

# --- Retrain on full data and predict test set ---
# Recreate features for test set
def create_features_test(test, train_hist):
    test_feat = test.copy()
    group = ["center_id", "meal_id"]
    test_feat["num_orders"] = np.nan
    test_feat["num_orders_lag_1"] = np.nan
    test_feat["rolling_mean_10"] = np.nan
    test_feat["checkout_price_lag_1"] = np.nan
    test_feat["price_diff"] = np.nan
    for idx, row in test_feat.iterrows():
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        test_feat.at[idx, "num_orders_lag_1"] = hist["num_orders"].iloc[-1] if len(hist) >= 1 else 0
        vals = hist["num_orders"].iloc[-10:] if len(hist) >= 1 else hist["num_orders"]
        test_feat.at[idx, "rolling_mean_10"] = vals.mean() if len(vals) > 0 else 0
        test_feat.at[idx, "checkout_price_lag_1"] = hist["checkout_price"].iloc[-1] if len(hist) >= 1 else row["checkout_price"]
        test_feat.at[idx, "price_diff"] = row["checkout_price"] - test_feat.at[idx, "checkout_price_lag_1"]
    test_feat["discount"] = test_feat["base_price"] - test_feat["checkout_price"]
    test_feat["discount_pct"] = test_feat["discount"] / test_feat["base_price"]
    test_feat["weekofyear"] = test_feat["week"] % 52
    test_feat["price_diff_x_emailer"] = test_feat["price_diff"] * test_feat["emailer_for_promotion"]
    test_feat["lag1_x_emailer"] = test_feat["num_orders_lag_1"] * test_feat["emailer_for_promotion"]
    test_feat["price_diff_x_home"] = test_feat["price_diff"] * test_feat["homepage_featured"]
    test_feat["lag1_x_home"] = test_feat["num_orders_lag_1"] * test_feat["homepage_featured"]
    # Minimal one-hot encoding
    cat_cols = []
    if "category" in test_feat.columns:
        cat_cols.append("category")
    if "cuisine" in test_feat.columns:
        cat_cols.append("cuisine")
    if "center_type" in test_feat.columns:
        cat_cols.append("center_type")
    if cat_cols:
        test_feat = pd.get_dummies(test_feat, columns=cat_cols)
    test_feat, _ = test_feat.align(df, join="right", axis=1, fill_value=0)
    # Fill NaNs
    lag_roll_cols = ["num_orders_lag_1", "rolling_mean_10", "price_diff", "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"]
    test_feat[lag_roll_cols] = test_feat[lag_roll_cols].fillna(0)
    return test_feat

# Retrain on all data (train+valid)
full_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
model = get_lgbm()
model.fit(full_df[best_features], full_df[TARGET], eval_metric=lgb_rmsle)

# Prepare test features and predict
test_feat = create_features_test(test, full_df)
test_feat["num_orders"] = np.clip(model.predict(test_feat[best_features]), 0, None).round().astype(int)
submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission_minimal_interaction.csv", index=False)
print("submission_minimal_interaction.csv saved.")

# --- SHAP analysis for interpretability ---
print("Calculating SHAP values for minimal interaction model...")
shap_sample = full_df[best_features].sample(n=min(2000, len(full_df)), random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_sample)
shap_df = pd.DataFrame(shap_values, columns=best_features)
shap_df.to_csv("shap_minimal_interaction_values.csv", index=False)
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': best_features,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv("shap_minimal_interaction_feature_importances.csv", index=False)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=best_features, show=False)
plt.tight_layout()
plt.savefig("shap_minimal_interaction_summary.png")
plt.close()
plt.figure(figsize=(10, 6))
shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
plt.title('Top 20 SHAP Feature Importances (Minimal Interaction Model)')
plt.ylabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('shap_minimal_interaction_top20.png')
plt.close()
print("SHAP analysis for minimal interaction model saved.")

# --- Optuna tuning for further improvement ---
import lightgbm as lgb
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    model = get_lgbm(params)
    model.fit(
        train_df[best_features], train_df[TARGET],
        eval_set=[(valid_df[best_features], valid_df[TARGET])],
        eval_metric=lgb_rmsle
    )
    preds = model.predict(valid_df[best_features])
    score = rmsle(valid_df[TARGET], preds)
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("Best Optuna params:", best_params)

# Retrain with best params on full data
model = get_lgbm(best_params)
model.fit(full_df[best_features], full_df[TARGET], eval_metric=lgb_rmsle)
test_feat["num_orders"] = np.clip(model.predict(test_feat[best_features]), 0, None).round().astype(int)
submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission_minimal_interaction_optuna.csv", index=False)
print("submission_minimal_interaction_optuna.csv saved.")
# SHAP for Optuna-tuned model
shap_sample = full_df[best_features].sample(n=min(2000, len(full_df)), random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_sample)
shap_df = pd.DataFrame(shap_values, columns=best_features)
shap_df.to_csv("shap_minimal_interaction_optuna_values.csv", index=False)
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': best_features,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv("shap_minimal_interaction_optuna_feature_importances.csv", index=False)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=best_features, show=False)
plt.tight_layout()
plt.savefig("shap_minimal_interaction_optuna_summary.png")
plt.close()
plt.figure(figsize=(10, 6))
shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
plt.title('Top 20 SHAP Feature Importances (Minimal Interaction Optuna Model)')
plt.ylabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('shap_minimal_interaction_optuna_top20.png')
plt.close()
print("SHAP analysis for Optuna-tuned minimal interaction model saved.")

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import optuna
import shap
import matplotlib.pyplot as plt

USE_TUNER = True  # Set to False to skip tuning

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
SEED = 42

# --- Load data ---
df = pd.read_csv(DATA_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test = pd.read_csv(TEST_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
meal_info = pd.read_csv(MEAL_INFO_PATH)
df = df.merge(meal_info, on="meal_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")
center_info = pd.read_csv(CENTER_INFO_PATH)
df = df.merge(center_info, on="center_id", how="left")
test = test.merge(center_info, on="center_id", how="left")

# --- Feature engineering (simple, robust) ---
def create_features(df):
    df = df.copy()
    for lag in LAG_WEEKS:
        df[f"num_orders_lag_{lag}"] = df.groupby(["center_id", "meal_id"])["num_orders"].shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = df.groupby(["center_id", "meal_id"])["num_orders"].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean().reset_index(0, drop=True)
        df[f"rolling_std_{window}"] = shifted.rolling(window).std().reset_index(0, drop=True)
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = df["discount"] / df["base_price"]
    df["price_diff"] = df.groupby(["center_id", "meal_id"])["checkout_price"].diff()
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted = df.groupby(["center_id", "meal_id"])[col].shift(1)
        df[f"{col}_rolling_sum_3"] = shifted.rolling(3).sum().reset_index(0, drop=True)
    df["weekofyear"] = df["week"] % 52
    return df

df = create_features(df)
df = df[df["num_orders"].notna()].reset_index(drop=True)

# --- One-hot encode categorical columns if present ---
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
# Align columns between train and test
df, test = df.align(test, join="left", axis=1, fill_value=0)

# --- Feature list ---
FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", "weekofyear"
]
FEATURES += [f"num_orders_lag_{lag}" for lag in LAG_WEEKS]
FEATURES += [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
FEATURES += [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
FEATURES += [f"{col}_rolling_sum_3" for col in ["emailer_for_promotion", "homepage_featured"]]
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

# --- Model training ---
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

if USE_TUNER:
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
            train_df[FEATURES], train_df[TARGET],
            eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
            eval_metric=lgb_rmsle,
            callbacks=[
                lgb.early_stopping(50, verbose=False)
            ]
        )
        preds = model.predict(valid_df[FEATURES])
        score = rmsle(valid_df[TARGET], preds)
        return score

    import lightgbm as lgb
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print("Best params:", best_params)
    model = get_lgbm(best_params)
    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric=lgb_rmsle,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
else:
    model = get_lgbm()
    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric=lgb_rmsle
    )

# --- Validation RMSLE reporting ---
preds_valid = model.predict(valid_df[FEATURES])
val_rmsle = rmsle(valid_df[TARGET], preds_valid)
print(f"Validation RMSLE: {val_rmsle:.5f}")

# --- SHAP Analysis ---
print("Calculating SHAP values for old model...")
shap_sample = train_df[FEATURES].sample(n=min(2000, len(train_df)), random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_sample)
# Save SHAP values to CSV
shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df.to_csv("shap_old_values.csv", index=False)
# Save SHAP feature importances to CSV
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv("shap_old_feature_importances.csv", index=False)
# Save SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("shap_old_summary.png")
plt.close()
# Save SHAP top 20 bar plot
plt.figure(figsize=(10, 6))
shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
plt.title('Top 20 SHAP Feature Importances (Old Model)')
plt.ylabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('shap_old_top20.png')
plt.close()
print("SHAP analysis for old model saved to shap_old_values.csv, shap_old_feature_importances.csv, shap_old_summary.png, shap_old_top20.png.")

# --- Predict on test set ---
def make_test_features(test, train_hist):
    test_feat = test.copy()
    test_feat["num_orders"] = np.nan
    for lag in LAG_WEEKS:
        test_feat[f"num_orders_lag_{lag}"] = np.nan
    for window in ROLLING_WINDOWS:
        test_feat[f"rolling_mean_{window}"] = np.nan
        test_feat[f"rolling_std_{window}"] = np.nan
    test_feat["price_diff"] = np.nan
    for col in ["emailer_for_promotion", "homepage_featured"]:
        test_feat[f"{col}_rolling_sum_3"] = np.nan
    test_feat["weekofyear"] = test_feat["week"] % 52
    # Fill lags and rolling features from train_hist
    for idx, row in test_feat.iterrows():
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        for lag in LAG_WEEKS:
            test_feat.at[idx, f"num_orders_lag_{lag}"] = hist["num_orders"].iloc[-lag] if len(hist) >= lag else np.nan
        for window in ROLLING_WINDOWS:
            vals = hist["num_orders"].iloc[-window:] if len(hist) >= window else hist["num_orders"]
            test_feat.at[idx, f"rolling_mean_{window}"] = vals.mean() if len(vals) > 0 else np.nan
            test_feat.at[idx, f"rolling_std_{window}"] = vals.std() if len(vals) > 1 else np.nan
        test_feat.at[idx, "price_diff"] = row["checkout_price"] - hist["checkout_price"].iloc[-1] if len(hist) >= 1 else np.nan
        for col in ["emailer_for_promotion", "homepage_featured"]:
            vals = hist[col].iloc[-3:] if len(hist) >= 3 else hist[col]
            test_feat.at[idx, f"{col}_rolling_sum_3"] = vals.sum() if len(vals) > 0 else np.nan
    return test_feat

test_feat = make_test_features(test, df)
# One-hot encode if needed, but only if columns exist
cat_cols_in_test = [col for col in cat_cols if col in test_feat.columns]
if cat_cols_in_test:
    test_feat = pd.get_dummies(test_feat, columns=cat_cols_in_test)
test_feat, _ = test_feat.align(df, join="right", axis=1, fill_value=0)

test_feat["num_orders"] = np.clip(model.predict(test_feat[FEATURES]), 0, None).round().astype(int)

submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
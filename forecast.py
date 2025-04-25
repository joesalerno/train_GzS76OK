import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
FORECAST_HORIZON = 10
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
SEED = 42
OPTUNA_STORAGE = "sqlite:///optuna_lgbm.db"
OPTUNA_STUDY_NAME = "lgbm_forecast"

# --- Detect GPU support for LightGBM ---
def detect_lgbm_device():
    try:
        # Try to train a tiny model with device='gpu'
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        test_model = LGBMRegressor(device='gpu', n_estimators=1)
        test_model.fit(X, y)
        print("LightGBM GPU support detected. Using GPU.")
        return "gpu"
    except Exception:
        print("LightGBM GPU support not detected. Using CPU.")
        return "cpu"

LGBM_DEVICE = detect_lgbm_device()

# Load main data
df = pd.read_csv(DATA_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test = pd.read_csv(TEST_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

# Load and merge meal info
meal_info = pd.read_csv(MEAL_INFO_PATH)
df = df.merge(meal_info, on="meal_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")

# Load and merge center info
center_info = pd.read_csv(CENTER_INFO_PATH)
df = df.merge(center_info, left_on="center_id", right_on="center_id", how="left")
test = test.merge(center_info, left_on="center_id", right_on="center_id", how="left")

def create_features(df):
    df = df.copy()
    # Lags
    for lag in LAG_WEEKS:
        df[f"num_orders_lag_{lag}"] = df.groupby(["center_id", "meal_id"])["num_orders"].shift(lag)
    # Rolling stats
    for window in ROLLING_WINDOWS:
        shifted = df.groupby(["center_id", "meal_id"])["num_orders"].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean().reset_index(0, drop=True)
        df[f"rolling_std_{window}"] = shifted.rolling(window).std().reset_index(0, drop=True)
        df[f"rolling_min_{window}"] = shifted.rolling(window).min().reset_index(0, drop=True)
        df[f"rolling_max_{window}"] = shifted.rolling(window).max().reset_index(0, drop=True)
        df[f"rolling_median_{window}"] = shifted.rolling(window).median().reset_index(0, drop=True)
    # Differencing for stationarity
    df["num_orders_diff_1"] = df.groupby(["center_id", "meal_id"])["num_orders"].diff(1)
    df["num_orders_diff_2"] = df.groupby(["center_id", "meal_id"])["num_orders"].diff(2)
    df["checkout_price_diff_1"] = df.groupby(["center_id", "meal_id"])["checkout_price"].diff(1)
    # Log transforms
    df["log_num_orders"] = np.log1p(df["num_orders"])
    df["log_checkout_price"] = np.log1p(df["checkout_price"])
    # Discount features
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = df["discount"] / df["base_price"]
    df["price_diff"] = df.groupby(["center_id", "meal_id"])["checkout_price"].diff()
    # Promotion rolling sums
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted = df.groupby(["center_id", "meal_id"])[col].shift(1)
        df[f"{col}_rolling_sum_3"] = shifted.rolling(3).sum().reset_index(0, drop=True)
    # Cumulative stats
    df["cummean_num_orders"] = df.groupby(["center_id", "meal_id"])["num_orders"].expanding().mean().reset_index(level=[0,1], drop=True)
    df["cumsum_num_orders"] = df.groupby(["center_id", "meal_id"])["num_orders"].cumsum()
    # Interaction features
    df["discount_x_promo"] = df["discount"] * df["emailer_for_promotion"]
    df["discount_x_homepage"] = df["discount"] * df["homepage_featured"]
    df["price_ratio"] = df["checkout_price"] / (df["base_price"] + 1e-3)
    # Week features
    df["weekofyear"] = df["week"] % 52
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    # Target encoding for category/cuisine/center_type
    for col in ["category", "cuisine", "center_type"]:
        if col in df.columns:
            means = df.groupby(col)["num_orders"].transform("mean")
            df[f"{col}_target_enc"] = means
    return df

df = create_features(df).dropna().reset_index(drop=True)

# Add new features from meal_info and center_info
cat_cols = []
if "category" in df.columns:
    cat_cols.append("category")
if "cuisine" in df.columns:
    cat_cols.append("cuisine")
if "center_type" in df.columns:
    cat_cols.append("center_type")

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=cat_cols)
test = pd.get_dummies(test, columns=cat_cols)
# Align columns between train and test
df, test = df.align(test, join="left", axis=1, fill_value=0)

FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured",
    "discount", "discount_pct", "price_diff", "weekofyear"
] + [col for col in df.columns if "lag_" in col or "rolling_" in col] \
  + [col for col in df.columns if col.startswith("category_") or col.startswith("cuisine_") or col.startswith("center_type_")] \
  + [col for col in df.columns if col in center_info.columns and col != "center_id"]

FEATURES = list(dict.fromkeys(FEATURES))
FEATURES = [f for f in FEATURES if f in df.columns and f != "num_orders"]

TARGET = "num_orders"

# --- Optuna hyperparameter tuning with KFold and more params ---
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": 1,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 1),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 20000, 300000),
        "random_state": SEED,
        "verbose": -1,
        "device": LGBM_DEVICE,
    }
    threshold = trial.suggest_categorical("feature_importance_threshold", [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmses = []
    for train_idx, valid_idx in kf.split(df):
        X_train, X_valid = df.iloc[train_idx][FEATURES], df.iloc[valid_idx][FEATURES]
        y_train, y_valid = df.iloc[train_idx][TARGET], df.iloc[valid_idx][TARGET]
        model = LGBMRegressor(**params, n_estimators=2000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        # Feature selection by importance
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': FEATURES,
            'importance': importances
        })
        selected = feature_importance_df[feature_importance_df['importance'] > threshold]['feature'].tolist()
        if not selected:
            # If no features left, penalize this trial
            return 1e6
        # Retrain with selected features
        model_sel = LGBMRegressor(**params, n_estimators=2000)
        model_sel.fit(
            X_train[selected], y_train,
            eval_set=[(X_valid[selected], y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        preds = model_sel.predict(X_valid[selected])
        rmses.append(np.sqrt(np.mean((preds - y_valid) ** 2)))
    return np.mean(rmses)

# Resume or create Optuna study
try:
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE)
    print("Loaded existing Optuna study.")
except Exception:
    study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE)
    print("Created new Optuna study.")

study.optimize(objective, n_trials=1)
print("Best params:", study.best_params)

# Use best params for final model
best_params = study.best_params
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "random_state": SEED,
    "bagging_freq": 1,
    "verbose": -1,
    "device": LGBM_DEVICE,
})
# Use last 10 weeks for validation as before
max_week = df["week"].max()
train_df = df[df["week"] <= max_week - FORECAST_HORIZON].copy()
valid_df = df[df["week"] > max_week - FORECAST_HORIZON].copy()
train_df, _ = train_df.align(df, join="right", axis=1, fill_value=0)
valid_df, _ = valid_df.align(df, join="right", axis=1, fill_value=0)

model = LGBMRegressor(**best_params, n_estimators=2000)
model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
    callbacks=[lgb.early_stopping(100, verbose=True)],
)

# --- Feature importance plot and quick report (as before) ---
lgb.plot_importance(model, max_num_features=30)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False)

# Print all feature importances
print("\nAll feature importances:")
print(feature_importance_df.to_string(index=False))

# --- Automated feature selection by importance threshold ---
thresholds = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
best_rmse = float('inf')
best_threshold = None
best_features = FEATURES
results = []

for thresh in thresholds:
    selected = feature_importance_df[feature_importance_df['importance'] > thresh]['feature'].tolist()
    if not selected:
        continue
    model_sel = LGBMRegressor(**best_params, n_estimators=2000)
    model_sel.fit(
        train_df[selected], train_df[TARGET],
        eval_set=[(valid_df[selected], valid_df[TARGET])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    preds = model_sel.predict(valid_df[selected])
    rmse = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))
    results.append((thresh, rmse, selected))
    print(f"Threshold {thresh}: {len(selected)} features, Validation RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_threshold = thresh
        best_features = selected

print(f"\nBest threshold: {best_threshold} (Validation RMSE: {best_rmse:.4f})")
print(f"Features kept ({len(best_features)}): {best_features}")

# Use best_features for final prediction
FEATURES = best_features

# --- Retrain final model on selected features ---
final_model = LGBMRegressor(**best_params, n_estimators=2000)
final_model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
    callbacks=[lgb.early_stopping(100, verbose=True)],
)

# --- Feature importance plot ---
fig, ax = plt.subplots(figsize=(max(10, 0.25*len(FEATURES)), min(20, 0.4*len(FEATURES))))
lgb.plot_importance(final_model, max_num_features=len(FEATURES), ax=ax, importance_type='split', title=None)
plt.title("LightGBM Feature Importance (Final Model)")
plt.tight_layout()
plt.show()

# --- Log all feature importances ---
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False)

# Print all feature importances
print("\nAll feature importances:")
print(feature_importance_df.to_string(index=False))

# Save to CSV
feature_importance_df.to_csv("feature_importances.csv", index=False)
print("All feature importances saved to feature_importances.csv.")

def make_test_features(test, train_hist):
    test_feat = test.copy()
    test_feat["num_orders"] = np.nan
    feature_rows = []
    for _, row in tqdm(test_feat.iterrows(), total=len(test_feat), desc="Building test features"):
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        feature_row = row.copy()
        # Lags
        for lag in LAG_WEEKS:
            feature_row[f"num_orders_lag_{lag}"] = hist.iloc[-lag]["num_orders"] if len(hist) >= lag else (hist["num_orders"].mean() if len(hist) > 0 else 0)
        # Rolling stats
        for window in ROLLING_WINDOWS:
            vals = hist["num_orders"].iloc[-window:] if len(hist) >= window else hist["num_orders"]
            feature_row[f"rolling_mean_{window}"] = vals.mean() if len(vals) > 0 else 0
            feature_row[f"rolling_std_{window}"] = vals.std() if len(vals) > 1 else 0
            feature_row[f"rolling_min_{window}"] = vals.min() if len(vals) > 0 else 0
            feature_row[f"rolling_max_{window}"] = vals.max() if len(vals) > 0 else 0
            feature_row[f"rolling_median_{window}"] = vals.median() if len(vals) > 0 else 0
        # Differencing
        feature_row["num_orders_diff_1"] = hist.iloc[-1]["num_orders"] - hist.iloc[-2]["num_orders"] if len(hist) >= 2 else 0
        feature_row["num_orders_diff_2"] = hist.iloc[-1]["num_orders"] - hist.iloc[-3]["num_orders"] if len(hist) >= 3 else 0
        feature_row["checkout_price_diff_1"] = row["checkout_price"] - hist.iloc[-1]["checkout_price"] if len(hist) >= 1 else 0
        # Log transforms
        feature_row["log_num_orders"] = np.log1p(hist.iloc[-1]["num_orders"]) if len(hist) >= 1 else 0
        feature_row["log_checkout_price"] = np.log1p(row["checkout_price"])
        # Discount features
        feature_row["discount"] = row["base_price"] - row["checkout_price"]
        feature_row["discount_pct"] = feature_row["discount"] / row["base_price"] if row["base_price"] != 0 else 0
        feature_row["price_diff"] = row["checkout_price"] - hist.iloc[-1]["checkout_price"] if len(hist) >= 1 else 0
        # Promotion rolling sums
        for col in ["emailer_for_promotion", "homepage_featured"]:
            vals = hist[col].iloc[-3:] if len(hist) >= 3 else hist[col]
            feature_row[f"{col}_rolling_sum_3"] = vals.sum() if len(vals) > 0 else 0
        # Cumulative stats
        feature_row["cummean_num_orders"] = hist["num_orders"].expanding().mean().iloc[-1] if len(hist) > 0 else 0
        feature_row["cumsum_num_orders"] = hist["num_orders"].sum() if len(hist) > 0 else 0
        # Interaction features
        feature_row["discount_x_promo"] = feature_row["discount"] * row["emailer_for_promotion"]
        feature_row["discount_x_homepage"] = feature_row["discount"] * row["homepage_featured"]
        feature_row["price_ratio"] = row["checkout_price"] / (row["base_price"] + 1e-3)
        # Week features
        feature_row["weekofyear"] = row["week"] % 52
        feature_row["week_sin"] = np.sin(2 * np.pi * feature_row["weekofyear"] / 52)
        feature_row["week_cos"] = np.cos(2 * np.pi * feature_row["weekofyear"] / 52)
        # Target encoding for category/cuisine/center_type
        for col in ["category", "cuisine", "center_type"]:
            if col in row and col in train_hist.columns:
                means = train_hist.groupby(col)["num_orders"].mean()
                feature_row[f"{col}_target_enc"] = means.get(row[col], train_hist["num_orders"].mean())
        feature_rows.append(feature_row)
    return pd.DataFrame(feature_rows)

test_feat = make_test_features(test, df)
existing_cat_cols = [col for col in cat_cols if col in test_feat.columns]
if existing_cat_cols:
    test_feat = pd.get_dummies(test_feat, columns=existing_cat_cols)
test_feat, _ = test_feat.align(df, join="right", axis=1, fill_value=0)

test_feat["num_orders"] = np.clip(final_model.predict(test_feat[FEATURES]), 0, None).round().astype(int)

submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
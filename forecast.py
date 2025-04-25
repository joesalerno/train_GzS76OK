import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold
import os

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
        preds = model.predict(X_valid)
        rmses.append(np.sqrt(np.mean((preds - y_valid) ** 2)))
    return np.mean(rmses)

# Resume or create Optuna study
try:
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE)
    print("Loaded existing Optuna study.")
except Exception:
    study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE)
    print("Created new Optuna study.")

study.optimize(objective, n_trials=100)
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

# --- Feature importance plot ---
plt.figure(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=30)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()

def make_test_features(test, train_hist):
    test_feat = test.copy()
    test_feat["num_orders"] = np.nan
    feature_rows = []
    for _, row in tqdm(test_feat.iterrows(), total=len(test_feat), desc="Building test features"):
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        feature_row = row.copy()
        for lag in LAG_WEEKS:
            feature_row[f"num_orders_lag_{lag}"] = hist.iloc[-lag]["num_orders"] if len(hist) >= lag else (hist["num_orders"].mean() if len(hist) > 0 else 0)
        for window in ROLLING_WINDOWS:
            vals = hist["num_orders"].iloc[-window:] if len(hist) >= window else hist["num_orders"]
            feature_row[f"rolling_mean_{window}"] = vals.mean() if len(vals) > 0 else 0
            feature_row[f"rolling_std_{window}"] = vals.std() if len(vals) > 1 else 0
        feature_row["discount"] = feature_row["base_price"] - feature_row["checkout_price"]
        feature_row["discount_pct"] = feature_row["discount"] / feature_row["base_price"] if feature_row["base_price"] != 0 else 0
        feature_row["price_diff"] = feature_row["checkout_price"] - hist.iloc[-1]["checkout_price"] if len(hist) >= 1 else 0
        for col in ["emailer_for_promotion", "homepage_featured"]:
            vals = hist[col].iloc[-3:] if len(hist) >= 3 else hist[col]
            feature_row[f"{col}_rolling_sum_3"] = vals.sum() if len(vals) > 0 else 0
        feature_row["weekofyear"] = feature_row["week"] % 52
        feature_rows.append(feature_row)
    return pd.DataFrame(feature_rows)

test_feat = make_test_features(test, df)
existing_cat_cols = [col for col in cat_cols if col in test_feat.columns]
if existing_cat_cols:
    test_feat = pd.get_dummies(test_feat, columns=existing_cat_cols)
test_feat, _ = test_feat.align(df, join="right", axis=1, fill_value=0)

test_feat["num_orders"] = np.clip(model.predict(test_feat[FEATURES]), 0, None).round().astype(int)

submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
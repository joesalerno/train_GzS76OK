import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna

# --- Configuration ---
USE_TUNER = True  # Set to False to skip tuning
N_TRIALS = 40     # Number of Optuna trials
VALIDATION_WEEKS = 8 # How many recent weeks for validation
SEED = 42

# File Paths
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"

# Feature Engineering Parameters
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]

# --- Load Data ---
df = pd.read_csv(DATA_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test = pd.read_csv(TEST_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
meal_info = pd.read_csv(MEAL_INFO_PATH)
df = df.merge(meal_info, on="meal_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")
center_info = pd.read_csv(CENTER_INFO_PATH)
df = df.merge(center_info, on="center_id", how="left")
test = test.merge(center_info, on="center_id", how="left")

# --- Feature engineering ---
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
    # --- New advanced features ---
    df["month"] = ((df["week"] - 1) // 4 + 1).astype(int)
    df["quarter"] = ((df["week"] - 1) // 13 + 1).astype(int)
    df["is_month_start"] = (df["week"] % 4 == 1).astype(int)
    df["is_month_end"] = (df["week"] % 4 == 0).astype(int)
    # Interaction features
    df["price_x_email"] = df["checkout_price"] * df["emailer_for_promotion"]
    df["price_x_home"] = df["checkout_price"] * df["homepage_featured"]
    # Lagged price and promo
    df["checkout_price_lag1"] = df.groupby(["center_id", "meal_id"])["checkout_price"].shift(1)
    df["emailer_lag1"] = df.groupby(["center_id", "meal_id"])["emailer_for_promotion"].shift(1)
    df["home_lag1"] = df.groupby(["center_id", "meal_id"])["homepage_featured"].shift(1)
    # --- Smarter NaN handling ---
    lag_roll_cols = [col for col in df.columns if any(x in col for x in ["lag", "rolling", "diff"])]
    df[lag_roll_cols] = df.groupby(["center_id", "meal_id"])[lag_roll_cols].ffill()
    for col in lag_roll_cols:
        df[col] = df[col].fillna(df[col].mean())
    # --- Aggregate features ---
    for stat in ["mean", "median", "std"]:
        df[f"center_orders_{stat}"] = df.groupby("center_id")["num_orders"].transform(stat)
        df[f"meal_orders_{stat}"] = df.groupby("meal_id")["num_orders"].transform(stat)
    return df

df = create_features(df)
df = df[df["num_orders"].notna()].reset_index(drop=True)

# --- One-hot encode categorical columns if present ---
cat_cols = []
for col in ["category", "cuisine", "center_type"]:
    if col in df.columns:
        cat_cols.append(col)
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols)
    test = pd.get_dummies(test, columns=cat_cols)
df, test = df.align(test, join="left", axis=1, fill_value=0)

# --- Feature list ---
FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", "weekofyear",
    "month", "quarter", "is_month_start", "is_month_end",
    "price_x_email", "price_x_home", "checkout_price_lag1", "emailer_lag1", "home_lag1",
    "center_orders_mean", "center_orders_median", "center_orders_std",
    "meal_orders_mean", "meal_orders_median", "meal_orders_std"
]
FEATURES += [f"num_orders_lag_{lag}" for lag in LAG_WEEKS]
FEATURES += [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
FEATURES += [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
FEATURES += [f"{col}_rolling_sum_3" for col in ["emailer_for_promotion", "homepage_featured"]]
FEATURES += [col for col in df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]
FEATURES = [f for f in FEATURES if f in df.columns]
# Remove zero-variance features
for f in FEATURES.copy():
    if df[f].nunique() <= 1:
        FEATURES.remove(f)
TARGET = "num_orders"

# --- Train/validation split (last 10 weeks for validation) ---
max_week = df["week"].max()
VALIDATION_WEEKS = 10
valid_df = df[df["week"] > max_week - VALIDATION_WEEKS].copy()
train_df = df[df["week"] <= max_week - VALIDATION_WEEKS].copy()

# --- Model training ---
def get_lgbm(params=None):
    default_params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_child_samples=20,
        lambda_l1=0.1,
        lambda_l2=0.1,
        max_depth=5,
        random_state=SEED,
        n_estimators=3000,
        verbose=-1
    )
    if params:
        default_params.update(params)
    return LGBMRegressor(**default_params)

if USE_TUNER:
    import lightgbm as lgb
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
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = model.predict(valid_df[FEATURES])
        rmse = np.sqrt(np.mean((preds - valid_df[TARGET]) ** 2))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    print("Best params:", best_params)
    model = get_lgbm(best_params)
    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
else:
    model = get_lgbm()
    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric="rmse"
    )

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
cat_cols_in_test = [col for col in cat_cols if col in test_feat.columns]
if cat_cols_in_test:
    test_feat = pd.get_dummies(test_feat, columns=cat_cols_in_test)
test_feat, _ = test_feat.align(df, join="right", axis=1, fill_value=0)

test_feat["num_orders"] = np.clip(model.predict(test_feat[FEATURES]), 0, None).round().astype(int)

submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")

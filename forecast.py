import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
FORECAST_HORIZON = 10
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
SEED = 42

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

max_week = df["week"].max()
train_df = df[df["week"] <= max_week - FORECAST_HORIZON].copy()
valid_df = df[df["week"] > max_week - FORECAST_HORIZON].copy()

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

# --- FIX: Align train/valid after split to ensure all columns exist ---
train_df, _ = train_df.align(df, join="right", axis=1, fill_value=0)
valid_df, _ = valid_df.align(df, join="right", axis=1, fill_value=0)

FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured",
    "discount", "discount_pct", "price_diff", "weekofyear"
] + [col for col in df.columns if "lag_" in col or "rolling_" in col] \
  + [col for col in df.columns if col.startswith("category_") or col.startswith("cuisine_") or col.startswith("center_type_")] \
  + [col for col in df.columns if col in center_info.columns and col != "center_id"]

# Remove duplicates and ensure all features exist in the DataFrame
FEATURES = list(dict.fromkeys(FEATURES))
FEATURES = [f for f in FEATURES if f in df.columns and f != "num_orders"]

TARGET = "num_orders"

lgb_params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=64,
    random_state=SEED,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
)

model = LGBMRegressor(**lgb_params, n_estimators=2000)
model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
    callbacks=[lgb.early_stopping(100, verbose=True)],
)

# --- Add this block to show feature importance ---
plt.figure(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=30)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()
# --- End block ---

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
# One-hot encode and align test features only if needed
existing_cat_cols = [col for col in cat_cols if col in test_feat.columns]
if existing_cat_cols:
    test_feat = pd.get_dummies(test_feat, columns=existing_cat_cols)
test_feat, _ = test_feat.align(df, join="right", axis=1, fill_value=0)

test_feat["num_orders"] = np.clip(model.predict(test_feat[FEATURES]), 0, None).round().astype(int)

submission = test_feat[["id", "num_orders"]].copy()
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
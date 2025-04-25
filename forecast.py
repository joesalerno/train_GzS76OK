import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold
import shap

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

def get_holiday_weeks():
    # Example: Add/adjust based on your country/region
    # Weeks for New Year, Christmas, Diwali, Eid, etc.
    return set([1, 52, 45, 10, 25])

def get_rare_categories(df, col, min_count=100):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index.tolist()
    return rare

def create_features(df):
    df = df.copy()
    # Ensure weekofyear is created first
    df["weekofyear"] = df["week"] % 52
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
    # --- Enhanced seasonality ---
    df["month"] = ((df["week"] - 1) // 4 + 1).clip(1, 12)
    df["quarter"] = ((df["week"] - 1) // 13 + 1).clip(1, 4)
    # Cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    # Fourier series for weekofyear
    for k in [1, 2, 3]:
        df[f"weekofyear_sin_{k}"] = np.sin(2 * np.pi * k * df["weekofyear"] / 52)
        df[f"weekofyear_cos_{k}"] = np.cos(2 * np.pi * k * df["weekofyear"] / 52)
    # Start/end of month/quarter
    df["is_month_start"] = (df["week"] % 4 == 1).astype(int)
    df["is_month_end"] = (df["week"] % 4 == 0).astype(int)
    df["is_quarter_start"] = (df["week"] % 13 == 1).astype(int)
    df["is_quarter_end"] = (df["week"] % 13 == 0).astype(int)
    # --- Holiday/event features ---
    holiday_weeks = get_holiday_weeks()
    df["is_holiday_week"] = df["weekofyear"].isin(holiday_weeks).astype(int)
    df["weeks_to_next_holiday"] = df["weekofyear"].apply(lambda w: min([(h-w)%52 for h in holiday_weeks]))
    # --- Demand volatility/change ---
    df["rolling_volatility_5"] = df.groupby(["center_id", "meal_id"])["num_orders"].shift(1).rolling(5).std().reset_index(0, drop=True)
    df["rolling_volatility_10"] = df.groupby(["center_id", "meal_id"])["num_orders"].shift(1).rolling(10).std().reset_index(0, drop=True)
    df["demand_pct_change_1"] = df["num_orders_diff_1"] / (df.groupby(["center_id", "meal_id"])["num_orders"].shift(1) + 1e-3)
    # --- Interactions ---
    df["price_diff_x_weekofyear"] = df["price_diff"] * df["weekofyear"]
    df["discount_x_weekofyear"] = df["discount"] * df["weekofyear"]
    df["promo_x_weekofyear"] = df["emailer_for_promotion"] * df["weekofyear"]
    # --- Rare category grouping ---
    for col in ["category", "cuisine", "center_type"]:
        if col in df.columns:
            rare = get_rare_categories(df, col, min_count=100)
            df[f"{col}_is_rare"] = df[col].isin(rare).astype(int)
    # --- Regime clustering (simple: high/med/low demand by weekofyear) ---
    week_means = df.groupby("weekofyear")["num_orders"].mean()
    bins = pd.qcut(week_means, 3, labels=["low", "med", "high"])
    week_regime = dict(zip(week_means.index, bins))
    df["seasonal_regime"] = df["weekofyear"].map(week_regime).astype("category").cat.codes
    return df

# --- Feature engineering ---
df = create_features(df)
df = df[df["num_orders"].notna()].reset_index(drop=True)

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

# Zero SHAP features to exclude
ZERO_SHAP_FEATURES = [
    'cuisine_Italian', 'category_Biryani', 'category_Desert', 'category_Extras', 'category_is_rare',
    'cuisine_Thai', 'category_Starters', 'cuisine_Continental', 'category_Soup', 'category_Pasta',
    'category_Other Snacks', 'category_Seafood', 'category_Fish', 'category_Pizza', 'cuisine_is_rare',
    'center_type_is_rare', 'category_Beverages', 'is_month_start', 'is_quarter_start', 'is_quarter_end'
]

# Build FEATURES list to include all advanced engineered features, but exclude zero-SHAP features
FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "homepage_featured",
    "discount", "discount_pct", "price_diff", "weekofyear"
]
# Add lags and rolling
FEATURES += [col for col in df.columns if ("lag_" in col or "rolling_" in col) and col not in ZERO_SHAP_FEATURES]
# Add one-hot and target encodings
FEATURES += [col for col in df.columns if (col.startswith("category_") or col.startswith("cuisine_") or col.startswith("center_type_")) and col not in ZERO_SHAP_FEATURES]
# Add center info columns
FEATURES += [col for col in df.columns if col in center_info.columns and col != "center_id" and col not in ZERO_SHAP_FEATURES]
# Add advanced engineered features (expanded to include all engineered features)
FEATURES += [col for col in df.columns if (
    col.startswith("month") or col.startswith("weekofyear_sin") or col.startswith("weekofyear_cos") or
    col.startswith("is_month_") or col.startswith("is_quarter_") or
    col.startswith("weeks_to_next_holiday") or
    col.startswith("demand_pct_change") or col.startswith("volatility") or
    col.startswith("regime") or col.endswith("_x_weekofyear") or
    col.endswith("_is_rare") or col == "seasonal_regime" or
    col.startswith("log_") or col.endswith("_diff_1") or col.endswith("_diff_2") or
    col.startswith("price_ratio") or col.startswith("discount_x_") or col.startswith("promo_x_") or
    col.startswith("cummean_") or col.startswith("cumsum_")
) and col not in ZERO_SHAP_FEATURES]
# Remove duplicates and target
FEATURES = list(dict.fromkeys(FEATURES))
FEATURES = [f for f in FEATURES if f in df.columns and f != "num_orders" and f not in ZERO_SHAP_FEATURES]

TARGET = "num_orders"

# --- RMSLE metric ---
def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def lgb_rmsle_eval(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    rmsle_val = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    return 'rmsle', rmsle_val, False

# --- Optuna hyperparameter tuning with KFold and more params ---
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 64),  # reduced upper bound
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": 1,
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1, 10),  # increased lower/upper bound
        "lambda_l2": trial.suggest_float("lambda_l2", 1, 10),  # increased lower/upper bound
        "max_depth": trial.suggest_int("max_depth", 3, 7),    # reduced upper bound
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 1),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 20000, 300000),
        "random_state": SEED,
        "verbose": -1,
        "device": LGBM_DEVICE,
    }
    threshold = trial.suggest_categorical("feature_importance_threshold", [0, 1, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1500, 2000])
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmsles = []
    for train_idx, valid_idx in kf.split(df):
        X_train, X_valid = df.iloc[train_idx][FEATURES], df.iloc[valid_idx][FEATURES]
        y_train, y_valid = df.iloc[train_idx][TARGET], df.iloc[valid_idx][TARGET]
        model = LGBMRegressor(**params, n_estimators=2000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=lgb_rmsle_eval,
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
            return 1e6
        # Retrain with selected features
        model_sel = LGBMRegressor(**params, n_estimators=2000)
        model_sel.fit(
            X_train[selected], y_train,
            eval_set=[(X_valid[selected], y_valid)],
            eval_metric=lgb_rmsle_eval,
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        preds = model_sel.predict(X_valid[selected])
        rmsles.append(rmsle(y_valid, preds))
    return np.mean(rmsles)

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
    "metric": "None",
    "random_state": SEED,
    "bagging_freq": 1,
    "verbose": -1,
    "device": LGBM_DEVICE,
})
# Use time-based split for generalization
HOLDOUT_WEEKS = 8
def get_time_based_train_valid_split(df, holdout_weeks=8):
    # Find the max week in the data
    max_week = df['week'].max()
    # Validation set: all rows in the last `holdout_weeks` weeks
    valid_df = df[df['week'] > max_week - holdout_weeks].copy()
    # Training set: all rows before the holdout period
    train_df = df[df['week'] <= max_week - holdout_weeks].copy()
    return train_df, valid_df

train_df, valid_df = get_time_based_train_valid_split(df, holdout_weeks=HOLDOUT_WEEKS)
train_df, _ = train_df.align(df, join="right", axis=1, fill_value=0)
valid_df, _ = valid_df.align(df, join="right", axis=1, fill_value=0)

# --- Debug logs after train/valid split ---
print(f"\n[DEBUG] train_df shape: {train_df.shape}, valid_df shape: {valid_df.shape}")
print(f"[DEBUG] Number of unique (center_id, meal_id) in train_df: {train_df.groupby(['center_id', 'meal_id']).ngroups}")
print(f"[DEBUG] Number of unique (center_id, meal_id) in valid_df: {valid_df.groupby(['center_id', 'meal_id']).ngroups}")
print(f"[DEBUG] Weeks in train_df: min={train_df['week'].min()}, max={train_df['week'].max()}, unique={train_df['week'].nunique()}")
print(f"[DEBUG] Weeks in valid_df: min={valid_df['week'].min()}, max={valid_df['week'].max()}, unique={valid_df['week'].nunique()}")
print(f"[DEBUG] Total training rows: {len(train_df)}, Total validation rows: {len(valid_df)}")
print("[DEBUG] train_df sample:\n", train_df.head())
print("[DEBUG] valid_df sample:\n", valid_df.head())

model = LGBMRegressor(**best_params, n_estimators=2000)
model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
    eval_metric=lgb_rmsle_eval,
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

print("\nAll feature importances:")
print(feature_importance_df.to_string(index=False))

# --- SHAP value analysis BEFORE feature elimination ---
print("\nComputing SHAP values for initial model (before feature elimination)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(valid_df[FEATURES])
shap.summary_plot(shap_values, valid_df[FEATURES], plot_type="bar", show=True)
shap.summary_plot(shap_values, valid_df[FEATURES], show=True)
print("SHAP summary plots displayed. Use these to guide further feature selection.")

# --- Save SHAP values to CSV ---
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv("shap_feature_importances.csv", index=False)
print("SHAP feature importances saved to shap_feature_importances.csv.")

# --- Automated feature selection by importance threshold ---
thresholds = [0, 1, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1500, 2000]
best_rmsle = float('inf')
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
        eval_metric=lgb_rmsle_eval,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    preds = model_sel.predict(valid_df[selected])
    rmsle_val = rmsle(valid_df[TARGET], preds)
    results.append((thresh, rmsle_val, selected))
    print(f"Threshold {thresh}: {len(selected)} features, Validation RMSLE: {rmsle_val:.4f}")
    if rmsle_val < best_rmsle:
        best_rmsle = rmsle_val
        best_threshold = thresh
        best_features = selected

print(f"\nBest threshold: {best_threshold} (Validation RMSLE: {best_rmsle:.4f})")
print(f"Features kept ({len(best_features)}): {best_features}")

# Use best_features for final prediction
FEATURES = best_features

# --- Retrain final model on selected features ---
final_model = LGBMRegressor(**best_params, n_estimators=2000)
final_model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
    eval_metric=lgb_rmsle_eval,
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

print("\nAll feature importances:")
print(feature_importance_df.to_string(index=False))

# Save to CSV
feature_importance_df.to_csv("feature_importances.csv", index=False)
print("All feature importances saved to feature_importances.csv.")

def make_test_features(test, train_hist):
    test_feat = test.copy()
    test_feat["num_orders"] = np.nan
    feature_rows = []
    holiday_weeks = get_holiday_weeks()
    week_means = train_hist.groupby("weekofyear")["num_orders"].mean()
    bins = pd.qcut(week_means, 3, labels=["low", "med", "high"])
    week_regime = dict(zip(week_means.index, bins))
    for _, row in tqdm(test_feat.iterrows(), total=len(test_feat), desc="Building test features"):
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        feature_row = row.copy()
        # Restore weekofyear feature
        feature_row["weekofyear"] = row["week"] % 52
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
        # --- Enhanced seasonality ---
        feature_row["month"] = ((row["week"] - 1) // 4 + 1)
        feature_row["quarter"] = ((row["week"] - 1) // 13 + 1)
        feature_row["month_sin"] = np.sin(2 * np.pi * feature_row["month"] / 12)
        feature_row["month_cos"] = np.cos(2 * np.pi * feature_row["month"] / 12)
        feature_row["quarter_sin"] = np.sin(2 * np.pi * feature_row["quarter"] / 4)
        feature_row["quarter_cos"] = np.cos(2 * np.pi * feature_row["quarter"] / 4)
        for k in [1, 2, 3]:
            feature_row[f"weekofyear_sin_{k}"] = np.sin(2 * np.pi * k * feature_row["weekofyear"] / 52)
            feature_row[f"weekofyear_cos_{k}"] = np.cos(2 * np.pi * k * feature_row["weekofyear"] / 52)
        feature_row["is_month_start"] = int(row["week"] % 4 == 1)
        feature_row["is_month_end"] = int(row["week"] % 4 == 0)
        feature_row["is_quarter_start"] = int(row["week"] % 13 == 1)
        feature_row["is_quarter_end"] = int(row["week"] % 13 == 0)
        # --- Holiday/event features ---
        feature_row["is_holiday_week"] = int(feature_row["weekofyear"] in holiday_weeks)
        feature_row["weeks_to_next_holiday"] = min([(h-feature_row["weekofyear"])%52 for h in holiday_weeks])
        # --- Demand volatility/change ---
        vals5 = hist["num_orders"].iloc[-5:] if len(hist) >= 5 else hist["num_orders"]
        vals10 = hist["num_orders"].iloc[-10:] if len(hist) >= 10 else hist["num_orders"]
        feature_row["rolling_volatility_5"] = vals5.std() if len(vals5) > 1 else 0
        feature_row["rolling_volatility_10"] = vals10.std() if len(vals10) > 1 else 0
        feature_row["demand_pct_change_1"] = (hist.iloc[-1]["num_orders"] - hist.iloc[-2]["num_orders"]) / (hist.iloc[-2]["num_orders"] + 1e-3) if len(hist) >= 2 else 0
        # --- Interactions ---
        feature_row["price_diff_x_weekofyear"] = feature_row["price_diff"] * feature_row["weekofyear"]
        feature_row["discount_x_weekofyear"] = feature_row["discount"] * feature_row["weekofyear"]
        feature_row["promo_x_weekofyear"] = row["emailer_for_promotion"] * feature_row["weekofyear"]
        # --- Rare category grouping ---
        for col in ["category", "cuisine", "center_type"]:
            if col in row and col in train_hist.columns:
                rare = get_rare_categories(train_hist, col, min_count=100)
                feature_row[f"{col}_is_rare"] = int(row[col] in rare)
        # --- Regime clustering ---
        regime_label = week_regime.get(feature_row["weekofyear"], "med")
        regime_map = {"low": 0, "med": 1, "high": 2}
        feature_row["seasonal_regime"] = regime_map.get(regime_label, 1)
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

# After final_model is trained and feature importances are logged
# --- SHAP value analysis ---
print("\nComputing SHAP values for final model...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(valid_df[FEATURES])
shap.summary_plot(shap_values, valid_df[FEATURES], plot_type="bar", show=True)
shap.summary_plot(shap_values, valid_df[FEATURES], show=True)
print("SHAP summary plots displayed. Use these to guide further feature selection.")

# --- Remove features with zero mean absolute SHAP value ---
shap_importance_df = pd.read_csv("shap_feature_importances.csv")
nonzero_shap_features = shap_importance_df[shap_importance_df['mean_abs_shap'] > 0]['feature'].tolist()
FEATURES = [f for f in FEATURES if f in nonzero_shap_features]
print(f"Removed features with zero SHAP value. {len(FEATURES)} features remain for selection and training.")
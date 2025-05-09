import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import random

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
N_TRIALS = 1

# Set random seeds for reproducibility (future-proof for numpy)
import numpy.random as npr
rng = npr.default_rng(SEED)
random.seed(SEED)

LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]
TARGET = "num_orders"

# --- Load data ---
df = pd.read_csv(DATA_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test = pd.read_csv(TEST_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test_ids = test["id"].copy()  # Save test IDs before feature engineering
meal_info = pd.read_csv(MEAL_INFO_PATH)
df = df.merge(meal_info, on="meal_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")
center_info = pd.read_csv(CENTER_INFO_PATH)
df = df.merge(center_info, on="center_id", how="left")
test = test.merge(center_info, on="center_id", how="left")

def get_rare_categories(df, col, min_count=100):
    if col not in df.columns:
        return []
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index.tolist()
    return rare

def cyclical_encode(df, col, max_val):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

# --- Overhauled feature engineering ---
def create_features_all(df, is_test=False, train_hist=None):
    df = df.copy()
    group = ["center_id", "meal_id"]
    # --- Fail-fast: Ensure all required columns are present before feature engineering ---
    required_cols = ["category", "cuisine", "center_type", "meal_id", "center_id", "week", "checkout_price", "base_price", "emailer_for_promotion", "homepage_featured"]
    if not is_test:
        required_cols.append("num_orders")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")
    # --- Rare category grouping ---
    for col in ["category", "cuisine", "center_type"]:
        rare = get_rare_categories(train_hist if is_test and train_hist is not None else df, col, min_count=100)
        df[col] = df[col].apply(lambda x: "Rare" if x in rare else x)
    # --- Lags, rollings, price, promo, interactions (as before) ---
    if is_test and train_hist is not None:
        # Row-wise for test, using train_hist for lags/rollings and all historical features
        for lag in LAG_WEEKS:
            df[f"num_orders_lag_{lag}"] = np.nan
        for window in ROLLING_WINDOWS:
            df[f"rolling_mean_{window}"] = np.nan
            df[f"rolling_std_{window}"] = np.nan
        df["price_diff"] = np.nan
        for col in ["emailer_for_promotion", "homepage_featured"]:
            df[f"{col}_rolling_sum_3"] = np.nan
        df["weekofyear"] = df["week"] % 52
        df["price_diff_x_emailer"] = np.nan
        df["lag1_x_emailer"] = np.nan
        df["price_diff_x_home"] = np.nan
        df["lag1_x_home"] = np.nan
        for idx, row in df.iterrows():
            cid, mid, week = row["center_id"], row["meal_id"], row["week"]
            hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
            for lag in LAG_WEEKS:
                df.at[idx, f"num_orders_lag_{lag}"] = hist["num_orders"].iloc[-lag] if len(hist) >= lag else 0
            for window in ROLLING_WINDOWS:
                vals = hist["num_orders"].iloc[-window:] if len(hist) >= window else hist["num_orders"]
                df.at[idx, f"rolling_mean_{window}"] = vals.mean() if len(vals) > 0 else 0
                df.at[idx, f"rolling_std_{window}"] = vals.std() if len(vals) > 1 else 0
            df.at[idx, "price_diff"] = row["checkout_price"] - hist["checkout_price"].iloc[-1] if len(hist) >= 1 else 0
            for col in ["emailer_for_promotion", "homepage_featured"]:
                vals = hist[col].iloc[-3:] if len(hist) >= 3 else hist[col]
                df.at[idx, f"{col}_rolling_sum_3"] = vals.sum() if len(vals) > 0 else 0
            df.at[idx, "price_diff_x_emailer"] = df.at[idx, "price_diff"] * row["emailer_for_promotion"]
            df.at[idx, "lag1_x_emailer"] = df.at[idx, "num_orders_lag_1"] * row["emailer_for_promotion"]
            df.at[idx, "price_diff_x_home"] = df.at[idx, "price_diff"] * row["homepage_featured"]
            df.at[idx, "lag1_x_home"] = df.at[idx, "num_orders_lag_1"] * row["homepage_featured"]
        # Impute rolling_volatility_10 and demand_pct_change_1 with train means if not enough history
        for col in ["rolling_volatility_10", "demand_pct_change_1"]:
            train_mean = train_hist[col].mean() if col in train_hist.columns else 0
            df[col] = df[col].fillna(train_mean)
    else:
        # Standard vectorized feature engineering for train/valid
        for lag in LAG_WEEKS:
            df[f"num_orders_lag_{lag}"] = df.groupby(group)["num_orders"].shift(lag)
        for window in ROLLING_WINDOWS:
            shifted = df.groupby(group)["num_orders"].shift(1)
            df[f"rolling_mean_{window}"] = shifted.rolling(window).mean().reset_index(0, drop=True)
            df[f"rolling_std_{window}"] = shifted.rolling(window).std().reset_index(0, drop=True)
        df["price_diff"] = df.groupby(group)["checkout_price"].diff()
        for col in ["emailer_for_promotion", "homepage_featured"]:
            shifted = df.groupby(group)[col].shift(1)
            df[f"{col}_rolling_sum_3"] = shifted.rolling(3).sum().reset_index(0, drop=True)
        df["weekofyear"] = df["week"] % 52
        df["price_diff_x_emailer"] = df["price_diff"] * df["emailer_for_promotion"]
        df["lag1_x_emailer"] = df["num_orders_lag_1"] * df["emailer_for_promotion"]
        df["price_diff_x_home"] = df["price_diff"] * df["homepage_featured"]
        df["lag1_x_home"] = df["num_orders_lag_1"] * df["homepage_featured"]
    # --- Target encoding for categorical features ---
    for col in ["category", "cuisine", "center_type"]:
        ref_df = train_hist if is_test and train_hist is not None else df
        means = ref_df.groupby(col)["num_orders"].mean()
        global_mean = ref_df["num_orders"].mean()
        df[f"{col}_target_enc"] = df[col].map(means).fillna(global_mean)
    # --- Aggregate features ---
    for col in ["meal_id", "center_id", "category", "cuisine", "center_type"]:
        for stat in ["mean", "median", "std"]:
            if not is_test:
                df[f"{col}_orders_{stat}"] = df.groupby(col)["num_orders"].transform(stat)
            else:
                agg_map = (train_hist if train_hist is not None else df).groupby(col)["num_orders"].agg(stat)
                df[f"{col}_orders_{stat}"] = df[col].map(agg_map).fillna(0)
    # --- Cyclical time features ---
    df["weekofyear"] = df["week"] % 52
    df = cyclical_encode(df, "weekofyear", 52)
    # --- Promo recency ---
    for col in ["emailer_for_promotion", "homepage_featured"]:
        df[f"weeks_since_{col}"] = (
            df.groupby(group)[col]
            .transform(lambda x: x[::-1].cumsum()[::-1].where(x == 1, 0))
        )
    # --- Holiday/event features ---
    holiday_weeks = set([1, 52, 45, 10, 25])
    df["is_holiday_week"] = df["weekofyear"].isin(holiday_weeks).astype(int)
    df["weeks_to_next_holiday"] = df["weekofyear"].apply(lambda w: min([(h-w)%52 for h in holiday_weeks]))
    # --- Demand volatility/change ---
    if not is_test:
        for window in [5, 10]:
            df[f"rolling_volatility_{window}"] = (
                df.groupby(group)["num_orders"]
                .transform(lambda x: x.shift(1).rolling(window).std())
            )
    else:
        for window in [5, 10]:
            df[f"rolling_volatility_{window}"] = 0
    # --- Regime clustering ---
    ref_df = train_hist if is_test and train_hist is not None else df
    if "weekofyear" not in ref_df.columns:
        ref_df = ref_df.copy()
        ref_df["weekofyear"] = ref_df["week"] % 52
    week_means = ref_df.groupby("weekofyear")["num_orders"].mean()
    if len(week_means.unique()) >= 3:
        bins = pd.qcut(week_means, 3, labels=["low", "med", "high"])
    else:
        bins = pd.Series(["med"] * len(week_means), index=week_means.index)
    week_regime = dict(zip(week_means.index, bins))
    df["seasonal_regime"] = df["weekofyear"].map(week_regime).astype("category").cat.codes
    # --- One-hot encoding (after rare grouping) ---
    cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
        # Ensure all dummies present in test match train
        if is_test and train_hist is not None:
            train_dummies = [col for col in train_hist.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]
            for col in train_dummies:
                if col not in df.columns:
                    df[col] = 0
            # Reorder columns to match train
            df = df.reindex(columns=list(df.columns) + [c for c in train_dummies if c not in df.columns], fill_value=0)
    # --- Fill NaNs for all lag/rolling/interaction features ---
    lag_roll_cols = [col for col in df.columns if any(x in col for x in ["lag", "rolling", "diff", "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home", "volatility", "demand_pct_change"])]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    # --- Final NaN fill for all features ---
    df.fillna(0, inplace=True)
    return df

# --- Robust feature list builder (prune zero-SHAP features if available) ---
def get_feature_list(df):
    base = [
        "center_id", "meal_id", "checkout_price", "base_price",
        "homepage_featured", "emailer_for_promotion",
        "discount", "discount_pct", "price_diff", "weekofyear",
        "weekofyear_sin", "weekofyear_cos", "is_holiday_week", "weeks_to_next_holiday", "seasonal_regime"
    ]
    base += [f"num_orders_lag_{lag}" for lag in LAG_WEEKS]
    base += [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    base += [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    base += [f"{col}_rolling_sum_3" for col in ["emailer_for_promotion", "homepage_featured"]]
    base += ["price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"]
    # REMOVE rolling_volatility_10 and demand_pct_change_1 from feature list for robustness
    # base += [f"rolling_volatility_{w}" for w in [5, 10]]
    # base += ["demand_pct_change_1"]
    base += [f"weeks_since_{col}" for col in ["emailer_for_promotion", "homepage_featured"]]
    base += [f"{col}_target_enc" for col in ["category", "cuisine", "center_type"] if f"{col}_target_enc" in df.columns]
    base += [f"{col}_orders_{stat}" for col in ["meal_id", "center_id", "category", "cuisine", "center_type"] for stat in ["mean", "median", "std"] if f"{col}_orders_{stat}" in df.columns]
    base += [col for col in df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]
    # Prune zero-SHAP features if available
    try:
        shap_df = pd.read_csv("shap_hybrid_feature_importances.csv")
        nonzero_shap = set(shap_df[shap_df['mean_abs_shap'] > 0]['feature'])
        base = [f for f in base if f in nonzero_shap or not f.startswith(('category_', 'cuisine_', 'center_type_'))]
    except Exception:
        pass
    return [f for f in base if f in df.columns]

# --- RMSLE metric ---
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred.clip(0)) - np.log1p(y_true.clip(0)))))


def lgb_rmsle(y_true, y_pred):
    rmsle_score = rmsle(y_true, y_pred)
    return 'rmsle', rmsle_score, False

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

# --- Unified Optuna feature+hyperparameter selection with CV ---
def optuna_feature_selection_cv(train_df, features, target, n_trials=N_TRIALS):
    from optuna.pruners import SuccessiveHalvingPruner
    OPTUNA_DB = "sqlite:///optuna_hybrid_optuna.db"
    OPTUNA_STUDY_NAME = "hybrid_optuna_feature_selection"
    def objective(trial):
        selected = [f for f in features if trial.suggest_categorical(f'use_{f}', [True, False])]
        if not selected:
            return 1e6
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
            'random_state': SEED,
        }
        gkf = GroupKFold(n_splits=3)
        rmsles = []
        for train_idx, valid_idx in gkf.split(train_df, groups=train_df["meal_id"]):
            X_tr, X_val = train_df.iloc[train_idx][selected], train_df.iloc[valid_idx][selected]
            y_tr, y_val = train_df.iloc[train_idx][target], train_df.iloc[valid_idx][target]
            model = get_lgbm(params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=lgb_rmsle)
            preds = model.predict(X_val)
            rmsles.append(rmsle(y_val, preds))
        trial.report(np.mean(rmsles), step=0)
        return np.mean(rmsles)

    # ...existing code...

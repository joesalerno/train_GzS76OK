import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
OUTPUT_DIRECTORY = "output"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

import re
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb  # Added for early stopping callback
from sklearn.model_selection import TimeSeriesSplit
import optuna
import shap
from tqdm import tqdm
import logging

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28]
# Other features (not directly dependent on recursive prediction)
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 1 # Number of Optuna trials (increased for better search)
OPTUNA_NJOBS = 1  # Use sequential Optuna trials for best resource usage with LGBM
OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "submission_recursive"
SHAP_FILE_PREFIX = "shap_recursive"
N_SHAP_SAMPLES = 2000

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    meal_info = pd.read_csv(MEAL_INFO_PATH)
    center_info = pd.read_csv(CENTER_INFO_PATH)
except FileNotFoundError as e:
    logging.error(f"Error loading data file: {e}. Ensure train.csv, test.csv, meal_info.csv, and fulfilment_center_info.csv are present.")
    raise

logging.info("Preprocessing data...")
def preprocess_data(df, meal_info, center_info):
    """Merges dataframes and sorts."""
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

df = preprocess_data(df, meal_info, center_info)
test = preprocess_data(test, meal_info, center_info)

# Add placeholder for num_orders in test for alignment
if 'num_orders' not in test.columns:
    test['num_orders'] = np.nan

logging.info("Creating features...")
GROUP_COLS = ["center_id", "meal_id"]

# Add cyclical encoding for weekofyear and month
from math import pi
def create_temporal_features(df):
    df_out = df.copy()
    df_out["weekofyear"] = df_out["week"] % 52
    df_out["weekofyear_sin"] = np.sin(2 * pi * df_out["weekofyear"] / 52)
    df_out["weekofyear_cos"] = np.cos(2 * pi * df_out["weekofyear"] / 52)
    if "month" not in df_out.columns:
        df_out["month"] = ((df_out["week"] - 1) // 4) % 12 + 1
    df_out["month_sin"] = np.sin(2 * pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * pi * df_out["month"] / 12)
    # Example holiday weeks (customize as needed)
    holiday_weeks = set([1, 10, 25, 45, 52])
    df_out["is_holiday_week"] = df_out["weekofyear"].isin(holiday_weeks).astype(int)
    return df_out

def create_lag_rolling_features(df, target_col='num_orders', lag_weeks=LAG_WEEKS, rolling_windows=ROLLING_WINDOWS):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS, observed=False)
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)
    return df_out

def create_other_features(df):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS, observed=False)
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = group["checkout_price"].diff()
    for col in OTHER_ROLLING_SUM_COLS:
        shifted = group[col].shift(1)
        df_out[f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}"] = shifted.rolling(OTHER_ROLLING_SUM_WINDOW, min_periods=1).sum().reset_index(drop=True)
    return df_out

def create_group_aggregates(df):
    df_out = df.copy()
    df_out['center_orders_mean'] = df_out.groupby('center_id', observed=False)['num_orders'].transform('mean')
    df_out['center_orders_std'] = df_out.groupby('center_id', observed=False)['num_orders'].transform('std')
    df_out['meal_orders_mean'] = df_out.groupby('meal_id', observed=False)['num_orders'].transform('mean')
    df_out['meal_orders_std'] = df_out.groupby('meal_id', observed=False)['num_orders'].transform('std')
    if 'category' in df_out.columns:
        df_out['category_orders_mean'] = df_out.groupby('category', observed=False)['num_orders'].transform('mean')
        df_out['category_orders_std'] = df_out.groupby('category', observed=False)['num_orders'].transform('std')
    return df_out

def create_interaction_features(df):
    df_out = df.copy()
    interactions = {
        "price_diff_x_emailer": ("price_diff", "emailer_for_promotion"),
        "price_diff_x_home": ("price_diff", "homepage_featured"),
        "rolling_mean_2_x_emailer": ("num_orders_rolling_mean_2", "emailer_for_promotion"),
        "rolling_mean_2_x_home": ("num_orders_rolling_mean_2", "homepage_featured"),
    }
    for name, (feat1, feat2) in interactions.items():
        if feat1 in df_out.columns and feat2 in df_out.columns:
            df_out[name] = df_out[feat1] * df_out[feat2]
        else:
            df_out[name] = 0
    return df_out

def create_advanced_interactions(df):
    df_out = df.copy()
    demand_feats = [
        'num_orders_rolling_mean_2', 'num_orders_rolling_mean_5', 'num_orders_rolling_mean_14',
        'meal_orders_mean', 'center_orders_mean'
    ]
    price_feats = ['checkout_price', 'price_diff', 'discount_pct']
    promo_feats = ['emailer_for_promotion', 'homepage_featured']
    time_feats = ['weekofyear_sin', 'weekofyear_cos', 'month_sin', 'month_cos', 'mean_orders_by_weekofyear', 'mean_orders_by_month']

    # Polynomial features (squared, cubic) -- skip *_sin and *_cos and binary promo_feats
    top_feats = demand_feats + price_feats + promo_feats + time_feats
    for feat in top_feats:
        if feat in df_out.columns:
            if feat.endswith('_sin') or feat.endswith('_cos') or feat in ['emailer_for_promotion', 'homepage_featured']:
                continue  # Do not create *_sin_sq, *_cos_sq, or *_sq/_cube for binary promo_feats
            df_out[f'{feat}_sq'] = df_out[feat] ** 2
            df_out[f'{feat}_cube'] = df_out[feat] ** 3

    # Pairwise interactions: only between different groups
    def group_of(feat):
        if feat in demand_feats: return 'demand'
        if feat in price_feats: return 'price'
        if feat in promo_feats: return 'promo'
        if feat in time_feats: return 'time'
        return None
    pairwise_dict = {}
    for i, feat1 in enumerate(top_feats):
        for feat2 in top_feats[i+1:]:
            if feat1 in df_out.columns and feat2 in df_out.columns:
                if group_of(feat1) != group_of(feat2):
                    colname = f'{feat1}_x_{feat2}'
                    pairwise_dict[colname] = df_out[feat1] * df_out[feat2]
    if pairwise_dict:
        new_pairwise = {k: v for k, v in pairwise_dict.items() if k not in df_out.columns}
        if new_pairwise:
            df_out = pd.concat([df_out, pd.DataFrame(new_pairwise, index=df_out.index)], axis=1)

    # Third-order and fourth-order interactions (same logic)
    third_order_dict = {}
    for d in demand_feats:
        for p in price_feats:
            for m in promo_feats:
                if all(f in df_out.columns for f in [d, p, m]):
                    colname = f'{d}_x_{p}_x_{m}'
                    third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m]
    # Optionally, add a time feature (fourth-order) to the above, but only one per interaction
    # Thoughtful fourth-order interactions
    # 1. demand x price x promo x time (e.g., rolling mean x discount x promo x weekofyear_sin)
    for d in demand_feats:
        for p in price_feats:
            for m in promo_feats:
                for t in time_feats:
                    if all(f in df_out.columns for f in [d, p, m, t]):
                        # Only add a few of the most interpretable fourth-order interactions
                        if d == 'num_orders_rolling_mean_2' and p == 'discount_pct' and m == 'emailer_for_promotion' and t == 'weekofyear_sin':
                            colname = f'{d}_x_{p}_x_{m}_x_{t}'
                            third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m] * df_out[t]
                        if d == 'num_orders_rolling_mean_2' and p == 'discount_pct' and m == 'emailer_for_promotion' and t == 'weekofyear_cos':
                            colname = f'{d}_x_{p}_x_{m}_x_{t}'
                            third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m] * df_out[t]
                        if d == 'meal_orders_mean' and p == 'price_diff' and m == 'homepage_featured' and t == 'month_sin':
                            colname = f'{d}_x_{p}_x_{m}_x_{t}'
                            third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m] * df_out[t]
                        if d == 'meal_orders_mean' and p == 'price_diff' and m == 'homepage_featured' and t == 'month_cos':
                            colname = f'{d}_x_{p}_x_{m}_x_{t}'
                            third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m] * df_out[t]
    # Additional thoughtful third- and fourth-order interactions
    # 1. Demand × Promotion × Time
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_5', 'emailer_for_promotion', 'weekofyear_sin']):
        third_order_dict['num_orders_rolling_mean_5_x_emailer_for_promotion_x_weekofyear_sin'] = (
            df_out['num_orders_rolling_mean_5'] * df_out['emailer_for_promotion'] * df_out['weekofyear_sin'])
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_5', 'emailer_for_promotion', 'weekofyear_cos']):
        third_order_dict['num_orders_rolling_mean_5_x_emailer_for_promotion_x_weekofyear_cos'] = (
            df_out['num_orders_rolling_mean_5'] * df_out['emailer_for_promotion'] * df_out['weekofyear_cos'])
    if all(f in df_out.columns for f in ['meal_orders_mean', 'homepage_featured', 'month_sin']):
        third_order_dict['meal_orders_mean_x_homepage_featured_x_month_sin'] = (
            df_out['meal_orders_mean'] * df_out['homepage_featured'] * df_out['month_sin'])
    if all(f in df_out.columns for f in ['meal_orders_mean', 'homepage_featured', 'month_cos']):
        third_order_dict['meal_orders_mean_x_homepage_featured_x_month_cos'] = (
            df_out['meal_orders_mean'] * df_out['homepage_featured'] * df_out['month_cos'])
    # 2. Demand × Price × Time
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_2', 'discount_pct', 'month_sin']):
        third_order_dict['num_orders_rolling_mean_2_x_discount_pct_x_month_sin'] = (
            df_out['num_orders_rolling_mean_2'] * df_out['discount_pct'] * df_out['month_sin'])
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_2', 'discount_pct', 'month_cos']):
        third_order_dict['num_orders_rolling_mean_2_x_discount_pct_x_month_cos'] = (
            df_out['num_orders_rolling_mean_2'] * df_out['discount_pct'] * df_out['month_cos'])
    if all(f in df_out.columns for f in ['center_orders_mean', 'price_diff', 'weekofyear_sin']):
        third_order_dict['center_orders_mean_x_price_diff_x_weekofyear_sin'] = (
            df_out['center_orders_mean'] * df_out['price_diff'] * df_out['weekofyear_sin'])
    if all(f in df_out.columns for f in ['center_orders_mean', 'price_diff', 'weekofyear_cos']):
        third_order_dict['center_orders_mean_x_price_diff_x_weekofyear_cos'] = (
            df_out['center_orders_mean'] * df_out['price_diff'] * df_out['weekofyear_cos'])
    # 3. Promotion × Price × Time
    if all(f in df_out.columns for f in ['emailer_for_promotion', 'discount_pct', 'weekofyear_sin']):
        third_order_dict['emailer_for_promotion_x_discount_pct_x_weekofyear_sin'] = (
            df_out['emailer_for_promotion'] * df_out['discount_pct'] * df_out['weekofyear_sin'])
    if all(f in df_out.columns for f in ['emailer_for_promotion', 'discount_pct', 'weekofyear_cos']):
        third_order_dict['emailer_for_promotion_x_discount_pct_x_weekofyear_cos'] = (
            df_out['emailer_for_promotion'] * df_out['discount_pct'] * df_out['weekofyear_cos'])
    if all(f in df_out.columns for f in ['homepage_featured', 'price_diff', 'month_sin']):
        third_order_dict['homepage_featured_x_price_diff_x_month_sin'] = (
            df_out['homepage_featured'] * df_out['price_diff'] * df_out['month_sin'])
    if all(f in df_out.columns for f in ['homepage_featured', 'price_diff', 'month_cos']):
        third_order_dict['homepage_featured_x_price_diff_x_month_cos'] = (
            df_out['homepage_featured'] * df_out['price_diff'] * df_out['month_cos'])
    # 4. Demand × Price × Promotion × Time (fourth-order)
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_5', 'discount_pct', 'emailer_for_promotion', 'weekofyear_sin']):
        third_order_dict['num_orders_rolling_mean_5_x_discount_pct_x_emailer_for_promotion_x_weekofyear_sin'] = (
            df_out['num_orders_rolling_mean_5'] * df_out['discount_pct'] * df_out['emailer_for_promotion'] * df_out['weekofyear_sin'])
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_5', 'discount_pct', 'emailer_for_promotion', 'weekofyear_cos']):
        third_order_dict['num_orders_rolling_mean_5_x_discount_pct_x_emailer_for_promotion_x_weekofyear_cos'] = (
            df_out['num_orders_rolling_mean_5'] * df_out['discount_pct'] * df_out['emailer_for_promotion'] * df_out['weekofyear_cos'])
    if all(f in df_out.columns for f in ['meal_orders_mean', 'price_diff', 'homepage_featured', 'month_sin']):
        third_order_dict['meal_orders_mean_x_price_diff_x_homepage_featured_x_month_sin'] = (
            df_out['meal_orders_mean'] * df_out['price_diff'] * df_out['homepage_featured'] * df_out['month_sin'])
    if all(f in df_out.columns for f in ['meal_orders_mean', 'price_diff', 'homepage_featured', 'month_cos']):
        third_order_dict['meal_orders_mean_x_price_diff_x_homepage_featured_x_month_cos'] = (
            df_out['meal_orders_mean'] * df_out['price_diff'] * df_out['homepage_featured'] * df_out['month_cos'])
    # Add third- and selected fourth-order features
    new_cols = {k: v for k, v in third_order_dict.items() if k not in df_out.columns}
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)
    return df_out

# --- Seasonality Smoothing and Outlier Flags ---
def add_seasonality_features(df, weekofyear_means=None, month_means=None, is_train=True):
    df = df.copy()
    if is_train:
        weekofyear_means = df.groupby('weekofyear')['num_orders'].mean()
        month_means = df.groupby('month')['num_orders'].mean()
    else:
        if weekofyear_means is None or month_means is None:
            raise ValueError("When is_train=False, weekofyear_means and month_means must be provided (not None).")
    df['mean_orders_by_weekofyear'] = df['weekofyear'].map(weekofyear_means)
    df['mean_orders_by_month'] = df['month'].map(month_means)
    df['is_outlier_weekofyear'] = df['weekofyear'].isin([5, 48]).astype(int)
    df['is_outlier_month'] = df['month'].isin([2]).astype(int)
    return df, weekofyear_means, month_means

def apply_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None):
    df_out = df.copy()
    df_out = create_temporal_features(df_out)
    if is_train or 'num_orders' in df_out.columns:
        df_out = create_lag_rolling_features(df_out)
    df_out = create_other_features(df_out)
    df_out = create_group_aggregates(df_out)
    df_out = create_interaction_features(df_out)
    df_out = create_advanced_interactions(df_out)
    # Add smoothed seasonality and outlier flags
    df_out, weekofyear_means, month_means = add_seasonality_features(df_out, weekofyear_means, month_means, is_train=is_train)
    # Fill NaNs for all engineered features
    lag_roll_diff_cols = [col for col in df_out.columns if any(sub in col for sub in [
        "lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home",
        "_x_discount_pct", "_x_price_diff", "_x_weekofyear", "_sq", "_cube", "_mean", "_std"
    ])]
    cols_to_fill = [col for col in lag_roll_diff_cols if col in df_out.columns and len(df_out[col]) == len(df_out)]
    if cols_to_fill:
        df_out.loc[:, cols_to_fill] = df_out[cols_to_fill].fillna(0)
    if "discount_pct" in df_out.columns:
        df_out["discount_pct"] = df_out["discount_pct"].fillna(0)
    # Defragment and deduplicate DataFrame ONCE at the end
    df_out = df_out.copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    return df_out, weekofyear_means, month_means

# --- One-hot encoding and feature engineering for train/test ---
logging.info("Applying feature engineering (no one-hot for native categoricals)...")
df_full = pd.concat([df, test], ignore_index=True)
df_full = create_other_features(df_full)
# Do NOT one-hot encode category, cuisine, center_type; use native categorical handling
# Remove any one-hot columns if present (from previous runs or merges)
for prefix in ["category_", "cuisine_", "center_type_"]:
    df_full = df_full.loc[:, ~df_full.columns.str.startswith(prefix)]

train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()

# --- Add seasonality features with smoothed means and outlier flags ---
train_df, weekofyear_means, month_means = apply_feature_engineering(train_df, is_train=True)
test_df, _, _ = apply_feature_engineering(test_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means)

# --- Set native categorical dtypes for LightGBM ---
CATEGORICAL_FEATURES = [col for col in ["category", "cuisine", "center_type", "center_id", "meal_id"] if col in train_df.columns]
for df_ in [train_df, test_df]:
    for col in CATEGORICAL_FEATURES:
        df_[col] = df_[col].astype("category")
# valid_df will be defined after train/valid split, so set dtypes after that

# Drop rows in train_df where target is NA (if any, though unlikely from problem desc)
train_df = train_df.dropna(subset=['num_orders']).reset_index(drop=True)


# --- Define Features and Target ---
TARGET = "num_orders"
features_set = set()
FEATURES = []

# Add base features
base_features = [
    "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", "weekofyear",
    "category", "cuisine", "center_type", "center_id", "meal_id"
]
for f in base_features:
    if f in train_df.columns and f not in features_set:
        FEATURES.append(f)
        features_set.add(f)

# Add rolling means/stds
for w in ROLLING_WINDOWS:
    mean_col = f"{TARGET}_rolling_mean_{w}"
    std_col = f"{TARGET}_rolling_std_{w}"
    if mean_col in train_df.columns and mean_col not in features_set:
        FEATURES.append(mean_col)
        features_set.add(mean_col)
    if std_col in train_df.columns and std_col not in features_set:
        FEATURES.append(std_col)
        features_set.add(std_col)

# Add rolling sums
for col in OTHER_ROLLING_SUM_COLS:
    sum_col = f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}"
    if sum_col in train_df.columns and sum_col not in features_set:
        FEATURES.append(sum_col)
        features_set.add(sum_col)

# Add interaction and advanced features
for col in train_df.columns:
    if (col.startswith("price_diff_x_") or col.startswith("rolling_mean_2_x_") or col.endswith("_sq") or "_x_" in col) and col not in features_set and col != TARGET and col != 'id':
        FEATURES.append(col)
        features_set.add(col)

# Add group-level aggregates
for prefix in ["center_orders_", "meal_orders_", "category_orders_"]:
    for col in train_df.columns:
        if col.startswith(prefix) and col not in features_set and col != TARGET and col != 'id':
            FEATURES.append(col)
            features_set.add(col)

# Remove one-hot columns for categoricals from FEATURES if present
for prefix in ["category_", "cuisine_", "center_type_"]:
    FEATURES = [f for f in FEATURES if not f.startswith(prefix)]

# Add seasonality and outlier features
seasonality_features = [
    'weekofyear_sin', 'weekofyear_cos', 'month_sin', 'month_cos',
    'mean_orders_by_weekofyear', 'mean_orders_by_month',
    'is_outlier_weekofyear', 'is_outlier_month'
]
for f in seasonality_features:
    if f in train_df.columns and f not in features_set:
        FEATURES.append(f)
        features_set.add(f)
# Remove raw integer weekofyear/month if present
for f in ['weekofyear', 'month']:
    if f in FEATURES:
        FEATURES.remove(f)
        features_set.discard(f)

logging.info(f"Using {len(FEATURES)} features: {FEATURES}")


# --- Remove manually identified highly correlated features ---
features_to_remove = []
FEATURES = [f for f in FEATURES if f not in features_to_remove]
logging.info(f"Removed manually identified correlated features. {len(FEATURES)} features remain.")

# --- Train/validation split ---
max_week = train_df["week"].max()
valid_df = train_df[train_df["week"] > max_week - VALIDATION_WEEKS].copy()
train_split_df = train_df[train_df["week"] <= max_week - VALIDATION_WEEKS].copy()

# Set categorical dtype for valid_df
for col in CATEGORICAL_FEATURES:
    if col in valid_df.columns:
        valid_df[col] = valid_df[col].astype("category")

logging.info(f"Train split shape: {train_split_df.shape}, Validation shape: {valid_df.shape}")

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0) # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False # lower is better

# --- Custom Early Stopping Callback with Overfitting Detection ---
def early_stopping_with_overfit(stopping_rounds=100, overfit_rounds=20, verbose=False):
    """
    Custom LightGBM callback for early stopping with overfitting detection.
    Stops if validation loss doesn't improve for `stopping_rounds` OR
    if validation loss increases for `overfit_rounds` while training loss decreases.
    """
    best_score = [float('inf')]
    best_iter = [0]
    overfit_count = [0]
    prev_train_loss = [float('inf')]
    prev_valid_loss = [float('inf')]
    def _callback(env):
        # Find train and valid loss
        train_loss = None
        valid_loss = None
        for eval_tuple in env.evaluation_result_list:
            name = eval_tuple[0]
            loss = eval_tuple[1]
            if 'train' in name:
                train_loss = loss
            elif 'valid' in name or 'validation' in name:
                valid_loss = loss
        if valid_loss is None or train_loss is None:
            return
        # Early stopping (standard)
        if valid_loss < best_score[0]:
            best_score[0] = valid_loss
            best_iter[0] = env.iteration
            overfit_count[0] = 0
        else:
            # Overfitting detection: valid loss increases, train loss decreases
            if valid_loss > prev_valid_loss[0] and train_loss < prev_train_loss[0]:
                overfit_count[0] += 1
            else:
                overfit_count[0] = 0
        prev_train_loss[0] = train_loss
        prev_valid_loss[0] = valid_loss
        # Verbose
        if verbose and env.iteration % 10 == 0:
            print(f"[Iter {env.iteration}] train: {train_loss:.5f}, valid: {valid_loss:.5f}, overfit_count: {overfit_count[0]}")
        # Stop if overfitting detected
        if overfit_count[0] >= overfit_rounds:
            if verbose:
                print(f"Stopping early due to overfitting at iteration {env.iteration}")
            raise lgb.callback.EarlyStopException(env.iteration, best_score[0])
        # Standard early stopping
        if env.iteration - best_iter[0] >= stopping_rounds:
            if verbose:
                print(f"Stopping early due to no improvement at iteration {env.iteration}")
            raise lgb.callback.EarlyStopException(env.iteration, best_score[0])
    return _callback

# --- Final Model Training ---
logging.info("Training final model on full training data with best params...")
# Use best_params from Optuna feature+hyperparam selection, merged with fixed params
final_params = {
    'objective': 'regression_l1',
    'boosting_type': 'gbdt',
    'n_estimators': 3000, # Increase slightly for final training
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'metric': 'None'
}
# best_params is set after Optuna feature+hyperparam selection below
# final_params will be updated after best_params is defined

# --- Optuna Feature Selection and Hyperparameter Tuning in a Single CV Loop ---
def optuna_feature_selection_and_hyperparam_objective(trial):
    # Hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'objective': 'regression_l1',
        'n_estimators': 500,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric': 'None',
    }
    # Only set bagging params if not using GOSS
    if params['boosting_type'] != 'goss':
        params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 1.0)
        params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 7)
    else:
        params['bagging_fraction'] = 1.0
        params['bagging_freq'] = 0
    # Find all features with sin/cos in their name (excluding those already in a pair)
    sincos_features = [f for f in FEATURES if re.search(r'(_sin|_cos)', f)]
    # Group into pairs by prefix (e.g. 'num_orders_rolling_mean_2_x_weekofyear')
    pair_prefixes = set()
    pair_map = {}
    for f in sincos_features:
        m = re.match(r'(.*)_sin$', f)
        if m and f.replace('_sin', '_cos') in sincos_features:
            prefix = m.group(1)
            pair_prefixes.add(prefix)
            pair_map[prefix] = (f, f.replace('_sin', '_cos'))
    # For each pair, add a trial param
    selected_features = []
    for prefix, (sin, cos) in pair_map.items():
        pair_name = f"{sin}_{cos}_pair"
        if trial.suggest_categorical(pair_name, [True, False]):
            selected_features.extend([sin, cos])
    # Only tune non-sin/cos features individually
    selected_features += [f for f in FEATURES if (f not in sincos_features) and trial.suggest_categorical(f, [True, False])]
    if len(selected_features) < 10:
        return float('inf')
    tscv = TimeSeriesSplit(n_splits=3)
    return np.mean([
        rmsle(
            train_split_df.iloc[valid_idx][TARGET],
            LGBMRegressor(**params).fit(
                train_split_df.iloc[train_idx][selected_features],
                train_split_df.iloc[train_idx][TARGET],
                eval_set=[(train_split_df.iloc[train_idx][selected_features], train_split_df.iloc[train_idx][TARGET]),
                          (train_split_df.iloc[valid_idx][selected_features], train_split_df.iloc[valid_idx][TARGET])],
                eval_metric=lgb_rmsle,
                callbacks=[early_stopping_with_overfit(100, 20, verbose=False)]
            ).predict(train_split_df.iloc[valid_idx][selected_features])
        )
        for train_idx, valid_idx in tscv.split(train_split_df)
    ])

# --- Optuna Feature Selection + Hyperparameter Tuning ---
logging.info("Starting Optuna feature+hyperparam selection (Cross Validation)...")
# Reduce Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

class TqdmOptunaCallback:
    def __init__(self, n_trials, study=None, print_every=1):
        self.pbar = tqdm(total=n_trials, desc="Optuna Trials", position=0, leave=True)
        self.print_every = print_every
        # Initialize best_value and best_trial from study if available
        if study is not None and study.best_trial is not None and study.best_trial.value is not None:
            self.best_value = study.best_trial.value
            self.best_trial = study.best_trial.number
        else:
            self.best_value = float('inf')
            self.best_trial = None
    def __call__(self, study, trial):
        self.pbar.update(1)
        msg = None
        # Count number of features selected (if available)
        n_features = sum([v for k, v in trial.params.items() if isinstance(v, bool) and v])
        # Show only main hyperparameters (not feature selectors)
        main_params = {k: v for k, v in trial.params.items() if not isinstance(v, bool) and not k.endswith('_pair')}
        def fmt_val(val):
            if isinstance(val, float):
                return f"{val:.6f}"
            return str(val)
        params_str = ', '.join(f"{k}={fmt_val(v)}" for k, v in list(main_params.items())[:5])
        if trial.value is not None and trial.value < self.best_value:
            self.best_value = trial.value
            self.best_trial = trial.number
            # ANSI green for new best
            msg = f"\033[92mTrial {trial.number} finished with value: {trial.value:.5f} | BEST! {self.best_value:.5f}\033[0m | Features: {n_features} | Params: {params_str}"
        elif trial.number % self.print_every == 0:
            msg = f"Trial {trial.number} finished with value: {trial.value:.5f} | Best: {self.best_value:.5f} | Features: {n_features} | Params: {params_str}"
        if msg:
            tqdm.write(msg)
    def close(self):
        self.pbar.close()

# Create the study first
optuna_storage = OPTUNA_DB
feature_hyperparam_study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=optuna_storage, load_if_exists=True)
# Pass the study to the callback so it can initialize best_value/best_trial
optuna_callback = TqdmOptunaCallback(OPTUNA_TRIALS, study=feature_hyperparam_study, print_every=1)
feature_hyperparam_study.optimize(optuna_feature_selection_and_hyperparam_objective, n_trials=OPTUNA_TRIALS, timeout=7200, callbacks=[optuna_callback], n_jobs=OPTUNA_NJOBS)
optuna_callback.close()

# Extract best features and params
best_mask = [feature_hyperparam_study.best_trial.params.get(f, False) for f in FEATURES]
SELECTED_FEATURES = [f for f, keep in zip(FEATURES, best_mask) if keep]
best_params = {k: v for k, v in feature_hyperparam_study.best_trial.params.items() if k not in FEATURES and not k.endswith('_pair')}
selected_pairs = {k: v for k, v in feature_hyperparam_study.best_trial.params.items() if k.endswith('_pair')}
logging.info(f"Optuna-selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

# Add both features from each selected cyclical pair
for pair_name, is_selected in selected_pairs.items():
    if is_selected:
        # Remove '_pair' and split to get the two feature names
        pair_feats = pair_name[:-5].split('_')
        # Find the split point between the two features (look for the second feature's suffix)
        # This assumes the format is always ..._sin_..._cos_pair or similar
        for i in range(1, len(pair_feats)):
            if pair_feats[i].endswith('sin') or pair_feats[i].endswith('cos'):
                feat1 = '_'.join(pair_feats[:i+1])
                feat2 = '_'.join(pair_feats[i+1:])
                # Add both features if they exist in train_df and not already in SELECTED_FEATURES
                for feat in [feat1, feat2]:
                    if feat and feat in train_df.columns and feat not in SELECTED_FEATURES:
                        SELECTED_FEATURES.append(feat)
                break

logging.info(f"Optuna-selected params: {best_params}")
logging.info(f"Optuna-selected cyclical pairs: {selected_pairs}")
logging.info(f"Optuna-selected features after: ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

# Use SELECTED_FEATURES for final model and ensemble
FEATURES = SELECTED_FEATURES
final_params.update(best_params)

# --- Set native categorical types for optimal LightGBM usage ---
for col in ["category", "cuisine", "center_type", "center_id", "meal_id"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype("category")
    if col in valid_df.columns:
        valid_df[col] = valid_df[col].astype("category")
    if col in test_df.columns:
        test_df[col] = test_df[col].astype("category")
CATEGORICAL_FEATURES = [col for col in ["category", "cuisine", "center_type", "center_id", "meal_id"] if col in FEATURES]

# --- Recursive Prediction and Ensemble Utilities ---
def recursive_predict(model, train_df, predict_df, FEATURES, weekofyear_means=None, month_means=None):
    """
    Perform recursive prediction for a given predict_df (test or validation set).
    """
    # Combine train and predict set, sort by time
    history_df = pd.concat([train_df, predict_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    predict_weeks = sorted(predict_df['week'].unique())
    for week_num in predict_weeks:
        current_week_mask = history_df['week'] == week_num
        # Recompute features for all rows (including updated num_orders)
        history_df, _, _ = apply_feature_engineering(
            history_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means
        ) if weekofyear_means is not None and month_means is not None else apply_feature_engineering(history_df, is_train=False)
        current_features = history_df.loc[current_week_mask, FEATURES]
        for col in [col for col in FEATURES if col not in current_features.columns]:
            current_features[col] = 0
        current_preds = np.clip(model.predict(current_features[FEATURES]), 0, None).round().astype(float)
        history_df.loc[current_week_mask, 'num_orders'] = current_preds
    final_predictions = history_df.loc[history_df['id'].isin(predict_df['id']), ['id', 'num_orders']].copy()
    final_predictions['num_orders'] = final_predictions['num_orders'].round().astype(int)
    final_predictions['id'] = final_predictions['id'].astype(int)
    return final_predictions.set_index('id')['num_orders']

def recursive_ensemble(train_df, test_df, FEATURES, weekofyear_means=None, month_means=None, n_models=5, eval_metric=None):
    preds_list = []
    models = []
    for i in tqdm(range(n_models), desc="Ensemble Models", position=0):
        logging.info(f"Training ensemble model {i+1}/{n_models}...")
        params = final_params.copy(); params.pop('seed', None)
        model = LGBMRegressor(**params, seed=SEED+i)
        if eval_metric:
            model.fit(
                train_df[FEATURES], train_df[TARGET],
                eval_set=[(train_df[FEATURES], train_df[TARGET]), (valid_df[FEATURES], valid_df[TARGET])],
                eval_metric=eval_metric,
                callbacks=[early_stopping_with_overfit(300, 20, verbose=False)],
                categorical_feature=CATEGORICAL_FEATURES
            )
        else:
            model.fit(train_df[FEATURES], train_df[TARGET], categorical_feature=CATEGORICAL_FEATURES)
        preds_list.append(recursive_predict(model, train_df, test_df, FEATURES, weekofyear_means, month_means).values)
        models.append(model)
    return np.mean(preds_list, axis=0).round().astype(int), models

# --- Recursive Ensemble Prediction with Selected Features ---
logging.info("Running recursive ensemble prediction with selected features...")
ensemble_preds, ensemble_models = recursive_ensemble(train_df, test_df, FEATURES, weekofyear_means, month_means, n_models=5, eval_metric=lgb_rmsle)
final_predictions_df = pd.DataFrame({'id': test_df['id'].astype(int), 'num_orders': ensemble_preds})
submission_path_ensemble_final = os.path.join(OUTPUT_DIRECTORY, f"{SUBMISSION_FILE_PREFIX}_final_optuna_ensemble.csv")
final_predictions_df.to_csv(submission_path_ensemble_final, index=False)
logging.info(f"Ensemble submission file saved to {submission_path_ensemble_final}")

# --- SHAP Analysis for Ensemble (using first model as representative) ---
logging.info("Calculating SHAP values for ensemble (using first model)...")
try:
    if len(train_df) > N_SHAP_SAMPLES:
        shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
    else:
        shap_sample = train_df.copy()
    explainer = shap.TreeExplainer(ensemble_models[0])
    shap_values = explainer.shap_values(shap_sample[FEATURES])
    shap_values_df = pd.DataFrame(shap_values, columns=FEATURES)
    shap_values_df.to_csv(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_values.csv"), index=False)
    np.save(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_values.npy"), shap_values)
    shap_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_feature_importances.csv"), index=False)
    plt.figure()
    shap.summary_plot(shap_values, shap_sample[FEATURES], show=False, max_display=len(FEATURES))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_summary_all_features.png"))
    plt.close()
    plt.figure(figsize=(10, 8))
    shap_importance_df.head(20).plot(kind='barh', x='feature', y='mean_abs_shap', legend=False, figsize=(10, 8))
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
    plt.title('Top 20 SHAP Feature Importances (Ensemble)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_top20_importance.png"))
    plt.close()
    for feat in shap_importance_df['feature']:
        shap.dependence_plot(feat, shap_values, shap_sample[FEATURES], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_dependence_{feat}.png"))
        plt.close()
    try:
        shap_interaction_values = explainer.shap_interaction_values(shap_sample[FEATURES])
        plt.figure()
        shap.summary_plot(shap_interaction_values, shap_sample[FEATURES], show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_interaction_summary.png"))
        plt.close()
    except Exception as e:
        logging.warning(f"Could not generate SHAP interaction summary plot: {e}")
    logging.info("SHAP analysis saved for ensemble.")
except Exception as e:
    logging.error(f"Error during SHAP analysis for ensemble: {e}")

# --- Validation Plot (using recursive predictions) ---
logging.info("Generating validation plot (using recursive predictions)...")
try:
    valid_preds_recursive = recursive_predict(
        ensemble_models[0], train_split_df, valid_df, FEATURES, weekofyear_means, month_means
    )
    valid_rmsle_recursive = rmsle(valid_df[TARGET].values, valid_preds_recursive.loc[valid_df['id']].values)
    plt.figure(figsize=(15, 6))
    plt.scatter(valid_df[TARGET], valid_preds_recursive.loc[valid_df['id']], alpha=0.5, s=10)
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2, label='Ideal')
    plt.xlabel("Actual Orders (Validation Set)")
    plt.ylabel("Predicted Orders (Validation Set)")
    plt.title(f"Actual vs. Predicted Orders (Validation Set, Recursive) - RMSLE: {valid_rmsle_recursive:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, "validation_actual_vs_predicted_ensemble_recursive.png"))
    plt.close()
    logging.info("Validation plot (recursive) saved.")
except Exception as e:
    logging.error(f"Error during plotting (recursive): {e}")
# --- End of Script ---
logging.info("All tasks completed.")

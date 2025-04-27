import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import logging
import lightgbm as lgb  # Added for early stopping callback
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
# Use a single rolling window configuration for clarity
ROLLING_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28]
# Other features (not directly dependent on recursive prediction)
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 50 # Number of Optuna trials
OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "submission_recursive"
SHAP_FILE_PREFIX = "shap_recursive"
N_SHAP_SAMPLES = 2000

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Data ---
logging.info("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    meal_info = pd.read_csv(MEAL_INFO_PATH)
    center_info = pd.read_csv(CENTER_INFO_PATH)
except FileNotFoundError as e:
    logging.error(f"Error loading data file: {e}. Ensure train.csv, test.csv, meal_info.csv, and fulfilment_center_info.csv are present.")
    raise

# --- Preprocessing ---
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

# --- Feature Engineering ---
logging.info("Creating features (IMPROVED)...")
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
    group = df_out.groupby(GROUP_COLS)
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)
    return df_out

def create_other_features(df):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = group["checkout_price"].diff()
    for col in OTHER_ROLLING_SUM_COLS:
        shifted = group[col].shift(1)
        df_out[f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}"] = shifted.rolling(OTHER_ROLLING_SUM_WINDOW, min_periods=1).sum().reset_index(drop=True)
    return df_out

def create_group_aggregates(df):
    df_out = df.copy()
    # Center-level aggregates
    df_out['center_orders_mean'] = df_out.groupby('center_id')['num_orders'].transform('mean')
    df_out['center_orders_std'] = df_out.groupby('center_id')['num_orders'].transform('std')
    # Meal-level aggregates
    df_out['meal_orders_mean'] = df_out.groupby('meal_id')['num_orders'].transform('mean')
    df_out['meal_orders_std'] = df_out.groupby('meal_id')['num_orders'].transform('std')
    # Category-level aggregates (if available)
    if 'category' in df_out.columns:
        df_out['category_orders_mean'] = df_out.groupby('category')['num_orders'].transform('mean')
        df_out['category_orders_std'] = df_out.groupby('category')['num_orders'].transform('std')
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
    # Define feature groups
    demand_feats = [
        'num_orders_rolling_mean_2', 'num_orders_rolling_mean_5', 'num_orders_rolling_mean_14',
        'meal_orders_mean', 'center_orders_mean'
    ]
    price_feats = ['checkout_price', 'price_diff', 'discount_pct']
    promo_feats = ['emailer_for_promotion', 'homepage_featured']
    time_feats = ['weekofyear_sin', 'weekofyear_cos', 'month_sin', 'month_cos', 'mean_orders_by_weekofyear', 'mean_orders_by_month']

    # Polynomial features (squared, cubic)
    top_feats = demand_feats + price_feats + promo_feats + time_feats
    for feat in top_feats:
        if feat in df_out.columns:
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
        df_out = pd.concat([df_out, pd.DataFrame(pairwise_dict, index=df_out.index)], axis=1)

    # Third-order interactions: one from each of three different groups (never more than one time feature)
    third_order_dict = {}
    for d in demand_feats:
        for p in price_feats:
            for m in promo_feats:
                if all(f in df_out.columns for f in [d, p, m]):
                    colname = f'{d}_x_{p}_x_{m}'
                    third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m]
    # Optionally, add a time feature (fourth-order) to the above, but only one per interaction
    # Thoughtful fourth-order examples:
    # 1. demand x price x promo x time (e.g., rolling mean x discount x promo x weekofyear_sin)
    for d in demand_feats:
        for p in price_feats:
            for m in promo_feats:
                for t in time_feats:
                    if all(f in df_out.columns for f in [d, p, m, t]):
                        # Only add a couple of the most interpretable fourth-order interactions
                        if d == 'num_orders_rolling_mean_2' and p == 'discount_pct' and m == 'emailer_for_promotion' and t == 'weekofyear_sin':
                            colname = f'{d}_x_{p}_x_{m}_x_{t}'
                            third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m] * df_out[t]
                        if d == 'meal_orders_mean' and p == 'price_diff' and m == 'homepage_featured' and t == 'month_cos':
                            colname = f'{d}_x_{p}_x_{m}_x_{t}'
                            third_order_dict[colname] = df_out[d] * df_out[p] * df_out[m] * df_out[t]
    # Additional thoughtful third- and fourth-order interactions
    # 1. Demand × Promotion × Time
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_5', 'emailer_for_promotion', 'weekofyear_cos']):
        third_order_dict['num_orders_rolling_mean_5_x_emailer_for_promotion_x_weekofyear_cos'] = (
            df_out['num_orders_rolling_mean_5'] * df_out['emailer_for_promotion'] * df_out['weekofyear_cos'])
    if all(f in df_out.columns for f in ['meal_orders_mean', 'homepage_featured', 'month_sin']):
        third_order_dict['meal_orders_mean_x_homepage_featured_x_month_sin'] = (
            df_out['meal_orders_mean'] * df_out['homepage_featured'] * df_out['month_sin'])
    # 2. Demand × Price × Time
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_2', 'discount_pct', 'month_cos']):
        third_order_dict['num_orders_rolling_mean_2_x_discount_pct_x_month_cos'] = (
            df_out['num_orders_rolling_mean_2'] * df_out['discount_pct'] * df_out['month_cos'])
    if all(f in df_out.columns for f in ['center_orders_mean', 'price_diff', 'weekofyear_sin']):
        third_order_dict['center_orders_mean_x_price_diff_x_weekofyear_sin'] = (
            df_out['center_orders_mean'] * df_out['price_diff'] * df_out['weekofyear_sin'])
    # 3. Promotion × Price × Time
    if all(f in df_out.columns for f in ['emailer_for_promotion', 'discount_pct', 'weekofyear_sin']):
        third_order_dict['emailer_for_promotion_x_discount_pct_x_weekofyear_sin'] = (
            df_out['emailer_for_promotion'] * df_out['discount_pct'] * df_out['weekofyear_sin'])
    if all(f in df_out.columns for f in ['homepage_featured', 'price_diff', 'month_cos']):
        third_order_dict['homepage_featured_x_price_diff_x_month_cos'] = (
            df_out['homepage_featured'] * df_out['price_diff'] * df_out['month_cos'])
    # 4. Demand × Price × Promotion × Time (fourth-order)
    if all(f in df_out.columns for f in ['num_orders_rolling_mean_5', 'discount_pct', 'emailer_for_promotion', 'weekofyear_cos']):
        third_order_dict['num_orders_rolling_mean_5_x_discount_pct_x_emailer_for_promotion_x_weekofyear_cos'] = (
            df_out['num_orders_rolling_mean_5'] * df_out['discount_pct'] * df_out['emailer_for_promotion'] * df_out['weekofyear_cos'])
    if all(f in df_out.columns for f in ['meal_orders_mean', 'price_diff', 'homepage_featured', 'month_sin']):
        third_order_dict['meal_orders_mean_x_price_diff_x_homepage_featured_x_month_sin'] = (
            df_out['meal_orders_mean'] * df_out['price_diff'] * df_out['homepage_featured'] * df_out['month_sin'])
    # Add third- and selected fourth-order features
    new_cols = {k: v for k, v in third_order_dict.items() if k not in df_out.columns}
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)
    return df_out

# --- Seasonality Smoothing and Outlier Flags ---
def add_seasonality_features(df, weekofyear_means=None, month_means=None, is_train=True):
    df = df.copy()
    # Smoothed mean demand by weekofyear/month
    if is_train:
        weekofyear_means = df.groupby('weekofyear')['num_orders'].mean()
        month_means = df.groupby('month')['num_orders'].mean()
    df['mean_orders_by_weekofyear'] = df['weekofyear'].map(weekofyear_means)
    df['mean_orders_by_month'] = df['month'].map(month_means)
    # Outlier flags
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
    # Only fill columns that exist in the DataFrame and have the correct length
    cols_to_fill = [col for col in lag_roll_diff_cols if col in df_out.columns and len(df_out[col]) == len(df_out)]
    if cols_to_fill:
        df_out.loc[:, cols_to_fill] = df_out[cols_to_fill].fillna(0)
    if "discount_pct" in df_out.columns:
        df_out["discount_pct"] = df_out["discount_pct"].fillna(0)
    # Defragment DataFrame to avoid future fragmentation issues
    df_out = df_out.copy()
    return df_out, weekofyear_means, month_means

# --- One-hot encoding and feature engineering for train/test ---
logging.info("Applying one-hot encoding and feature engineering...")
df_full = pd.concat([df, test], ignore_index=True)
df_full = create_other_features(df_full)
cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in df_full.columns]
if cat_cols:
    df_full = pd.get_dummies(df_full, columns=cat_cols, dummy_na=False) # Avoid NaN columns from dummies

train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()

# --- Add seasonality features with smoothed means and outlier flags ---
train_df, weekofyear_means, month_means = apply_feature_engineering(train_df, is_train=True)
test_df, _, _ = apply_feature_engineering(test_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means)

# Drop rows in train_df where target is NA (if any, though unlikely from problem desc)
train_df = train_df.dropna(subset=['num_orders']).reset_index(drop=True)


# --- Define Features and Target ---
TARGET = "num_orders"
features_set = set()
FEATURES = []

# Add base features
base_features = [
    "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", "weekofyear"
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

# Add one-hot columns if present
for prefix in ["category_", "cuisine_", "center_type_"]:
    for col in train_df.columns:
        if col.startswith(prefix) and col not in features_set and col != TARGET and col != 'id':
            FEATURES.append(col)
            features_set.add(col)

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
features_to_remove = [
  'base_price',
  'discount',
  'num_orders_rolling_mean_3',
  'num_orders_rolling_mean_5',
  'num_orders_rolling_std_5',
  'num_orders_rolling_mean_6',
  'num_orders_rolling_mean_7',
  'num_orders_rolling_std_7',
  'num_orders_rolling_mean_8',
  'num_orders_rolling_std_10',
  'num_orders_rolling_mean_12',
  'num_orders_rolling_std_12',
  'num_orders_rolling_mean_14',
  'num_orders_rolling_mean_21',
  'price_diff_x_emailer',
  'checkout_price_sq',
  'emailer_for_promotion_sq',
  'homepage_featured_sq',
  'num_orders_rolling_mean_2_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_homepage_featured',
  'price_diff_x_emailer_for_promotion',
  'price_diff_x_homepage_featured',
  'num_orders_rolling_mean_2_x_discount_pct_x_homepage_featured',
  'center_orders_mean',
  'center_orders_mean_cube',
  'meal_orders_mean',
  'meal_orders_mean_cube',
  # Added from SHAP importances for further testing:
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_14',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5',
  'num_orders_rolling_mean_2_x_meal_orders_mean',
  'meal_orders_mean_x_center_orders_mean',
  'num_orders_rolling_mean_2_x_center_orders_mean',
  'checkout_price',
  'num_orders_rolling_mean_2',
  'mean_orders_by_weekofyear',
  'num_orders_rolling_mean_5_x_num_orders_rolling_mean_14',
  'num_orders_rolling_mean_4',
  'meal_orders_mean_x_homepage_featured',
  'num_orders_rolling_mean_2_sq',
  'homepage_featured_rolling_sum_3',
  'num_orders_rolling_mean_21',
  'num_orders_rolling_mean_5_x_meal_orders_mean',
  'meal_orders_mean_x_emailer_for_promotion',
  'checkout_price_sq',
  'meal_orders_mean_sq',
  'num_orders_rolling_mean_28',
  'meal_orders_std',
  'num_orders_rolling_mean_2_x_checkout_price',
  'meal_orders_mean_x_discount_pct',
  'category_Beverages',
  'num_orders_rolling_mean_5_x_discount_pct_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_price_diff_x_emailer_for_promotion',
  'emailer_for_promotion_rolling_sum_3',
  'base_price',
  'num_orders_rolling_std_14',
  'num_orders_rolling_mean_14_x_weekofyear',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_price_diff',
  'num_orders_rolling_mean_14_x_meal_orders_mean',
  'price_diff',
  'meal_orders_mean_x_checkout_price',
  'meal_orders_mean_x_weekofyear',
  'num_orders_rolling_mean_5_x_center_orders_mean',
  'discount_pct_x_emailer_for_promotion',
  'meal_orders_mean',
  'checkout_price_x_homepage_featured',
  'num_orders_rolling_mean_14_x_discount_pct',
  'checkout_price_x_price_diff',
  'center_orders_mean_x_discount_pct',
  'num_orders_rolling_mean_10',
  'checkout_price_x_weekofyear',
  'num_orders_rolling_mean_14_x_center_orders_mean',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_discount_pct',
  'weekofyear_sq',
  'num_orders_rolling_mean_2_x_discount_pct_x_emailer_for_promotion',
  'mean_orders_by_month',
  'center_orders_std',
  'num_orders_rolling_std_21',
  'num_orders_rolling_mean_14_x_checkout_price',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_month_cos',
  'num_orders_rolling_mean_14_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_weekofyear',
  'num_orders_rolling_mean_3',
  'num_orders_rolling_std_2',
  'num_orders_rolling_mean_14_x_price_diff',
  'discount_pct_x_weekofyear',
  'num_orders_rolling_mean_5_x_checkout_price',
  'num_orders_rolling_mean_2_x_discount_pct_x_homepage_featured',
  'center_orders_mean_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_emailer_for_promotion',
  'center_orders_mean_sq',
  'weekofyear_sin',
  'num_orders_rolling_mean_5',
  'weekofyear_cos',
  'center_orders_mean_x_homepage_featured',
  'num_orders_rolling_mean_5_x_discount_pct_x_homepage_featured',
  'num_orders_rolling_mean_5_x_discount_pct',
  'price_diff_sq',
  'num_orders_rolling_mean_5_x_emailer_for_promotion',
  'checkout_price_x_emailer_for_promotion',
  'num_orders_rolling_std_28',
  'num_orders_rolling_std_6',
  'weekofyear_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_month_sin',
  'center_type_TYPE_B',
  'num_orders_rolling_std_12',
  'cuisine_Italian',
  'cuisine_Continental',
  'num_orders_rolling_std_3',
  'num_orders_rolling_mean_5_x_price_diff_x_emailer_for_promotion',
  'num_orders_rolling_mean_5_x_homepage_featured',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_weekofyear_sin',
  'category_Extras',
  'month_sin_x_discount_pct_x_emailer_for_promotion',
  'num_orders_rolling_std_7',
  'num_orders_rolling_mean_5_x_discount_pct_x_price_diff',
  'discount_pct_x_homepage_featured',
  'num_orders_rolling_mean_5_x_weekofyear',
  'category_Rice Bowl',
  'discount_pct_sq',
  'num_orders_rolling_mean_8',
  'num_orders_rolling_std_8',
  'cuisine_Thai',
  'price_diff_x_discount_pct',
  'discount_pct_x_price_diff_x_emailer_for_promotion',
  'month_sin',
  'checkout_price_x_discount_pct',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_homepage_featured',
  'num_orders_rolling_mean_2_x_num_orders_rolling_mean_5_x_weekofyear_cos',
  'center_type_TYPE_C',
  'meal_orders_mean_x_price_diff',
  'num_orders_rolling_mean_2_x_discount_pct',
  'discount',
  'num_orders_rolling_mean_2_x_price_diff',
  'num_orders_rolling_mean_14',
  'discount_pct_x_emailer_for_promotion_x_homepage_featured',
  'category_Salad',
  'num_orders_rolling_mean_12',
  'weekofyear_x_homepage_featured',
  'center_orders_mean_x_price_diff',
  'month_cos',
  'center_orders_mean_x_checkout_price',
  'discount_pct',
  'num_orders_rolling_std_10',
  'num_orders_rolling_mean_14_x_homepage_featured',
  'num_orders_rolling_mean_2_x_weekofyear_sin_x_price_diff',
  'rolling_mean_2_x_home',
  'num_orders_rolling_std_5',
  'num_orders_rolling_mean_5_x_month_sin_x_price_diff',
  'price_diff_x_weekofyear',
  'num_orders_rolling_mean_5_x_price_diff_x_homepage_featured',
  'emailer_for_promotion',
  'num_orders_rolling_mean_5_x_weekofyear_cos_x_discount_pct',
  'num_orders_rolling_mean_5_x_month_sin_x_emailer_for_promotion',
  'is_outlier_weekofyear',
  'weekofyear_sin_x_emailer_for_promotion_x_homepage_featured',
  'num_orders_rolling_mean_2_x_month_cos_x_discount_pct',
  'center_orders_mean_x_weekofyear',
  'price_diff_x_emailer',
  'num_orders_rolling_mean_5_x_month_cos_x_emailer_for_promotion',
  'num_orders_rolling_mean_5_x_emailer_for_promotion_x_homepage_featured',
  'weekofyear_sin_x_discount_pct_x_price_diff',
  'num_orders_rolling_mean_5_x_month_sin_x_discount_pct',
  'num_orders_rolling_mean_5_x_weekofyear_sin_x_price_diff',
  'num_orders_rolling_mean_2_x_month_sin_x_price_diff',
  'weekofyear_cos_x_discount_pct_x_emailer_for_promotion',
  'num_orders_rolling_mean_5_sq',
  'num_orders_rolling_mean_2_x_month_sin_x_discount_pct',
  'num_orders_rolling_mean_7',
  'num_orders_rolling_mean_2_x_weekofyear_sin_x_discount_pct',
  'num_orders_rolling_std_4',
  'num_orders_rolling_mean_5_x_month_cos_x_price_diff',
  'num_orders_rolling_mean_6',
  'num_orders_rolling_mean_2_x_weekofyear_cos_x_price_diff',
  'category_Sandwich',
  'num_orders_rolling_mean_2_x_emailer_for_promotion_x_homepage_featured',
  'num_orders_rolling_mean_2_x_weekofyear_cos_x_discount_pct',
  'num_orders_rolling_mean_2_x_month_cos_x_price_diff',
  'num_orders_rolling_mean_5_x_weekofyear_sin_x_emailer_for_promotion',
  'num_orders_rolling_mean_5_x_price_diff',
  'weekofyear_sin_x_price_diff_x_emailer_for_promotion',
  'month_sin_x_discount_pct_x_price_diff',
  'num_orders_rolling_mean_2_x_price_diff_x_homepage_featured',
  'category_Other Snacks',
  'num_orders_rolling_mean_5_x_month_cos_x_discount_pct',
  'num_orders_rolling_mean_2_x_homepage_featured',
  'num_orders_rolling_mean_5_x_weekofyear_cos_x_price_diff',
  'num_orders_rolling_mean_5_x_weekofyear_sin_x_discount_pct',
  'num_orders_rolling_mean_5_x_month_cos_x_homepage_featured',
  'num_orders_rolling_mean_2_x_month_sin_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_month_cos_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_discount_pct_x_price_diff',
  'price_diff_x_home',
  'num_orders_rolling_mean_14_sq',
  'weekofyear_sin_x_discount_pct_x_emailer_for_promotion',
  'center_orders_mean',
  'num_orders_rolling_mean_2_x_month_cos_x_homepage_featured',
  'month_cos_x_discount_pct_x_price_diff',
  'weekofyear_cos_x_discount_pct_x_homepage_featured',
  'weekofyear_sin_x_discount_pct_x_homepage_featured',
  'cuisine_Indian',
  'center_type_TYPE_A',
  'rolling_mean_2_x_emailer',
  'category_Starters',
  'center_orders_mean_cube',
  'meal_orders_mean_cube',
  'weekofyear_cos_x_discount_pct_x_price_diff',
  'discount_pct_x_price_diff_x_homepage_featured',
  'num_orders_rolling_mean_5_x_month_sin_x_homepage_featured',
  'weekofyear_cos_x_price_diff_x_emailer_for_promotion',
  'weekofyear_sin_x_price_diff_x_homepage_featured',
  'month_cos_x_price_diff_x_emailer_for_promotion',
  'num_orders_rolling_mean_5_x_weekofyear_cos_x_homepage_featured',
  'num_orders_rolling_mean_5_x_weekofyear_cos_x_emailer_for_promotion',
  'num_orders_rolling_mean_2_x_weekofyear_sin_x_homepage_featured',
  'num_orders_rolling_mean_2_x_weekofyear_cos_x_emailer_for_promotion',
  'num_orders_rolling_mean_5_x_weekofyear_sin_x_homepage_featured',
  'month_cos_x_discount_pct_x_homepage_featured',
  'month_sin_x_price_diff_x_emailer_for_promotion',
  'weekofyear_cos_x_emailer_for_promotion_x_homepage_featured',
  'month_sin_x_price_diff_x_homepage_featured',
  'weekofyear_cos_x_price_diff_x_homepage_featured',
  'month_cos_x_price_diff_x_homepage_featured',
  'category_Seafood',
  'category_Pizza',
  'category_Desert',
  'num_orders_rolling_mean_2_x_emailer_for_promotion',
  'category_Fish',
  'month_cos_x_emailer_for_promotion_x_homepage_featured',
  'price_diff_x_homepage_featured',
  'month_sin_x_emailer_for_promotion_x_homepage_featured',
  'is_outlier_month',
  'category_Soup',
  'emailer_for_promotion_sq',
  'category_Pasta',
  'homepage_featured',
  'homepage_featured_sq',
  'emailer_for_promotion_x_homepage_featured',
  'category_Biryani',
]
# FEATURES = [f for f in FEATURES if f not in features_to_remove]
logging.info(f"Removed manually identified correlated features. {len(FEATURES)} features remain.")

# --- Train/validation split ---
max_week = train_df["week"].max()
valid_df = train_df[train_df["week"] > max_week - VALIDATION_WEEKS].copy()
train_split_df = train_df[train_df["week"] <= max_week - VALIDATION_WEEKS].copy()

logging.info(f"Train split shape: {train_split_df.shape}, Validation shape: {valid_df.shape}")

# --- RMSLE Metric ---
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0) # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False # lower is better

# --- Model Training Function ---
def get_lgbm(params=None):
    """Initializes LGBMRegressor with default or provided params."""
    default_params = {
        'objective': 'regression_l1', # MAE objective often works well for RMSLE
        'metric': 'None', # Use custom metric
        'boosting_type': 'gbdt',
        'n_estimators': 2000, # Increase estimators, use early stopping
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    if params:
        default_params.update(params)
        # Ensure metric is None if custom metric is used during fit
        if 'eval_metric' in params and params['eval_metric'] == lgb_rmsle:
             default_params['metric'] = 'None'
    return LGBMRegressor(**default_params)

# --- Optuna Hyperparameter Tuning ---
logging.info("Starting Optuna hyperparameter tuning...")

# Use Optuna's SQLite storage for persistence (no joblib)
try:
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Loaded existing Optuna study from {OPTUNA_DB}")
except Exception:
    study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Created new Optuna study at {OPTUNA_DB}")

def objective(trial):
    """Optuna objective function."""
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
    }
    # Add fixed params
    params.update({
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric':'None', # Crucial when using feval
    })

    model = LGBMRegressor(**params)
    model.fit(
        train_split_df[FEATURES], train_split_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        eval_metric=lgb_rmsle, # Use custom RMSLE metric
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmsle'), # Pruning based on validation RMSLE
                   lgb.early_stopping(100, verbose=False)] # Early stopping
    )
    preds = model.predict(valid_df[FEATURES])
    score = rmsle(valid_df[TARGET], preds)
    return score

# Run Optuna optimization
study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=1800) # Add a timeout (e.g., 30 minutes)

# No need to save with joblib, study is persisted in SQLite
logging.info(f"Optuna study saved to {OPTUNA_DB}")

best_params = study.best_params
logging.info(f"Best Optuna params: {best_params}")
logging.info(f"Best validation RMSLE: {study.best_value:.5f}")







# --- Final Model Training ---
logging.info("Training final model on full training data with best params...")
# Merge best params with fixed params for the final model
final_params = {
    'objective': 'regression_l1',
    'boosting_type': 'gbdt',
    'n_estimators': 3000, # Increase slightly for final training
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'metric': 'None'
}
final_params.update(best_params) # Best params from Optuna override defaults


# --- Feature Removal Experiment ---
def feature_removal_experiment(train_df, valid_df, FEATURES, features_to_test, TARGET, model_params, n_trials=1):
    """
    For each feature in features_to_test, remove it from FEATURES, retrain, and report validation RMSLE.
    Returns a DataFrame with feature, RMSLE, and delta vs. baseline.
    """
    results = []
    # Baseline
    model = LGBMRegressor(**model_params)
    model.fit(train_df[FEATURES], train_df[TARGET])
    preds = model.predict(valid_df[FEATURES])
    baseline_rmsle = rmsle(valid_df[TARGET], preds)
    results.append({'feature': 'BASELINE', 'rmsle': baseline_rmsle, 'delta_vs_baseline': 0.0})
    # Test each feature
    for feat in tqdm(features_to_test, desc='Feature Removal Experiment'):
        if feat not in FEATURES:
            continue
        test_features = [f for f in FEATURES if f != feat]
        model = LGBMRegressor(**model_params)
        model.fit(train_df[test_features], train_df[TARGET])
        preds = model.predict(valid_df[test_features])
        rmsle_score = rmsle(valid_df[TARGET], preds)
        results.append({'feature': feat, 'rmsle': rmsle_score, 'delta_vs_baseline': rmsle_score - baseline_rmsle})
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmsle')
    results_df.to_csv('feature_removal_experiment.csv', index=False)
    print('Feature removal experiment results saved to feature_removal_experiment.csv')
    return results_df

# --- Run Feature Removal Experiment ---
features_to_test = features_to_remove
model_params = final_params.copy()
model_params['n_estimators'] = 500 # Use fewer estimators for speed
feature_removal_experiment(train_split_df, valid_df, FEATURES, features_to_test, TARGET, model_params)



# --- Cross-Validation Feature Removal Experiment ---
from sklearn.model_selection import KFold

def crossval_feature_removal_experiment(df, FEATURES, features_to_test, TARGET, model_params, n_splits=5):
    """
    For each feature in features_to_test, remove it from FEATURES, run KFold CV, and report mean/STD RMSLE.
    Returns a DataFrame with feature, mean RMSLE, std RMSLE, and delta vs. baseline.
    """
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    # Baseline
    baseline_scores = []
    for train_idx, valid_idx in tqdm(list(kf.split(df)), desc='CV Baseline'):
        train_fold, valid_fold = df.iloc[train_idx], df.iloc[valid_idx]
        model = LGBMRegressor(**model_params)
        model.fit(train_fold[FEATURES], train_fold[TARGET])
        preds = model.predict(valid_fold[FEATURES])
        baseline_scores.append(rmsle(valid_fold[TARGET], preds))
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)
    results.append({'feature': 'BASELINE', 'mean_rmsle': baseline_mean, 'std_rmsle': baseline_std, 'delta_vs_baseline': 0.0})
    # Test each feature
    for feat in tqdm(features_to_test, desc='CV Feature Removal'):
        if feat not in FEATURES:
            continue
        test_features = [f for f in FEATURES if f != feat]
        scores = []
        for train_idx, valid_idx in kf.split(df):
            train_fold, valid_fold = df.iloc[train_idx], df.iloc[valid_idx]
            model = LGBMRegressor(**model_params)
            model.fit(train_fold[test_features], train_fold[TARGET])
            preds = model.predict(valid_fold[test_features])
            scores.append(rmsle(valid_fold[TARGET], preds))
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results.append({'feature': feat, 'mean_rmsle': mean_score, 'std_rmsle': std_score, 'delta_vs_baseline': mean_score - baseline_mean})
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_rmsle')
    results_df.to_csv('crossval_feature_removal_experiment.csv', index=False)
    print('Cross-validation feature removal experiment results saved to crossval_feature_removal_experiment.csv')
    return results_df

# --- Run Cross-Validation Feature Removal Experiment ---
features_to_test = features_to_remove
model_params = get_lgbm().get_params()
model_params['n_estimators'] = 300 # Use fewer estimators for speed
crossval_feature_removal_experiment(train_split_df, FEATURES, features_to_test, TARGET, model_params, n_splits=5)



final_model = LGBMRegressor(**final_params)

# Train on the entire training dataset (train_split_df + valid_df)
# No early stopping here, train for the specified number of estimators
final_model.fit(train_df[FEATURES], train_df[TARGET], eval_metric=lgb_rmsle)

# --- Recursive Prediction ---
logging.info("Starting recursive prediction on the test set...")
# Prepare the combined data history (training data + test structure)
# We need the structure of test_df but will fill num_orders recursively
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

test_weeks = sorted(test_df['week'].unique())

for week_num in test_weeks:
    logging.info(f"Predicting for week {week_num}...")
    # Identify rows for the current week to predict
    current_week_mask = history_df['week'] == week_num

    # Re-apply feature engineering for the current state
    history_df, _, _ = apply_feature_engineering(
        history_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means
    )

    current_features = history_df.loc[current_week_mask, FEATURES]

    # Handle potential missing columns in test data after alignment (should not happen with proper alignment, but defensive)
    missing_cols = [col for col in FEATURES if col not in current_features.columns]
    if missing_cols:
        logging.warning(f"Missing columns during prediction for week {week_num}: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            current_features[col] = 0
    current_features = current_features[FEATURES] # Ensure correct order

    # Predict for the current week
    current_preds = final_model.predict(current_features)
    current_preds = np.clip(current_preds, 0, None).round().astype(float) # Use float for potential later calculations

    # Update the 'num_orders' in history_df for the current week with predictions
    # This ensures the next iteration uses the predicted values to calculate lags/rolling features
    history_df.loc[current_week_mask, 'num_orders'] = current_preds

logging.info("Recursive prediction finished.")

# Extract final predictions for the original test set IDs
final_predictions_df = history_df.loc[history_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int) # Final conversion to int
final_predictions_df['id'] = final_predictions_df['id'].astype(int)

# --- Create Submission File ---
submission_path = f"{SUBMISSION_FILE_PREFIX}_optuna.csv"
final_predictions_df.to_csv(submission_path, index=False)
logging.info(f"Submission file saved to {submission_path}")

# --- SHAP Analysis ---
logging.info("Calculating SHAP values...")
try:
    # Sample data for SHAP to keep computation reasonable
    if len(train_df) > N_SHAP_SAMPLES:
        shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
    else:
        shap_sample = train_df.copy()

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample[FEATURES])

    # Save SHAP values and importance
    shap_values_df = pd.DataFrame(shap_values, columns=FEATURES)
    shap_values_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_values.csv", index=False)

    shap_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_feature_importances.csv", index=False)

    # Generate SHAP plots
    logging.info("Generating SHAP plots...")
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, shap_sample[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_summary.png")
    plt.close()

    # Importance Bar Plot (Top 20)
    plt.figure(figsize=(10, 8))
    shap_importance_df.head(20).plot(kind='barh', x='feature', y='mean_abs_shap', legend=False, figsize=(10, 8))
    plt.gca().invert_yaxis() # Display most important at the top
    plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
    plt.title('Top 20 SHAP Feature Importances (Recursive Optuna Model)')
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_top20_importance.png")
    plt.close()

    logging.info("SHAP analysis saved.")

except Exception as e:
    logging.error(f"Error during SHAP analysis: {e}")

def prune_features_by_shap_and_corr(train_df, FEATURES, shap_importance_path, corr_threshold=0.95):
    """
    Prune features by correlation only:
    - For highly correlated pairs, keep only the one with higher SHAP importance.
    - Do NOT prune based on SHAP top N or minimum importance.
    """
    import pandas as pd
    # Load SHAP importances
    shap_df = pd.read_csv(shap_importance_path)
    shap_df = shap_df.set_index('feature')
    # Only consider features present in both FEATURES and SHAP importances
    keep_features = [f for f in FEATURES if f in shap_df.index]
    # Remove highly correlated features (keep higher SHAP)
    corr = train_df[keep_features].corr().abs()
    to_remove = set()
    for i, f1 in enumerate(keep_features):
        for f2 in keep_features[i+1:]:
            if corr.loc[f1, f2] > corr_threshold:
                # Remove the one with lower SHAP
                if shap_df.loc[f1, 'mean_abs_shap'] >= shap_df.loc[f2, 'mean_abs_shap']:
                    to_remove.add(f2)
                else:
                    to_remove.add(f1)
    pruned_features = [f for f in keep_features if f not in to_remove]
    logging.info(f"Pruned features from {len(FEATURES)} to {len(pruned_features)} using only correlation.")
    return FEATURES # Uncomment the following line to run the pruning function
    # return pruned_features

# --- Prune features before final model training ---
shap_importance_path = f"{SHAP_FILE_PREFIX}_optuna_feature_importances.csv"
FEATURES = prune_features_by_shap_and_corr(train_df, FEATURES, shap_importance_path, corr_threshold=0.95)
logging.info(f"Final pruned feature set (correlation only): {FEATURES}")
logging.info(f"Pruned features: {[f for f in FEATURES if f not in train_df.columns]}")

# --- Plotting Example: Actual vs Predicted for Validation Set ---
logging.info("Generating validation plot...")
try:
    valid_preds = final_model.predict(valid_df[FEATURES])
    valid_preds = np.clip(valid_preds, 0, None)

    plt.figure(figsize=(15, 6))
    plt.scatter(valid_df[TARGET], valid_preds, alpha=0.5, s=10)
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2, label='Ideal')
    plt.xlabel("Actual Orders (Validation Set)")
    plt.ylabel("Predicted Orders (Validation Set)")
    plt.title(f"Actual vs. Predicted Orders (Validation Set) - RMSLE: {rmsle(valid_df[TARGET], valid_preds):.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_actual_vs_predicted.png")
    plt.close()
    logging.info("Validation plot saved.")

except Exception as e:
    logging.error(f"Error during plotting: {e}")

logging.info("Script finished.")

# --- Recursive Stacking/Ensembling ---
def recursive_predict(model, train_df, test_df, FEATURES):
    """Run recursive prediction for a single model."""
    history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    test_weeks = sorted(test_df['week'].unique())
    for week_num in test_weeks:
        current_week_mask = history_df['week'] == week_num
        history_df = apply_feature_engineering(history_df, is_train=False)
        current_features = history_df.loc[current_week_mask, FEATURES]
        missing_cols = [col for col in FEATURES if col not in current_features.columns]
        if missing_cols:
            for col in missing_cols:
                current_features[col] = 0
        current_features = current_features[FEATURES]
        current_preds = model.predict(current_features)
        current_preds = np.clip(current_preds, 0, None).round().astype(float)
        history_df.loc[current_week_mask, 'num_orders'] = current_preds
    final_predictions = history_df.loc[history_df['id'].isin(test_df['id']), ['id', 'num_orders']].copy()
    final_predictions['num_orders'] = final_predictions['num_orders'].round().astype(int)
    final_predictions['id'] = final_predictions['id'].astype(int)
    return final_predictions.set_index('id')['num_orders']


def recursive_ensemble(train_df, test_df, FEATURES, n_models=5):
    """Train n_models with different seeds, run recursive prediction, and average results."""
    preds_list = []
    for i in range(n_models):
        logging.info(f"Training ensemble model {i+1}/{n_models}...")
        params = final_params.copy()
        params['seed'] = SEED + i
        model = LGBMRegressor(**params)
        model.fit(train_df[FEATURES], train_df[TARGET], eval_metric=lgb_rmsle)
        preds = recursive_predict(model, train_df, test_df, FEATURES)
        preds_list.append(preds)
    # Average predictions
    ensemble_preds = np.mean(preds_list, axis=0).round().astype(int)
    return ensemble_preds

# --- Recursive Ensemble Prediction ---
logging.info("Running recursive ensemble prediction...")
ensemble_preds = recursive_ensemble(train_df, test_df, FEATURES, n_models=5)
final_predictions_df['num_orders_ensemble'] = ensemble_preds
submission_path_ensemble = f"{SUBMISSION_FILE_PREFIX}_optuna_ensemble.csv"
final_predictions_df[['id', 'num_orders_ensemble']].rename(columns={'num_orders_ensemble': 'num_orders'}).to_csv(submission_path_ensemble, index=False)
logging.info(f"Ensemble submission file saved to {submission_path_ensemble}")

import os
import random
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

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
# SEED = 42
SEED = random.randint(0, 1000) # Random seed for remote execution

# Feature engineering window configurations - optimized to reduce redundancy while keeping predictive power
LAG_WEEKS = [1, 2, 4, 8] # Optimized lags based on SHAP importance
ROLLING_WINDOWS = [2, 5, 14] # Optimized windows for rolling features (short, medium, long-term)
PROMOTION_COLS = ["emailer_for_promotion", "homepage_featured"] # Promotional feature columns
PROMOTION_WINDOWS = [3, 8, 14] # Optimized windows for promotional features
ROLLING_SUM_WINDOWS = [3, 8, 16] # Optimized windows for binary features
COMBINED_PROMO_WINDOWS = [3, 14] # Reduced windows for combined promotional effects
EWM_ALPHAS = [0.5] # Single alpha value for exponentially weighted means
POLYNOMIAL_ROLLING_WINDOWS = [2, 5, 14] # Matching main rolling windows for polynomial features
POLYNOMIAL_LAG_WINDOWS = [1, 2, 4] # Focus on most important lags for polynomial features
CUBIC_LAG_WINDOWS = [1, 2] # Only apply cubic to the most important lags
# Model configuration
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 75 # Number of Optuna trials
OPTUNA_STUDY_NAME = "woot"
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "woot_submission"
SHAP_FILE_PREFIX = "shap_woot"
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
logging.info("Creating features...")
GROUP_COLS = ["center_id", "meal_id"]

def create_lag_rolling_features(df, target_col='num_orders', lag_weeks=LAG_WEEKS, rolling_windows=ROLLING_WINDOWS):
    """Creates lag and rolling window features for a given target column."""
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Lags
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)

    # Rolling features (use shift(1) to avoid data leakage)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)

    return df_out

def create_other_features(df):
    """Creates features not directly dependent on recursive prediction."""
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Price features
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan) # Avoid division by zero
    df_out["price_diff"] = group["checkout_price"].diff()    # Rolling sums for promo/featured (use shift(1))
    for col in PROMOTION_COLS:
        shifted = group[col].shift(1)
        for window in ROLLING_SUM_WINDOWS:
            df_out[f"{col}_rolling_sum_{window}"] = shifted.rolling(window, min_periods=1).sum().reset_index(drop=True)

    # Time features
    df_out["weekofyear"] = df_out["week"] % 52

    return df_out

def create_group_aggregates(df):
    df_out = df.copy()
    # Center-level aggregates
    df_out['center_orders_mean'] = df_out.groupby('center_id')['num_orders'].transform('mean')
    df_out['center_orders_median'] = df_out.groupby('center_id')['num_orders'].transform('median')
    df_out['center_orders_std'] = df_out.groupby('center_id')['num_orders'].transform('std')
    # Meal-level aggregates
    df_out['meal_orders_mean'] = df_out.groupby('meal_id')['num_orders'].transform('mean')
    df_out['meal_orders_median'] = df_out.groupby('meal_id')['num_orders'].transform('median')
    df_out['meal_orders_std'] = df_out.groupby('meal_id')['num_orders'].transform('std')
    # Category-level aggregates (if available)
    if 'category' in df_out.columns:
        df_out['category_orders_mean'] = df_out.groupby('category')['num_orders'].transform('mean')
        df_out['category_orders_median'] = df_out.groupby('category')['num_orders'].transform('median')
        df_out['category_orders_std'] = df_out.groupby('category')['num_orders'].transform('std')
    
    # High-value cross aggregates (based on SHAP importance from both test.py and newtest.py)
    df_out['center_meal_orders_mean_prod'] = df_out['center_orders_mean'] * df_out['meal_orders_mean']
    df_out['center_meal_orders_median_prod'] = df_out['center_orders_median'] * df_out['meal_orders_median']
    df_out['center_meal_orders_mean_div'] = df_out['center_orders_mean'] / df_out['meal_orders_mean'].replace(0, 1)
    
    # Additional high-value cross aggregates based on SHAP analysis
    df_out['center_meal_orders_std_prod'] = df_out['center_orders_std'] * df_out['meal_orders_std']
    
    # Weighted center-meal features (high SHAP importance in test.py)
    # These capture the proportion of orders for this meal relative to all meals at the center
    center_total_orders = df_out.groupby('center_id')['num_orders'].transform('sum')
    meal_total_orders = df_out.groupby('meal_id')['num_orders'].transform('sum')
    df_out['center_meal_ratio'] = df_out['meal_orders_mean'] / (df_out['center_orders_mean'].replace(0, 1))
    df_out['center_meal_weighted'] = df_out['center_meal_orders_mean_prod'] / (center_total_orders + meal_total_orders).replace(0, 1)
    
    return df_out

def cyclical_encode(df, col, max_val):
    df_out = df.copy()
    df_out[f'{col}_sin'] = np.sin(2 * np.pi * df_out[col] / max_val)
    df_out[f'{col}_cos'] = np.cos(2 * np.pi * df_out[col] / max_val)
    return df_out

def create_advanced_interactions(df):
    df_out = df.copy()
    # Interactions with rolling_mean_2 (a highly important feature from SHAP analysis)
    if 'num_orders_rolling_mean_2' in df_out.columns:
        df_out['rolling_mean_2_x_discount_pct'] = df_out['num_orders_rolling_mean_2'] * df_out.get('discount_pct', 0)
        df_out['rolling_mean_2_x_price_diff'] = df_out['num_orders_rolling_mean_2'] * df_out.get('price_diff', 0)
        df_out['rolling_mean_2_x_weekofyear'] = df_out['num_orders_rolling_mean_2'] * df_out.get('weekofyear', 0)
        # Polynomial features
        df_out['rolling_mean_2_sq'] = df_out['num_orders_rolling_mean_2'] ** 2
        df_out['rolling_mean_2_sqrt'] = np.sqrt(df_out['num_orders_rolling_mean_2'].clip(0))
        # Add cubic for the most important feature
        df_out['rolling_mean_2_cubic'] = df_out['num_orders_rolling_mean_2'] ** 3
      # Extending polynomial features for rolling statistics (important in SHAP)
    for col in [f'num_orders_rolling_mean_{w}' for w in POLYNOMIAL_ROLLING_WINDOWS if f'num_orders_rolling_mean_{w}' in df_out.columns]:
        df_out[f'{col}_sq'] = df_out[col] ** 2
        df_out[f'{col}_sqrt'] = np.sqrt(df_out[col].clip(0))
    
    # Add polynomial features for important numeric columns
    for col in ['checkout_price', 'base_price', 'discount', 'discount_pct', 'price_diff',
                'center_orders_mean', 'meal_orders_mean', 'center_meal_orders_mean_prod']:
        if col in df_out.columns:
            df_out[f'{col}_sq'] = df_out[col] ** 2
            # Add cubic for the most important features
            if col in ['center_orders_mean', 'meal_orders_mean', 'center_meal_orders_mean_prod']:
                df_out[f'{col}_cubic'] = df_out[col] ** 3
      # Add polynomial features for lag variables (highly important in SHAP)
    for lag in POLYNOMIAL_LAG_WINDOWS:
        lag_col = f'num_orders_lag_{lag}'
        if lag_col in df_out.columns:
            df_out[f'{lag_col}_sq'] = df_out[lag_col] ** 2
            # Add cubic for the most important lags
            if lag in CUBIC_LAG_WINDOWS:
                df_out[f'{lag_col}_cubic'] = df_out[lag_col] ** 3
    
    # Ratio features for price-related columns
    if all(c in df_out.columns for c in ['checkout_price', 'base_price']):
        df_out['price_ratio'] = df_out['checkout_price'] / df_out['base_price'].replace(0, np.nan)
        
    # Price discount polynomial interactions (from test.py SHAP)
    if all(c in df_out.columns for c in ['base_price', 'discount_pct']):
        df_out['base_price_poly2_discount_pct'] = df_out['base_price'] * (df_out['discount_pct'] ** 2)
        
    # Promotional polynomial interactions
    if all(c in df_out.columns for c in ['homepage_featured', 'discount']):
        df_out['homepage_featured_poly2_discount'] = df_out['homepage_featured'] * (df_out['discount'] ** 2)
    
    # Interactions with seasonality if present
    if all(c in df_out.columns for c in ['mean_orders_by_weekofyear', 'checkout_price']):
        df_out['seasonal_week_x_price'] = df_out['mean_orders_by_weekofyear'] * df_out['checkout_price']
        
    # Center-meal interactions (top performers in test.py)
    if all(c in df_out.columns for c in ['center_orders_mean', 'meal_orders_mean']):
        df_out['center_orders_mean_poly2_meal_orders_mean'] = df_out['center_orders_mean'] * (df_out['meal_orders_mean'] ** 2)
        # Add reverse polynomial interaction
        df_out['meal_orders_mean_poly2_center_orders_mean'] = df_out['meal_orders_mean'] * (df_out['center_orders_mean'] ** 2)
    
    # Add centered quadratic features for dates to capture non-linear seasonality
    if 'weekofyear' in df_out.columns:
        # Center around middle of year (26) before squaring to reduce correlation
        df_out['weekofyear_centered_sq'] = ((df_out['weekofyear'] - 26) ** 2) / 676  # Normalize by 26^2
    if 'month' in df_out.columns:
        # Center around middle of year (6.5) before squaring
        df_out['month_centered_sq'] = ((df_out['month'] - 6.5) ** 2) / 42.25  # Normalize by 6.5^2
        
    return df_out

def create_interaction_features(df):
    """Creates interaction features."""
    df_out = df.copy()
    interactions = {
        # Price and promotional interactions
        "price_diff_x_emailer": ("price_diff", "emailer_for_promotion"),
        "lag1_x_emailer": ("num_orders_lag_1", "emailer_for_promotion"),
        "price_diff_x_home": ("price_diff", "homepage_featured"),
        "lag1_x_home": ("num_orders_lag_1", "homepage_featured"),
        
        # Rolling mean interactions with promotions
        "rolling_mean_2_x_emailer": ("num_orders_rolling_mean_2", "emailer_for_promotion"),
        "rolling_mean_2_x_home": ("num_orders_rolling_mean_2", "homepage_featured"),
        
        # Additional rolling mean windows with promotions (adjusted for new window sizes)
        "rolling_mean_5_x_emailer": ("num_orders_rolling_mean_5", "emailer_for_promotion"),
        "rolling_mean_14_x_emailer": ("num_orders_rolling_mean_14", "emailer_for_promotion"),
        
        # Meal/center aggregates interactions
        "meal_mean_x_discount": ("meal_orders_mean", "discount"),
        "center_mean_x_discount": ("center_orders_mean", "discount"),
        "discount_pct_x_center_mean": ("discount_pct", "center_orders_mean"),
        "base_price_x_homepage": ("base_price", "homepage_featured"),
        
        # Lag and rolling interactions (adjusted for new window sizes)
        "lag1_x_rolling_mean_2": ("num_orders_lag_1", "num_orders_rolling_mean_2"),
        "lag1_x_rolling_mean_5": ("num_orders_lag_1", "num_orders_rolling_mean_5"),
        "rolling_mean_2_x_rolling_mean_5": ("num_orders_rolling_mean_2", "num_orders_rolling_mean_5"),
        "lag1_x_lag2": ("num_orders_lag_1", "num_orders_lag_2"),
        
        # Additional high-value lag interactions (adjusted for new window sizes)
        "lag1_x_lag4": ("num_orders_lag_1", "num_orders_lag_4"),
        "lag2_x_lag4": ("num_orders_lag_2", "num_orders_lag_4"),
        "lag2_x_rolling_mean_2": ("num_orders_lag_2", "num_orders_rolling_mean_2"),
        "lag2_x_rolling_mean_5": ("num_orders_lag_2", "num_orders_rolling_mean_5"),
        
        # Seasonality interactions
        "lag1_x_weekofyear_sin": ("num_orders_lag_1", "weekofyear_sin"),
        "lag1_x_month_sin": ("num_orders_lag_1", "month_sin"),
        "mean_by_weekofyear_x_checkout": ("mean_orders_by_weekofyear", "checkout_price"),
        
        # Additional seasonality interactions from test.py SHAP
        "mean_by_month_x_checkout": ("mean_orders_by_month", "checkout_price"),
        "mean_by_month_x_discount": ("mean_orders_by_month", "discount"),
        
        # Price based interactions (from test.py SHAP)
        "checkout_x_homepage_x_discount": ("checkout_price", "homepage_featured", "discount"),
        "base_price_x_discount_pct": ("base_price", "discount_pct"),
        
        # Additional triple interactions (adjusted for new window sizes)
        "checkout_x_homepage_x_month_mean": ("checkout_price", "homepage_featured", "mean_orders_by_month"),
        "center_mean_x_meal_mean_x_discount": ("center_orders_mean", "meal_orders_mean", "discount"),
        "rolling_mean_2_x_rolling_mean_5_x_emailer": ("num_orders_rolling_mean_2", "num_orders_rolling_mean_5", "emailer_for_promotion"),
    }
    for name, features in interactions.items():
        # Handle both two-feature and three-feature interactions
        if len(features) == 2:
            feat1, feat2 = features
            if feat1 in df_out.columns and feat2 in df_out.columns:
                df_out[name] = df_out[feat1] * df_out[feat2]
            else:
                logging.warning(f"Skipping interaction '{name}' because base feature(s) missing.")
                df_out[name] = 0  # Add column with default value if base features missing
        elif len(features) == 3:
            # Triple interaction
            feat1, feat2, feat3 = features
            if feat1 in df_out.columns and feat2 in df_out.columns and feat3 in df_out.columns:
                df_out[name] = df_out[feat1] * df_out[feat2] * df_out[feat3]
            else:
                logging.warning(f"Skipping triple interaction '{name}' because base feature(s) missing.")
                df_out[name] = 0
        
    return df_out

def create_temporal_features(df):
    """Creates additional temporal features like month."""
    df_out = df.copy()
    # Month feature (derived from week)
    df_out["month"] = ((df_out["week"] - 1) // 4) % 12 + 1
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)
    return df_out

def add_seasonality_features(df, weekofyear_means=None, month_means=None, is_train=True):
    """
    Adds seasonality features based on weekly and monthly patterns.
    These capture the average order patterns for different weeks/months of the year.
    """
    df_out = df.copy()
    if is_train:
        # Calculate these means from training data
        weekofyear_means = df_out.groupby('weekofyear')['num_orders'].mean()
        month_means = df_out.groupby('month')['num_orders'].mean()
    else:
        # Use pre-calculated means from training
        if weekofyear_means is None or month_means is None:
            raise ValueError("When is_train=False, weekofyear_means and month_means must be provided")
    
    # Map the means back to the dataframe
    df_out['mean_orders_by_weekofyear'] = df_out['weekofyear'].map(weekofyear_means)
    df_out['mean_orders_by_month'] = df_out['month'].map(month_means)
    return df_out

def add_binary_rolling_means(df, binary_cols=PROMOTION_COLS, windows=PROMOTION_WINDOWS):
    """
    Creates rolling mean features for binary columns like promotions or homepage features.
    This helps capture the effect of recent marketing activities over different time spans.
    Based on SHAP analysis, these features capture important promotional patterns.
    """
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)
    for col in binary_cols:
        if col in df_out.columns:
            # Shift by 1 to avoid data leakage
            shifted = group[col].shift(1)
            
            # Add rolling means
            for window in windows:
                df_out[f"{col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
            
            # Add expanded rolling windows for the most important binary features
            if col in PROMOTION_COLS:
                # Add cumulative sum of promotions in last N periods
                for window in ROLLING_SUM_WINDOWS:
                    df_out[f"{col}_rolling_sum_{window}"] = shifted.rolling(window, min_periods=1).sum().reset_index(drop=True)
                  # Add exponentially weighted means to capture decaying impact of promotions
                for alpha in EWM_ALPHAS:
                    df_out[f"{col}_ewm_alpha_{alpha}"] = shifted.ewm(alpha=alpha, min_periods=1).mean().reset_index(drop=True)
                
                # Add interaction between promotional features if both exist
                if "emailer_for_promotion" in df_out.columns and "homepage_featured" in df_out.columns:
                    df_out["emailer_homepage_combined"] = df_out["emailer_for_promotion"] * df_out["homepage_featured"]
                    # Add rolling mean of the combined promotional effect
                    combined_shifted = group["emailer_homepage_combined"].shift(1)
                    for window in COMBINED_PROMO_WINDOWS:
                        df_out[f"emailer_homepage_combined_rolling_mean_{window}"] = combined_shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
    
    return df_out

def apply_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None):
    """Applies all feature engineering steps consistently for both train and test."""
    df_out = df.copy()
    df_out = create_temporal_features(df_out)
    if is_train or 'num_orders' in df_out.columns:
        df_out = create_lag_rolling_features(df_out)
    df_out = create_other_features(df_out)
    df_out = create_group_aggregates(df_out)
    df_out = cyclical_encode(df_out, 'weekofyear', 52)
    df_out = add_seasonality_features(df_out, weekofyear_means=weekofyear_means, month_means=month_means, is_train=is_train)
    df_out = add_binary_rolling_means(df_out)
    df_out = create_interaction_features(df_out)
    df_out = create_advanced_interactions(df_out)
    return df_out

# --- One-hot encoding and feature engineering for train/test ---
logging.info("Applying one-hot encoding and feature engineering...")
df_full = pd.concat([df, test], ignore_index=True)
df_full = create_other_features(df_full)
df_full = create_temporal_features(df_full)
cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in df_full.columns]
if cat_cols:
    df_full = pd.get_dummies(df_full, columns=cat_cols, dummy_na=False) # Avoid NaN columns from dummies

train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()

# First apply feature engineering to train to get seasonality means
train_df = apply_feature_engineering(train_df, is_train=True)

# Extract seasonality means for use in test data
weekofyear_means = train_df.groupby('weekofyear')['num_orders'].mean()
month_means = train_df.groupby('month')['num_orders'].mean()

# Now apply feature engineering to test with the seasonality means
test_df = apply_feature_engineering(test_df, is_train=False, 
                                   weekofyear_means=weekofyear_means, 
                                   month_means=month_means)

# Drop rows in train_df where target is NA (if any, though unlikely from problem desc)
train_df = train_df.dropna(subset=['num_orders']).reset_index(drop=True)


# --- Define Features and Target ---
TARGET = "num_orders"
FEATURES = [
    # Base features
    "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", "weekofyear",
    
    # Temporal and cyclical encoding
    "weekofyear_sin", "weekofyear_cos", "month_sin", "month_cos",
    
    # Seasonality features
    "mean_orders_by_weekofyear", "mean_orders_by_month",
    
    # Centered quadratic temporal features
    "weekofyear_centered_sq", "month_centered_sq",
    
    # Price-derived features
    "price_ratio"
]

# Add lag features
FEATURES += [f"{TARGET}_lag_{lag}" for lag in LAG_WEEKS if f"{TARGET}_lag_{lag}" in train_df.columns]

# Add rolling statistics
FEATURES += [f"{TARGET}_rolling_mean_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_mean_{w}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_std_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_std_{w}" in train_df.columns]

# Add binary rolling means with expanded windows
for col in PROMOTION_COLS:
    FEATURES += [f"{col}_rolling_mean_{w}" for w in PROMOTION_WINDOWS if f"{col}_rolling_mean_{w}" in train_df.columns]

# Add promo rolling sums
FEATURES += [f"{col}_rolling_sum_{w}" for col in PROMOTION_COLS for w in ROLLING_SUM_WINDOWS if f"{col}_rolling_sum_{w}" in train_df.columns]

# Add EWM features
for col in PROMOTION_COLS:
    FEATURES += [f"{col}_ewm_alpha_{alpha}" for alpha in EWM_ALPHAS if f"{col}_ewm_alpha_{alpha}" in train_df.columns]

# Add combined promotional features
if "emailer_homepage_combined" in train_df.columns:
    FEATURES.append("emailer_homepage_combined")
    FEATURES += [f"emailer_homepage_combined_rolling_mean_{w}" for w in COMBINED_PROMO_WINDOWS if f"emailer_homepage_combined_rolling_mean_{w}" in train_df.columns]

# Add all interaction features
FEATURES += [col for col in train_df.columns if (
    col.startswith("price_diff_x_") or 
    col.startswith("rolling_mean_") and "_x_" in col or
    col.startswith("lag1_x_") or
    col.startswith("lag2_x_") or
    col.startswith("meal_mean_x_") or
    col.startswith("center_mean_x_") or
    col.startswith("seasonal_") or
    "mean_by_month_x_" in col
) and col in train_df.columns]  # Extra check to ensure column exists

# Add all polynomial features
FEATURES += [col for col in train_df.columns if (
    col.endswith("_sq") or 
    col.endswith("_sqrt") or
    col.endswith("_cubic") or
    "poly" in col or
    "center_orders_mean_poly2" in col or
    "meal_orders_mean_poly2" in col or
    "base_price_poly2" in col or
    "homepage_featured_poly2" in col
) and col in train_df.columns]  # Extra check to ensure column exists

# Add group-level aggregates
FEATURES += [col for col in train_df.columns if any(col.startswith(prefix) for prefix in ["center_orders_", "meal_orders_", "category_orders_"])]

# Add cross-aggregate features (center-meal interactions)
FEATURES += [col for col in train_df.columns if col.startswith("center_meal_orders_") or col.startswith("center_meal_")]

# Add one-hot columns if present
FEATURES += [col for col in train_df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]

# Filter out any features that don't exist or are target/id
FEATURES = [f for f in FEATURES if f in train_df.columns and f != TARGET and f != 'id']

# Remove duplicates while preserving order
FEATURES = list(dict.fromkeys(FEATURES))

logging.info(f"Using {len(FEATURES)} features: {FEATURES}")


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
        'min_data_in_leaf': 20,
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

# --- Custom Early Stopping Callback with Overfitting Detection ---
def early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=15, verbose=False):
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
        
        for item in env.evaluation_result_list:
            if 'train' in item[0]:
                train_loss = item[1]
            elif 'valid' in item[0]:
                valid_loss = item[1]
                
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
            logging.info(f"[Iter {env.iteration}] train: {train_loss:.5f}, valid: {valid_loss:.5f}, overfit_count: {overfit_count[0]}")
            
        # Stop if overfitting detected
        if overfit_count[0] >= overfit_rounds:
            if verbose:
                logging.info(f"Stopping early due to overfitting at iteration {env.iteration}")
            raise lgb.callback.EarlyStopException(env.iteration, best_score[0])
            
        # Standard early stopping
        if env.iteration - best_iter[0] >= stopping_rounds:
            if verbose:
                logging.info(f"Stopping early due to no improvement at iteration {env.iteration}")
            raise lgb.callback.EarlyStopException(env.iteration, best_score[0])
            
    return _callback

# --- Optuna Hyperparameter Tuning ---
logging.info("Starting Optuna hyperparameter tuning...")

# Use Optuna's SQLite storage for persistence (no joblib)
try:
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    logging.info(f"Loaded existing Optuna study from {OPTUNA_DB}")
except Exception:
    study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB, sampler=optuna.samplers.TPESampler(constant_liar=True))
    logging.info(f"Created new Optuna study at {OPTUNA_DB}")

def objective(trial):
    """Optuna objective function."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 512),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1000.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1000.0, log=True),
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
        eval_set=[
            (train_split_df[FEATURES], train_split_df[TARGET]),  # Add training set for overfitting detection
            (valid_df[FEATURES], valid_df[TARGET])
        ],
        eval_metric=lgb_rmsle, # Use custom RMSLE metric
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, 'rmsle'),  # Pruning based on validation RMSLE
            early_stopping_with_overfit(stopping_rounds=200, overfit_rounds=15, verbose=False)  # Use custom early stopping with overfitting detection
        ]
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
logging.info("Training final model on full training data with best params and early stopping...")
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

# Ensure key regularization parameters from test.py are present to prevent overfitting
if 'lambda_l1' not in final_params or final_params['lambda_l1'] < 1.0:
    final_params['lambda_l1'] = max(final_params.get('lambda_l1', 0), 5.0)
if 'lambda_l2' not in final_params or final_params['lambda_l2'] < 1.0:
    final_params['lambda_l2'] = max(final_params.get('lambda_l2', 0), 5.0)
if 'min_data_in_leaf' not in final_params or final_params['min_data_in_leaf'] < 20:
    final_params['min_data_in_leaf'] = max(final_params.get('min_data_in_leaf', 0), 100)

final_model = LGBMRegressor(**final_params)

# Create a slightly more robust validation split for final training
# Use the most recent weeks as validation to better represent the test set
train_weeks = sorted(train_df['week'].unique())
n_val_weeks = max(3, int(0.15 * len(train_weeks)))  # Use at least 3 weeks or 15% of weeks for validation
val_weeks = train_weeks[-n_val_weeks:]
train_weeks = train_weeks[:-n_val_weeks]

train_final = train_df[train_df['week'].isin(train_weeks)].copy()
val_final = train_df[train_df['week'].isin(val_weeks)].copy()

logging.info(f"Final training: {len(train_final)} samples, validation: {len(val_final)} samples")
logging.info(f"Using validation weeks: {val_weeks}")

# Train with stronger early stopping to detect and prevent overfitting
final_model.fit(
    train_final[FEATURES], train_final[TARGET], 
    eval_set=[
        (train_final[FEATURES], train_final[TARGET]),  # Include training set to monitor for overfitting
        (val_final[FEATURES], val_final[TARGET])
    ],
    eval_metric=lgb_rmsle,
    callbacks=[early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=15, verbose=True)]
)

# Evaluate final model on validation set
val_preds = final_model.predict(val_final[FEATURES])
val_rmsle = rmsle(val_final[TARGET], val_preds)
logging.info(f"Final model validation RMSLE: {val_rmsle:.5f}")

# --- Enhanced Error Correction Utilities ---
def compute_error_metrics(y_true, y_pred):
    """Calculate comprehensive error statistics from validation data."""
    errors = y_true - y_pred
    relative_errors = errors / np.maximum(y_true, 1)  # Avoid division by zero
    
    # Calculate standard error metrics
    error_stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'mae': np.mean(np.abs(errors)),
        'mape': np.mean(np.abs(relative_errors)) * 100,
        'rmse': np.sqrt(np.mean(np.square(errors))),
        'rmsle_original': rmsle(y_true, y_pred)
    }
    
    # Calculate quantile errors for more robust correction
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        error_stats[f'q{int(q*100)}_error'] = np.quantile(errors, q)
    
    # Calculate segment-specific errors for more targeted correction
    segments = {}
    
    # Low/medium/high value segments
    low_mask = y_true <= np.quantile(y_true, 0.33)
    med_mask = (y_true > np.quantile(y_true, 0.33)) & (y_true <= np.quantile(y_true, 0.67))
    high_mask = y_true > np.quantile(y_true, 0.67)
    
    segments['low_value'] = errors[low_mask].mean() if np.any(low_mask) else 0
    segments['med_value'] = errors[med_mask].mean() if np.any(med_mask) else 0
    segments['high_value'] = errors[high_mask].mean() if np.any(high_mask) else 0
    
    error_stats['segments'] = segments
    
    return error_stats

def find_optimal_offset(y_true, y_pred):
    """Find the best additive correction (offset) that minimizes RMSLE."""
    offsets = np.linspace(-5, 5, 101)  # Test offsets from -5 to +5
    best_offset = 0
    best_rmsle = rmsle(y_true, y_pred)
    
    for offset in offsets:
        adjusted_preds = y_pred + offset
        adjusted_preds = np.maximum(adjusted_preds, 0)  # Ensure non-negative
        current_rmsle = rmsle(y_true, adjusted_preds)
        
        if current_rmsle < best_rmsle:
            best_rmsle = current_rmsle
            best_offset = offset
    
    improvement = rmsle(y_true, y_pred) - best_rmsle
    return best_offset, improvement

def find_optimal_scaling(y_true, y_pred):
    """Find the best multiplicative correction (scaling) that minimizes RMSLE."""
    scales = np.linspace(0.85, 1.15, 61)  # More focused range
    best_scale = 1.0
    best_rmsle = rmsle(y_true, y_pred)
    
    for scale in scales:
        adjusted_preds = y_pred * scale
        current_rmsle = rmsle(y_true, adjusted_preds)
        
        if current_rmsle < best_rmsle:
            best_rmsle = current_rmsle
            best_scale = scale
    
    improvement = rmsle(y_true, y_pred) - best_rmsle
    return best_scale, improvement

def get_seasonality_factors(df, week_num, weekofyear_means, month_means):
    """Get seasonality-based correction factors for the given prediction week."""
    week_data = df[df['week'] == week_num]
    if len(week_data) == 0:
        return None
    
    weekofyear = week_data['weekofyear'].iloc[0]
    month = week_data['month'].iloc[0]
    
    # Get statistics from similar weeks/months in training data
    similar_week_mask = df['weekofyear'] == weekofyear
    similar_month_mask = df['month'] == month
    
    # Create a collection of seasonality statistics
    seasonality = {}
    
    # Global means from pre-computed aggregates
    if weekofyear in weekofyear_means.index:
        seasonality['weekofyear_mean'] = weekofyear_means[weekofyear]
    else:
        seasonality['weekofyear_mean'] = df[~df[TARGET].isna()][TARGET].mean()
        
    if month in month_means.index:
        seasonality['month_mean'] = month_means[month]
    else:
        seasonality['month_mean'] = df[~df[TARGET].isna()][TARGET].mean()
    
    # Calculate actual means from similar periods in training data
    if np.sum(similar_week_mask & ~df[TARGET].isna()) > 10:  # Only if we have enough samples
        seasonality['similar_weekofyear_actual'] = df.loc[similar_week_mask & ~df[TARGET].isna(), TARGET].mean()
    
    if np.sum(similar_month_mask & ~df[TARGET].isna()) > 10:
        seasonality['similar_month_actual'] = df.loc[similar_month_mask & ~df[TARGET].isna(), TARGET].mean()
    
    return seasonality

def validate_error_correction(y_true, y_pred, correction_fn, *args):
    """Test if a correction function actually improves predictions."""
    base_rmsle = rmsle(y_true, y_pred)
    corrected_preds = correction_fn(y_pred, *args)
    corrected_rmsle = rmsle(y_true, corrected_preds)
    
    # Only return the correction if it actually helps
    if corrected_rmsle < base_rmsle:
        improvement = base_rmsle - corrected_rmsle
        return corrected_preds, improvement
    else:
        return y_pred, 0.0

# --- End Enhanced Error Correction Utilities ---

# --- Recursive Prediction with Error Correction ---
logging.info("Starting recursive prediction with error correction on the test set...")

# Prepare the combined data history (training data + test structure)
# We need the structure of test_df but will fill num_orders recursively
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

# Extract seasonality means from train_df for use in recursive prediction
weekofyear_means = train_df.groupby('weekofyear')['num_orders'].mean()
month_means = train_df.groupby('month')['num_orders'].mean()

# Create a validation dataset from the most recent training data for error correction calibration
train_weeks = sorted(train_df['week'].unique())
validation_weeks = train_weeks[-VALIDATION_WEEKS:]
validation_df = train_df[train_df['week'].isin(validation_weeks)].copy()
    
# Generate validation predictions to measure error patterns
validation_features = validation_df[FEATURES].copy()
missing_val_cols = [col for col in FEATURES if col not in validation_features.columns]
for col in missing_val_cols:
    validation_features[col] = 0
validation_preds = final_model.predict(validation_features)
validation_true = validation_df[TARGET].values

# Calculate detailed error statistics
logging.info("Calculating error correction parameters from validation data...")
error_metrics = compute_error_metrics(validation_true, validation_preds)
logging.info(f"Validation error statistics: {error_metrics}")

# Find optimal offset and scaling corrections
offset, offset_improvement = find_optimal_offset(validation_true, validation_preds)
scaling, scaling_improvement = find_optimal_scaling(validation_true, validation_preds)

logging.info(f"Optimal offset: {offset:.4f} (RMSLE improvement: {offset_improvement:.4f})")
logging.info(f"Optimal scaling factor: {scaling:.4f} (RMSLE improvement: {scaling_improvement:.4f})")

# Determine which correction gives better improvement
if offset_improvement > scaling_improvement:
    logging.info(f"Using offset-based correction (better improvement)")
    primary_correction = "offset"
    correction_value = offset
    correction_improvement = offset_improvement
else:
    logging.info(f"Using scaling-based correction (better improvement)")
    primary_correction = "scaling"
    correction_value = scaling
    correction_improvement = scaling_improvement

# Create segment-based error correction for different value ranges
# This helps correct errors differently for small, medium, and large prediction values
validation_df['pred'] = validation_preds
validation_df['error'] = validation_true - validation_preds
validation_df['relative_error'] = validation_df['error'] / np.maximum(validation_true, 1)

# Segment by prediction value
low_threshold = np.percentile(validation_preds, 33)
high_threshold = np.percentile(validation_preds, 67)

validation_df['value_segment'] = 'medium'
validation_df.loc[validation_df['pred'] <= low_threshold, 'value_segment'] = 'low'
validation_df.loc[validation_df['pred'] > high_threshold, 'value_segment'] = 'high'

# Calculate segment-specific corrections
segment_corrections = {}
segment_scaling = {}

for segment in ['low', 'medium', 'high']:
    segment_mask = validation_df['value_segment'] == segment
    if sum(segment_mask) >= 20:  # Only if we have enough samples
        segment_true = validation_true[segment_mask]
        segment_pred = validation_preds[segment_mask]
        
        # Find segment-specific offset
        segment_offset, segment_offset_improvement = find_optimal_offset(segment_true, segment_pred)
        
        # Find segment-specific scaling
        segment_scale, segment_scale_improvement = find_optimal_scaling(segment_true, segment_pred)
        
        # Save the better of the two
        if segment_offset_improvement > segment_scale_improvement:
            segment_corrections[segment] = {'method': 'offset', 'value': segment_offset, 'improvement': segment_offset_improvement}
        else:
            segment_corrections[segment] = {'method': 'scaling', 'value': segment_scale, 'improvement': segment_scale_improvement}
            
        logging.info(f"Segment '{segment}' correction: {segment_corrections[segment]['method']} = {segment_corrections[segment]['value']:.4f} " +
                    f"(improvement: {segment_corrections[segment]['improvement']:.4f})")

test_weeks = sorted(test_df['week'].unique())
weeks_predicted = 0
weekly_correction_stats = []

for week_num in test_weeks:
    logging.info(f"Predicting for week {week_num}...")
    # Identify rows for the current week to predict
    current_week_mask = history_df['week'] == week_num

    # Re-apply feature engineering for the current state with seasonality means
    history_df = apply_feature_engineering(history_df, is_train=False, 
                                          weekofyear_means=weekofyear_means, 
                                          month_means=month_means)

    current_features = history_df.loc[current_week_mask, FEATURES]
    # Handle potential missing columns in test data after alignment
    missing_cols = [col for col in FEATURES if col not in current_features.columns]
    if missing_cols:
        logging.warning(f"Missing columns during prediction for week {week_num}: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            current_features[col] = 0
    current_features = current_features[FEATURES] # Ensure correct order

    # Get base predictions from the model
    base_preds = final_model.predict(current_features)
    base_preds = np.clip(base_preds, 0, None) # Keep as float for corrections
    
    # Get seasonality correction factors for the current week
    seasonality = get_seasonality_factors(history_df, week_num, weekofyear_means, month_means)
    
    # Apply corrections with adaptive approach
    # Make a copy for corrections
    corrected_preds = base_preds.copy()
    
    # 1. First apply primary correction approach (best one from validation)
    if primary_correction == "offset":
        # Add offset with temporal decay
        correction_factor = correction_value * (0.85 ** weeks_predicted)  # exponential decay
        corrected_preds += correction_factor
    else:  # scaling
        # Scale with minimal decay (scaling is more robust over time)
        correction_factor = 1.0 + (correction_value - 1.0) * (0.95 ** weeks_predicted)
        corrected_preds *= correction_factor
    
    # 2. Apply segment-specific corrections for better accuracy
    # Split predictions into segments
    segment_masks = {
        'low': base_preds <= low_threshold,
        'medium': (base_preds > low_threshold) & (base_preds <= high_threshold),
        'high': base_preds > high_threshold
    }
    
    # Apply segment corrections where helpful
    for segment, mask in segment_masks.items():
        if segment in segment_corrections and sum(mask) > 0:
            correction = segment_corrections[segment]
            segment_improvement = correction['improvement']
            
            # Only apply if improvement is significant
            if segment_improvement > 0.01:
                # Create segment-specific correction with decay
                decay = 0.9 ** weeks_predicted  # Decay segment corrections more slowly
                
                if correction['method'] == 'offset':
                    # Apply offset to just this segment with decay
                    segment_value = correction['value'] * decay
                    corrected_preds[mask] += segment_value
                else:  # scaling
                    # Apply scaling to just this segment with decay
                    scale_factor = 1.0 + (correction['value'] - 1.0) * decay
                    corrected_preds[mask] *= scale_factor
    
    # 3. Blend with seasonality for later weeks when we have less confidence
    if seasonality and weeks_predicted > 0:
        # As we get further into the future, seasonality becomes more important
        seasonality_weight = min(0.4, 0.08 * weeks_predicted)  # gradually increase up to 40%
        
        # Create a seasonality baseline - prefer similar_week/month actual if available
        if 'similar_weekofyear_actual' in seasonality and 'similar_month_actual' in seasonality:
            # We have actual values from similar seasonal periods
            seasonality_baseline = seasonality['similar_weekofyear_actual'] * 0.6 + seasonality['similar_month_actual'] * 0.4
        else:
            # Fall back to global seasonal means
            seasonality_baseline = seasonality['weekofyear_mean'] * 0.6 + seasonality['month_mean'] * 0.4
        
        # Apply the seasonal blend
        corrected_preds = (1 - seasonality_weight) * corrected_preds + seasonality_weight * seasonality_baseline
    
    # 4. Final validation - ensure we're not making predictions worse
    # Define a simple modification function we can validate
    def simple_mix(preds):
        # This creates a safer middle ground between base and corrected predictions
        return 0.7 * corrected_preds + 0.3 * base_preds
    
    # Apply only if corrections are reasonable
    max_correction_pct = np.max(np.abs((corrected_preds - base_preds) / np.maximum(base_preds, 1)))
    if max_correction_pct > 0.5:  # If any correction is more than 50%
        logging.warning(f"Large corrections detected (max {max_correction_pct:.1f}%). Testing a milder correction blend.")
        corrected_preds = simple_mix(base_preds)
    
    # Ensure predictions are non-negative and in the proper format
    corrected_preds = np.clip(corrected_preds, 0, None).round().astype(float)
    
    # Log correction statistics
    avg_base = np.mean(base_preds)
    avg_corrected = np.mean(corrected_preds)
    correction_diff = avg_corrected - avg_base
    correction_pct = 100 * correction_diff / max(1, avg_base)
    logging.info(f"Week {week_num} corrections: Base avg={avg_base:.2f}, " +
                 f"Corrected avg={avg_corrected:.2f}, Diff={correction_diff:.2f} ({correction_pct:.1f}%)")
    
    # Save weekly correction stats
    weekly_correction_stats.append({
        'week': week_num,
        'base_mean': avg_base,
        'corrected_mean': avg_corrected,
        'abs_correction': np.mean(np.abs(corrected_preds - base_preds)),
        'rel_correction_pct': correction_pct,
        'n_samples': len(base_preds)
    })
    
    # Update the 'num_orders' in history_df for the current week with corrected predictions
    # This ensures the next iteration uses the corrected values for lags/rolling features
    history_df.loc[current_week_mask, 'num_orders'] = corrected_preds
    weeks_predicted += 1

logging.info("Recursive prediction finished.")

# Extract final predictions for the original test set IDs
final_predictions_df = history_df.loc[history_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int) # Final conversion to int
final_predictions_df['id'] = final_predictions_df['id'].astype(int)

# Save weekly correction statistics
if weekly_correction_stats:
    correction_stats_df = pd.DataFrame(weekly_correction_stats)
    correction_stats_path = f"{SUBMISSION_FILE_PREFIX}_weekly_correction_stats.csv"
    correction_stats_df.to_csv(correction_stats_path, index=False)
    logging.info(f"Weekly correction statistics saved to {correction_stats_path}")
    
    # Create a plot of weekly corrections
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(correction_stats_df['week'], correction_stats_df['base_mean'], 'b-', label='Base Predictions')
        plt.plot(correction_stats_df['week'], correction_stats_df['corrected_mean'], 'r-', label='Corrected Predictions')
        plt.xlabel('Week')
        plt.ylabel('Mean Predicted Orders')
        plt.title('Effect of Error Correction by Week')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{SUBMISSION_FILE_PREFIX}_weekly_corrections.png")
        plt.close()
        
        # Plot relative correction percentage
        plt.figure(figsize=(10, 6))
        plt.bar(correction_stats_df['week'], correction_stats_df['rel_correction_pct'])
        plt.xlabel('Week')
        plt.ylabel('Correction (%)')
        plt.title('Relative Correction Percentage by Week')
        plt.grid(True)
        plt.savefig(f"{SUBMISSION_FILE_PREFIX}_correction_percentage.png")
        plt.close()
        
        logging.info("Weekly correction plots saved.")
    except Exception as e:
        logging.error(f"Error creating correction plots: {e}")

# --- Create Submission File ---
submission_path = f"{SUBMISSION_FILE_PREFIX}_ec_optuna.csv"
final_predictions_df.to_csv(submission_path, index=False)
logging.info(f"Error-corrected submission saved to {submission_path}")

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

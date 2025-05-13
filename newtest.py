import os
import random
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import logging
import lightgbm as lgb  # Added for early stopping callback
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_log_error
import datetime
from sklearn.linear_model import LinearRegression

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
# SEED = 42
SEED = random.randint(0, 1000) # Random seed for reproducibility
LAG_WEEKS = [1, 2, 3, 5, 10] # Lags based on num_orders
ROLLING_WINDOWS = [2, 3, 5, 10, 14, 21] # Added 14 and 21
# Other features (not directly dependent on recursive prediction)
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 20 # Number of Optuna trials
OPTUNA_STUDY_NAME = "experiment_4"
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "experiment_4_submission"
SHAP_FILE_PREFIX = "shap_experiment_4"
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
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, 1e-12) # Avoid division by zero
    df_out["price_diff"] = group["checkout_price"].diff()

    # Rolling sums for promo/featured (use shift(1))
    for col in OTHER_ROLLING_SUM_COLS:
        shifted = group[col].shift(1)
        df_out[f"{col}_rolling_sum_{OTHER_ROLLING_SUM_WINDOW}"] = shifted.rolling(OTHER_ROLLING_SUM_WINDOW, min_periods=1).sum().reset_index(drop=True)

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
    
    # High-value cross aggregates (based on SHAP importance from test.py)
    df_out['center_meal_orders_mean_prod'] = df_out['center_orders_mean'] * df_out['meal_orders_mean']
    df_out['center_meal_orders_median_prod'] = df_out['center_orders_median'] * df_out['meal_orders_median']
    df_out['center_meal_orders_mean_div'] = df_out['center_orders_mean'] / df_out['meal_orders_mean'].replace(0, 1e-12)
    
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
        # df_out['rolling_mean_2_sq'] = df_out['num_orders_rolling_mean_2'] ** 2
        # df_out['rolling_mean_2_sqrt'] = np.sqrt(df_out['num_orders_rolling_mean_2'].clip(0))
    
    # Extending polynomial features for rolling statistics (important in SHAP)
    # for col in [f'num_orders_rolling_mean_{w}' for w in [3, 5, 14, 21] if f'num_orders_rolling_mean_{w}' in df_out.columns]:
        # df_out[f'{col}_sq'] = df_out[col] ** 2
        # df_out[f'{col}_sqrt'] = np.sqrt(df_out[col].clip(0))
    
    # Add polynomial features for important numeric columns
    # for col in ['checkout_price', 'base_price', 'discount', 'discount_pct', 'price_diff', 'center_orders_mean', 'meal_orders_mean']:
    #     if col in df_out.columns:
            # df_out[f'{col}_sq'] = df_out[col] ** 2
    
    # Add polynomial features for lag variables (highly important in SHAP)
    # for lag in [1, 2, 3, 5, 10]:
    #     lag_col = f'num_orders_lag_{lag}'
    #     if lag_col in df_out.columns:
    #         df_out[f'{lag_col}_sq'] = df_out[lag_col] ** 2
    
    # Ratio features for price-related columns
    if all(c in df_out.columns for c in ['checkout_price', 'base_price']):
        df_out['price_ratio'] = df_out['checkout_price'] / df_out['base_price'].replace(0, 1e-12)

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
        
        # Additional rolling mean windows with promotions
        "rolling_mean_3_x_emailer": ("num_orders_rolling_mean_3", "emailer_for_promotion"),
        "rolling_mean_5_x_emailer": ("num_orders_rolling_mean_5", "emailer_for_promotion"),
        
        # Meal/center aggregates interactions
        "meal_mean_x_discount": ("meal_orders_mean", "discount"),
        "center_mean_x_discount": ("center_orders_mean", "discount"),
        "discount_pct_x_center_mean": ("discount_pct", "center_orders_mean"),
        "base_price_x_homepage": ("base_price", "homepage_featured"),
        
        # Lag and rolling interactions (most important according to SHAP)
        "lag1_x_rolling_mean_2": ("num_orders_lag_1", "num_orders_rolling_mean_2"),
        "lag1_x_rolling_mean_3": ("num_orders_lag_1", "num_orders_rolling_mean_3"),
        "rolling_mean_2_x_rolling_mean_3": ("num_orders_rolling_mean_2", "num_orders_rolling_mean_3"),
        "lag1_x_lag2": ("num_orders_lag_1", "num_orders_lag_2"),
        
        # Seasonality interactions
        "lag1_x_weekofyear_sin": ("num_orders_lag_1", "weekofyear_sin"),
        "lag1_x_month_sin": ("num_orders_lag_1", "month_sin"),
        "mean_by_weekofyear_x_checkout": ("mean_orders_by_weekofyear", "checkout_price"),
        
        # Price based interactions (from test.py SHAP)
        "checkout_x_homepage_x_discount": ("checkout_price", "homepage_featured", "discount"),
        "base_price_x_discount_pct": ("base_price", "discount_pct"),
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

def add_binary_rolling_means(df, binary_cols=["emailer_for_promotion", "homepage_featured"], binary_rolling_means_windows=[2, 3, 5, 7, 14, 21]):
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
            for window in binary_rolling_means_windows:
                df_out[f"{col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
            
            # Add expanded rolling windows for the most important binary features
            if col in ["emailer_for_promotion", "homepage_featured"]:
                for window in [8, 13, 20]:  # Additional windows from test.py SHAP
                    df_out[f"{col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
            
                # Add cumulative sum of promotions in last N periods
                for window in [4, 8, 12]:
                    df_out[f"{col}_rolling_sum_{window}"] = shifted.rolling(window, min_periods=1).sum().reset_index(drop=True)
    
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
test_df = apply_feature_engineering(test_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means)

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
for col in ["emailer_for_promotion", "homepage_featured"]:
    FEATURES += [f"{col}_rolling_mean_{w}" for w in [2, 3, 5, 7, 8, 13, 14, 20, 21] if f"{col}_rolling_mean_{w}" in train_df.columns]

# Add promo rolling sums
FEATURES += [f"{col}_rolling_sum_{w}" for col in OTHER_ROLLING_SUM_COLS for w in [3, 4, 8, 12] if f"{col}_rolling_sum_{w}" in train_df.columns]

# Add all interaction features
FEATURES += [col for col in train_df.columns if (
    col.startswith("price_diff_x_") or 
    col.startswith("rolling_mean_") and "_x_" in col or
    col.startswith("lag1_x_") or
    col.startswith("meal_mean_x_") or
    col.startswith("center_mean_x_") or
    col.startswith("seasonal_")
)]

# Add all polynomial features
FEATURES += [col for col in train_df.columns if (
    # col.endswith("_sq") or 
    # col.endswith("_sqrt") or
    "poly" in col or  # include all polynomial features, not just target ones
    "center_orders_mean_poly2" in col or
    "base_price_poly2" in col or
    "homepage_featured_poly2" in col
)]

# Add group-level aggregates
FEATURES += [col for col in train_df.columns if any(col.startswith(prefix) for prefix in ["center_orders_", "meal_orders_", "category_orders_"])]

# Add cross-aggregate features (center-meal interactions)
FEATURES += [col for col in train_df.columns if col.startswith("center_meal_orders_")]

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

def cat_rmsle(predictions, data):
    """RMSLE metric for CatBoost"""
    return 'RMSLE', rmsle(data.get_target(), predictions), False

def analyze_feature_error_correction(ensemble_result, valid_df, features, target, n_bins=10):
    """
    Analyzes how the error correction effectiveness varies across different feature values.
    
    Args:
        ensemble_result: Result dictionary from train_stacking_ensemble
        valid_df: Validation dataframe
        features: List of features to analyze
        target: Target column
        n_bins: Number of bins to divide feature values into
        
    Returns:
        dict: Dictionary of DataFrames with binned error analysis
    """
    logging.info(f"Analyzing error correction effectiveness across {len(features)} features...")
    
    # Extract predictions
    lgb_preds = ensemble_result['predictions']['valid']['lgb']
    cat_preds = ensemble_result['predictions']['valid']['cat']
    avg_preds = ensemble_result['predictions']['valid']['avg']
    meta_preds = ensemble_result['predictions']['valid']['meta']
    
    # Calculate errors
    actual = valid_df[target].values
    lgb_errors = np.abs(actual - lgb_preds)
    cat_errors = np.abs(actual - cat_preds)
    avg_errors = np.abs(actual - avg_preds)
    meta_errors = np.abs(actual - meta_preds)
    
    # Track improvement metrics for each feature
    results = {}
    
    # Analyze top important features
    important_features = features[:20]  # Analyze top 20 features 
    
    for feature in important_features:
        if feature not in valid_df.columns:
            logging.warning(f"Feature {feature} not in validation dataframe")
            continue
        
        feature_values = valid_df[feature].values
        
        if len(np.unique(feature_values)) <= n_bins:
            # For categorical or low-cardinality features, use unique values
            bins = np.unique(feature_values)
            labels = bins
            indices = np.array([np.where(bins == val)[0][0] for val in feature_values])
        else:
            # For continuous features, create bins
            bins = np.linspace(feature_values.min(), feature_values.max(), n_bins+1)
            labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
            indices = np.digitize(feature_values, bins[1:-1])
        
        # Initialize bin stats
        bin_stats = []
        
        # Calculate metrics for each bin
        for i in range(len(np.unique(indices))):
            mask = indices == i
            if np.sum(mask) < 10:  # Skip bins with too few samples
                continue
                
            bin_actual = actual[mask]
            bin_lgb_preds = lgb_preds[mask]
            bin_cat_preds = cat_preds[mask]
            bin_avg_preds = avg_preds[mask]
            bin_meta_preds = meta_preds[mask]
            
            # Calculate errors and improvements
            bin_lgb_rmsle = rmsle(bin_actual, bin_lgb_preds)
            bin_cat_rmsle = rmsle(bin_actual, bin_cat_preds)
            bin_avg_rmsle = rmsle(bin_actual, bin_avg_preds)
            bin_meta_rmsle = rmsle(bin_actual, bin_meta_preds)
            
            best_base_rmsle = min(bin_lgb_rmsle, bin_cat_rmsle)
            meta_improvement = (best_base_rmsle - bin_meta_rmsle) / best_base_rmsle * 100 if best_base_rmsle > 0 else 0
            
            bin_label = labels[i] if i < len(labels) else f"Bin {i}"
            
            bin_stats.append({
                'Bin': bin_label,
                'Count': np.sum(mask),
                'Avg_Feature_Value': np.mean(feature_values[mask]),
                'Avg_Target': np.mean(bin_actual),
                'LGB_RMSLE': bin_lgb_rmsle,
                'CAT_RMSLE': bin_cat_rmsle,
                'AVG_RMSLE': bin_avg_rmsle,
                'META_RMSLE': bin_meta_rmsle,
                'Improvement_Pct': meta_improvement
            })
        
        # Create DataFrame for this feature
        if bin_stats:
            results[feature] = pd.DataFrame(bin_stats)
    
    logging.info(f"Feature-wise error correction analysis completed")
    return results

def ensemble_predict_recursive(ensemble_result, history_df, test_weeks, features, target='num_orders'):
    """
    Makes recursive predictions with the ensemble model.
    
    Args:
        ensemble_result: Result from train_stacking_ensemble
        history_df: DataFrame with historical data
        test_weeks: List of weeks to predict
        features: List of feature columns
        target: Target column name
        
    Returns:
        DataFrame: Predictions for test weeks
    """
    logging.info("Starting recursive ensemble prediction...")
    
    # Extract models
    lgb_models = ensemble_result['models']['lgb']
    cat_models = ensemble_result['models']['cat']
    meta_model = ensemble_result['models']['meta']
    
    # Extract seasonality means from history_df for feature engineering
    weekofyear_means = history_df[history_df[target].notna()].groupby('weekofyear')[target].mean()
    month_means = history_df[history_df[target].notna()].groupby('month')[target].mean()
    
    # Make copy of history_df for predictions
    predict_df = history_df.copy()
    
    # Track timing for each week
    start_time = datetime.datetime.now()
    
    for week_idx, week_num in enumerate(test_weeks):
        week_start_time = datetime.datetime.now()
        logging.info(f"Predicting week {week_num} ({week_idx+1}/{len(test_weeks)})...")
        
        # Identify rows for current week
        current_week_mask = predict_df['week'] == week_num
        
        # Re-apply feature engineering
        predict_df = apply_feature_engineering(predict_df, is_train=False, 
                                              weekofyear_means=weekofyear_means, 
                                              month_means=month_means)
        
        # Get features for current week
        current_features = predict_df.loc[current_week_mask, features].copy()
        
        # Handle missing columns
        missing_cols = [col for col in features if col not in current_features.columns]
        if missing_cols:
            logging.warning(f"Missing {len(missing_cols)} columns in prediction: {missing_cols[:5]}...")
            for col in missing_cols:
                current_features[col] = 0
                
        # Ensure correct column order
        current_features = current_features[features]
        
        # Initialize prediction arrays
        lgb_preds = np.zeros(len(current_features))
        cat_preds = np.zeros(len(current_features))
        
        # Make predictions with each model
        for i, (lgb_model, cat_model) in enumerate(zip(lgb_models, cat_models)):
            lgb_preds += lgb_model.predict(current_features) / len(lgb_models)
            cat_preds += cat_model.predict(current_features) / len(cat_models)
        
        # Create meta-features for meta-model (matching training features)
        meta_features = pd.DataFrame({
            'lgb_pred': lgb_preds,
            'cat_pred': cat_preds,
            # Model agreement features
            'pred_diff': np.abs(lgb_preds - cat_preds),
            'pred_mean': (lgb_preds + cat_preds) / 2,
            'pred_product': lgb_preds * cat_preds,
            # Important original features
            'checkout_price': current_features['checkout_price'].values if 'checkout_price' in current_features else 0,
            'base_price': current_features['base_price'].values if 'base_price' in current_features else 0,
            'discount': current_features['discount'].values if 'discount' in current_features else 0,
            'weekofyear': current_features['weekofyear'].values if 'weekofyear' in current_features else 0,
            'mean_orders_by_weekofyear': current_features['mean_orders_by_weekofyear'].values if 'mean_orders_by_weekofyear' in current_features else 0,
            'center_orders_mean': current_features['center_orders_mean'].values if 'center_orders_mean' in current_features else 0,
            'meal_orders_mean': current_features['meal_orders_mean'].values if 'meal_orders_mean' in current_features else 0
        })
        
        # Make meta-model predictions
        meta_preds = meta_model.predict(meta_features)
        meta_preds = np.clip(meta_preds, 0, None).round().astype(float)
        
        # Update history with predictions
        predict_df.loc[current_week_mask, target] = meta_preds
        
        # Report timing
        week_end_time = datetime.datetime.now()
        week_duration = (week_end_time - week_start_time).total_seconds()
        elapsed_time = (week_end_time - start_time).total_seconds()
        remaining_weeks = len(test_weeks) - (week_idx + 1)
        est_remaining_time = remaining_weeks * (elapsed_time / (week_idx + 1))
        
        logging.info(f"Week {week_num} predicted in {week_duration:.1f}s. Est. remaining time: {est_remaining_time/60:.1f} mins")
    
    logging.info(f"All predictions completed in {(datetime.datetime.now() - start_time).total_seconds()/60:.1f} mins")
    
    # Return predictions for test weeks
    return predict_df[predict_df['week'].isin(test_weeks)]

def plot_ensemble_diagnostics(ensemble_result, train_df, valid_df, target):
    """Generates diagnostic plots for the ensemble model"""
    logging.info("Creating ensemble diagnostic plots...")
    
    try:
        # Create folder for plots
        plots_dir = "ensemble_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prediction vs Actual scatter plot
        plt.figure(figsize=(12, 10))
        
        # Training data
        plt.subplot(2, 2, 1)
        plt.scatter(train_df[target], ensemble_result['predictions']['oof']['lgb'], alpha=0.5, s=5, label='LightGBM')
        plt.scatter(train_df[target], ensemble_result['predictions']['oof']['cat'], alpha=0.5, s=5, label='CatBoost')
        plt.scatter(train_df[target], ensemble_result['predictions']['oof']['meta'], alpha=0.5, s=5, label='Meta-model')
        plt.plot([0, train_df[target].max()], [0, train_df[target].max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted (OOF)')
        plt.title('Training Predictions vs Actual')
        plt.legend()
        
        # Validation data
        plt.subplot(2, 2, 2)
        plt.scatter(valid_df[target], ensemble_result['predictions']['valid']['lgb'], alpha=0.5, s=5, label='LightGBM')
        plt.scatter(valid_df[target], ensemble_result['predictions']['valid']['cat'], alpha=0.5, s=5, label='CatBoost')
        plt.scatter(valid_df[target], ensemble_result['predictions']['valid']['meta'], alpha=0.5, s=5, label='Meta-model')
        plt.plot([0, valid_df[target].max()], [0, valid_df[target].max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted (Validation)')
        plt.title('Validation Predictions vs Actual')
        plt.legend()
        
        # Error distributions
        plt.subplot(2, 2, 3)
        plt.hist(train_df[target] - ensemble_result['predictions']['oof']['lgb'], bins=50, alpha=0.5, label='LightGBM')
        plt.hist(train_df[target] - ensemble_result['predictions']['oof']['cat'], bins=50, alpha=0.5, label='CatBoost')
        plt.hist(train_df[target] - ensemble_result['predictions']['oof']['meta'], bins=50, alpha=0.5, label='Meta-model')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Count')
        plt.title('Training Error Distribution')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.hist(valid_df[target] - ensemble_result['predictions']['valid']['lgb'], bins=50, alpha=0.5, label='LightGBM')
        plt.hist(valid_df[target] - ensemble_result['predictions']['valid']['cat'], bins=50, alpha=0.5, label='CatBoost')
        plt.hist(valid_df[target] - ensemble_result['predictions']['valid']['meta'], bins=50, alpha=0.5, label='Meta-model')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Count')
        plt.title('Validation Error Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/ensemble_prediction_diagnostics.png")
        plt.close()
        
        # Plot model comparison
        models = ['LightGBM', 'CatBoost', 'Meta-model', 'Simple Average']
        oof_scores = [
            ensemble_result['scores']['oof']['lgb'],
            ensemble_result['scores']['oof']['cat'],
            ensemble_result['scores']['oof']['meta'],
            ensemble_result['scores']['oof']['avg']
        ]
        valid_scores = [
            ensemble_result['scores']['valid']['lgb'],
            ensemble_result['scores']['valid']['cat'],
            ensemble_result['scores']['valid']['meta'],
            ensemble_result['scores']['valid']['avg']
        ]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, oof_scores, width, label='OOF RMSLE')
        plt.bar(x + width/2, valid_scores, width, label='Valid RMSLE')
        
        plt.xlabel('Model')
        plt.ylabel('RMSLE')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models)
        plt.legend()
        
        # Add value labels
        for i, v in enumerate(oof_scores):
            plt.text(i - width/2, v + 0.01, f"{v:.4f}", ha='center')
        for i, v in enumerate(valid_scores):
            plt.text(i + width/2, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/ensemble_model_comparison.png")
        plt.close()
        
        logging.info(f"Ensemble diagnostic plots saved to {plots_dir}/")
        
    except Exception as e:
        logging.error(f"Error creating ensemble diagnostic plots: {e}")

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
                train_loss = item[2]
            elif 'valid' in item[0]:
                valid_loss = item[2]
                
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

# --- Advanced Ensemble Architecture ---
def train_stacking_ensemble(train_df, valid_df, features, target, n_folds=5):
    """
    Trains a stacking ensemble with out-of-fold predictions to avoid data leakage.
    Uses both LightGBM and CatBoost as base models and a Ridge meta-model for error correction.
    
    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe
        features: List of feature columns
        target: Target column name
        n_folds: Number of cross-validation folds
        
    Returns:
        dict: Dictionary with trained models and predictions
    """
    logging.info(f"Training {n_folds}-fold stacking ensemble with error correction...")
    
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize arrays for out-of-fold and validation predictions
    oof_lgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    valid_lgb = np.zeros(len(valid_df))
    valid_cat = np.zeros(len(valid_df))
    
    # Store models for later use
    lgb_models = []
    cat_models = []
    
    # Initialize best parameters
    lgb_params = {
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
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
        'metric': 'None',
    }
    
    cat_params = {
        'loss_function': 'MAE',  # MAE works well for RMSLE
        'eval_metric': 'RMSE',   # We'll use our custom metric in eval_set
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False
    }
    
    # Train models with cross-validation to get OOF predictions
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        logging.info(f"Training fold {fold+1}/{n_folds}")
        
        # Split data for this fold
        X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target]
        
        # Train LightGBM
        lgb_model = LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=lgb_rmsle,
            callbacks=[early_stopping_with_overfit(stopping_rounds=200, overfit_rounds=15)]
        )
        lgb_models.append(lgb_model)
        
        # Get OOF and validation predictions
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        valid_lgb += lgb_model.predict(valid_df[features]) / n_folds
        
        # Train CatBoost
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=False
        )
        cat_models.append(cat_model)
        
        # Get OOF and validation predictions
        oof_cat[val_idx] = cat_model.predict(X_val)
        valid_cat += cat_model.predict(valid_df[features]) / n_folds
    
    # Calculate performance of individual models
    lgb_score = rmsle(train_df[target], oof_lgb)
    cat_score = rmsle(train_df[target], oof_cat)
    logging.info(f"LightGBM OOF RMSLE: {lgb_score:.5f}")
    logging.info(f"CatBoost OOF RMSLE: {cat_score:.5f}")
    
    # Create meta-features for error correction model
    # Create meta-features without direct leaks (avoid using true error features)
    meta_features_train = pd.DataFrame({
        'lgb_pred': oof_lgb,
        'cat_pred': oof_cat,
        # Model agreement features
        'pred_diff': np.abs(oof_lgb - oof_cat),
        'pred_mean': (oof_lgb + oof_cat) / 2,
        'pred_product': oof_lgb * oof_cat,
        # Important original features
        'checkout_price': train_df['checkout_price'].values,
        'base_price': train_df['base_price'].values,
        'discount': train_df['discount'].values,
        'weekofyear': train_df['weekofyear'].values,
        'mean_orders_by_weekofyear': train_df['mean_orders_by_weekofyear'].values,
        'center_orders_mean': train_df['center_orders_mean'].values,
        'meal_orders_mean': train_df['meal_orders_mean'].values
    })
    
    # Prepare validation meta-features (no error leaks)
    meta_features_valid = pd.DataFrame({
        'lgb_pred': valid_lgb,
        'cat_pred': valid_cat,
        # Model agreement features
        'pred_diff': np.abs(valid_lgb - valid_cat),
        'pred_mean': (valid_lgb + valid_cat) / 2,
        'pred_product': valid_lgb * valid_cat,
        # Important original features
        'checkout_price': valid_df['checkout_price'].values,
        'base_price': valid_df['base_price'].values,
        'discount': valid_df['discount'].values,
        'weekofyear': valid_df['weekofyear'].values,
        'mean_orders_by_weekofyear': valid_df['mean_orders_by_weekofyear'].values,
        'center_orders_mean': valid_df['center_orders_mean'].values,
        'meal_orders_mean': valid_df['meal_orders_mean'].values
    })
    
    # Retrieve or tune meta-model hyperparameters via Optuna
    try:
        meta_study = optuna.load_study(
            study_name=f"{OPTUNA_STUDY_NAME}_meta", storage=OPTUNA_DB
        )
        logging.info("Loaded existing meta-model Optuna study")
    except Exception:
        meta_study = optuna.create_study(
            direction="minimize",
            study_name=f"{OPTUNA_STUDY_NAME}_meta",
            storage=OPTUNA_DB,
            sampler=optuna.samplers.TPESampler()
        )
        meta_study.optimize(objective_meta, n_trials=max(5, OPTUNA_TRIALS), timeout=600)
        logging.info("Completed meta-model hyperparameter tuning")
    best_meta_params = meta_study.best_params
    # Build final meta-model as LightGBM using tuned params
    lgbm_meta_params = {
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': best_meta_params.get('meta_n_estimators', 2000),
        'learning_rate': best_meta_params.get('meta_learning_rate', 0.05),
        'num_leaves': best_meta_params.get('meta_num_leaves', 31),
        'max_depth': best_meta_params.get('meta_max_depth', -1),
        'feature_fraction': best_meta_params.get('meta_feature_fraction', 1.0),
        'bagging_fraction': best_meta_params.get('meta_bagging_fraction', 1.0),
        'bagging_freq': best_meta_params.get('meta_bagging_freq', 0),
        'lambda_l1': best_meta_params.get('meta_lambda_l1', 0.0),
        'lambda_l2': best_meta_params.get('meta_lambda_l2', 0.0),
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric': 'None'
    }
    meta_model = LGBMRegressor(**lgbm_meta_params)
    # Final meta-model training with early stopping (no pruning callback)
    meta_model.fit(
        meta_features_train,
        train_df[target],
        eval_set=[
            (meta_features_train, train_df[target]),
            (meta_features_valid, valid_df[target])
        ],
        eval_metric=lgb_rmsle,
        callbacks=[
            early_stopping_with_overfit(
                stopping_rounds=100,
                overfit_rounds=15,
                verbose=False
            )
        ]
    )
    
    # Make meta-model predictions
    oof_meta = meta_model.predict(meta_features_train)
    valid_meta = meta_model.predict(meta_features_valid)
    
    # Clip negative predictions
    oof_meta = np.clip(oof_meta, 0, None)
    valid_meta = np.clip(valid_meta, 0, None)
    
    # Calculate meta-model performance
    meta_score = rmsle(train_df[target], oof_meta)
    logging.info(f"Meta-model OOF RMSLE: {meta_score:.5f}")
    
    # Calculate validation performance
    valid_lgb_score = rmsle(valid_df[target], valid_lgb)
    valid_cat_score = rmsle(valid_df[target], valid_cat)
    valid_meta_score = rmsle(valid_df[target], valid_meta)
    logging.info(f"LightGBM validation RMSLE: {valid_lgb_score:.5f}")
    logging.info(f"CatBoost validation RMSLE: {valid_cat_score:.5f}")
    logging.info(f"Meta-model validation RMSLE: {valid_meta_score:.5f}")
    
    # Calculate simple average for comparison
    avg_oof = (oof_lgb + oof_cat) / 2
    avg_valid = (valid_lgb + valid_cat) / 2
    avg_oof_score = rmsle(train_df[target], avg_oof)
    avg_valid_score = rmsle(valid_df[target], avg_valid)
    logging.info(f"Simple average OOF RMSLE: {avg_oof_score:.5f}")
    logging.info(f"Simple average validation RMSLE: {avg_valid_score:.5f}")
    
    # Return all models and predictions
    return {
        'models': {
            'lgb': lgb_models,
            'cat': cat_models,
            'meta': meta_model
        },
        'predictions': {
            'oof': {
                'lgb': oof_lgb,
                'cat': oof_cat,
                'meta': oof_meta,
                'avg': avg_oof
            },
            'valid': {
                'lgb': valid_lgb,
                'cat': valid_cat,
                'meta': valid_meta,
                'avg': avg_valid
            }
        },
        'scores': {
            'oof': {
                'lgb': lgb_score,
                'cat': cat_score,
                'meta': meta_score,
                'avg': avg_oof_score
            },
            'valid': {
                'lgb': valid_lgb_score,
                'cat': valid_cat_score,
                'meta': valid_meta_score,
                'avg': avg_valid_score
            }
        }
    }

def track_learning_curves(lgb_models, cat_models, valid_df, features, target, directory="learning_curves"):
    """
    Tracks and visualizes learning curves for the trained models.
    
    Args:
        lgb_models: List of trained LightGBM models
        cat_models: List of trained CatBoost models
        valid_df: Validation dataframe
        features: Feature columns
        target: Target column
        directory: Directory to save visualizations
    """
    logging.info("Tracking learning curves for ensemble models...")
    os.makedirs(directory, exist_ok=True)
    
    # Extract LightGBM learning curves
    lgb_train_curves = []
    lgb_valid_curves = []
    lgb_best_iters = []
    
    for i, model in enumerate(lgb_models):
        # Get evaluation results
        evals_result = model.evals_result_
        
        if evals_result:
            train_metrics = list(evals_result['training'].values())[0]
            valid_metrics = list(evals_result['valid_1'].values())[0]
            
            lgb_train_curves.append(train_metrics)
            lgb_valid_curves.append(valid_metrics)
            lgb_best_iters.append(model.best_iteration_)
    
    # Extract CatBoost learning curves
    cat_train_curves = []
    cat_valid_curves = []
    cat_best_iters = []
    
    for i, model in enumerate(cat_models):
        # Get training stats if available
        if hasattr(model, 'get_evals_result') and model.get_evals_result():
            metrics = model.get_evals_result()
            if 'learn' in metrics and 'validation' in metrics:
                train_metrics = list(metrics['learn'].values())[0]
                valid_metrics = list(metrics['validation'].values())[0]
                
                cat_train_curves.append(train_metrics)
                cat_valid_curves.append(valid_metrics)
                cat_best_iters.append(model.get_best_iteration())
    
    # Plot LightGBM learning curves
    if lgb_train_curves and lgb_valid_curves:
        plt.figure(figsize=(12, 8))
        
        for i, (train_curve, valid_curve, best_iter) in enumerate(zip(lgb_train_curves, lgb_valid_curves, lgb_best_iters)):
            iterations = range(1, len(train_curve) + 1)
            plt.plot(iterations, train_curve, '-', alpha=0.5, label=f'Train Fold {i+1}')
            plt.plot(iterations, valid_curve, '--', alpha=0.7, label=f'Valid Fold {i+1}')
            plt.axvline(x=best_iter, color=f'C{i}', linestyle=':', alpha=0.7)
            plt.text(best_iter, max(train_curve), f"  Best: {best_iter}", va='top')
        
        plt.xlabel('Iterations')
        plt.ylabel('RMSLE')
        plt.title('LightGBM Learning Curves Across Folds')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{directory}/lgb_learning_curves.png")
        plt.close()
    
    # Plot CatBoost learning curves
    if cat_train_curves and cat_valid_curves:
        plt.figure(figsize=(12, 8))
        
        for i, (train_curve, valid_curve, best_iter) in enumerate(zip(cat_train_curves, cat_valid_curves, cat_best_iters)):
            iterations = range(1, len(train_curve) + 1)
            plt.plot(iterations, train_curve, '-', alpha=0.5, label=f'Train Fold {i+1}')
            plt.plot(iterations, valid_curve, '--', alpha=0.7, label=f'Valid Fold {i+1}')
            plt.axvline(x=best_iter, color=f'C{i}', linestyle=':', alpha=0.7)
            plt.text(best_iter, max(train_curve), f"  Best: {best_iter}", va='top')
        
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('CatBoost Learning Curves Across Folds')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{directory}/cat_learning_curves.png")
        plt.close()
    
    # Compare convergence across models
    if lgb_best_iters and cat_best_iters:
        plt.figure(figsize=(10, 6))
        
        plt.boxplot([lgb_best_iters, cat_best_iters], labels=['LightGBM', 'CatBoost'])
        plt.ylabel('Best Iteration')
        plt.title('Model Convergence Comparison')
        plt.grid(axis='y', alpha=0.3)
        
        # Add individual points
        for i, iters in enumerate([lgb_best_iters, cat_best_iters]):
            x = np.random.normal(i+1, 0.04, size=len(iters))
            plt.plot(x, iters, 'o', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{directory}/convergence_comparison.png")
        plt.close()
    
    logging.info(f"Learning curve visualizations saved to {directory}/")


# --- Analyze Ensemble Stability ---
def analyze_ensemble_stability(ensemble_result, valid_df, target, n_bins=10):
    """
    Analyzes the stability of ensemble predictions compared to individual models.
    
    Args:
        ensemble_result: Result from train_stacking_ensemble
        valid_df: Validation dataframe
        target: Target column
        n_bins: Number of bins for analysis
        
    Returns:
        DataFrame: Stability analysis results
    """
    logging.info("Analyzing ensemble prediction stability...")
    
    # Extract predictions
    lgb_preds = ensemble_result['predictions']['valid']['lgb']
    cat_preds = ensemble_result['predictions']['valid']['cat']
    meta_preds = ensemble_result['predictions']['valid']['meta']
    
    # Calculate model disagreement (variability)
    model_disagreement = np.abs(lgb_preds - cat_preds)
    
    # Calculate prediction errors for each model
    actual = valid_df[target].values
    lgb_errors = np.abs(actual - lgb_preds)
    cat_errors = np.abs(actual - cat_preds)
    meta_errors = np.abs(actual - meta_preds)
    
    # Create bins based on model disagreement
    disagreement_bins = np.linspace(model_disagreement.min(), model_disagreement.max(), n_bins+1)
    bin_indices = np.digitize(model_disagreement, disagreement_bins)
    
    # Initialize results
    stability_results = []
    
    # Analyze each bin
    for i in range(1, n_bins+1):
        mask = bin_indices == i
        if np.sum(mask) < 5:  # Skip bins with too few samples
            continue
            
        bin_disagreement = model_disagreement[mask].mean()
        bin_actual = actual[mask]
        bin_lgb_preds = lgb_preds[mask]
        bin_cat_preds = cat_preds[mask]
        bin_meta_preds = meta_preds[mask]
        
        # Calculate errors
        bin_lgb_errors = lgb_errors[mask]
        bin_cat_errors = cat_errors[mask]
        bin_meta_errors = meta_errors[mask]
        
        # Calculate error variances
        lgb_error_var = np.var(bin_lgb_errors)
        cat_error_var = np.var(bin_cat_errors)
        meta_error_var = np.var(bin_meta_errors)
        
        # Calculate RMSLEs
        bin_lgb_rmsle = rmsle(bin_actual, bin_lgb_preds)
        bin_cat_rmsle = rmsle(bin_actual, bin_cat_preds)
        bin_meta_rmsle = rmsle(bin_actual, bin_meta_preds)
        
        # Calculate best base model error
        best_base_model_rmsle = min(bin_lgb_rmsle, bin_cat_rmsle)
        
        # Calculate improvement ratio
        improvement_pct = (best_base_model_rmsle - bin_meta_rmsle) / best_base_model_rmsle * 100
        
        # Calculate error variance reduction
        best_base_error_var = min(lgb_error_var, cat_error_var)
        var_reduction_pct = (best_base_error_var - meta_error_var) / best_base_error_var * 100
        
        stability_results.append({
            'Disagreement_Bin': f"{disagreement_bins[i-1]:.2f}-{disagreement_bins[i]:.2f}",
            'Avg_Disagreement': bin_disagreement,
            'Sample_Count': np.sum(mask),
            'LGB_RMSLE': bin_lgb_rmsle,
            'CAT_RMSLE': bin_cat_rmsle,
            'META_RMSLE': bin_meta_rmsle,
            'Error_Improvement_Pct': improvement_pct,
            'LGB_Error_Var': lgb_error_var,
            'CAT_Error_Var': cat_error_var,
            'META_Error_Var': meta_error_var,
            'Variance_Reduction_Pct': var_reduction_pct
        })
    
    # Convert to DataFrame
    stability_df = pd.DataFrame(stability_results)
    
    # Create visualizations
    try:
        # Create directory
        os.makedirs("stability_analysis", exist_ok=True)
        
        # Plot error improvement by disagreement
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.bar(range(len(stability_df)), stability_df['Error_Improvement_Pct'], color='green')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Model Disagreement (Increasing →)')
        plt.ylabel('Error Improvement %')
        plt.title('Meta-Model Error Improvement by Base Model Disagreement')
        plt.xticks(range(len(stability_df)), stability_df['Disagreement_Bin'], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.bar(range(len(stability_df)), stability_df['Variance_Reduction_Pct'], color='blue')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Model Disagreement (Increasing →)')
        plt.ylabel('Variance Reduction %')
        plt.title('Meta-Model Error Variance Reduction by Base Model Disagreement')
        plt.xticks(range(len(stability_df)), stability_df['Disagreement_Bin'], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("stability_analysis/disagreement_analysis.png")
        plt.close()
        
        # Plot RMSLE by disagreement
        plt.figure(figsize=(12, 6))
        
        plt.plot(range(len(stability_df)), stability_df['LGB_RMSLE'], 'o-', label='LightGBM')
        plt.plot(range(len(stability_df)), stability_df['CAT_RMSLE'], 'o-', label='CatBoost')
        plt.plot(range(len(stability_df)), stability_df['META_RMSLE'], 'o-', label='Meta-Model')
        
        plt.xlabel('Model Disagreement (Increasing →)')
        plt.ylabel('RMSLE')
        plt.title('Model Performance by Base Model Disagreement')
        plt.xticks(range(len(stability_df)), stability_df['Disagreement_Bin'], rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("stability_analysis/rmsle_by_disagreement.png")
        plt.close()
        
        # Save results
        stability_df.to_csv("stability_analysis/ensemble_stability.csv", index=False)
        
    except Exception as e:
        logging.error(f"Error creating stability visualizations: {e}")
    
    return stability_df

def objective_lgbm(trial):
    """Optuna objective function for LightGBM."""
    params = {
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.001, 0.05, log=True),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 4, 512),
        'max_depth': trial.suggest_int('lgbm_max_depth', 2, 30),
        'feature_fraction': trial.suggest_float('lgbm_feature_fraction', 0.2, 1.0),
        'bagging_fraction': trial.suggest_float('lgbm_bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('lgbm_bagging_freq', 0, 10),
        'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 10, 2000),
        'lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1e-8, 1000.0, log=True),
        'lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1e-8, 1000.0, log=True),
    }    # Add fixed params
    params.update({
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric': 'None', # Crucial when using feval - using 'None' instead of None
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

def objective_catboost(trial):
    """Optuna objective function for CatBoost."""
    params = {
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 0.1, 10.0, log=True),
        'random_strength': trial.suggest_float('cat_random_strength', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('cat_bagging_temperature', 0.0, 10.0),
        'border_count': trial.suggest_int('cat_border_count', 32, 255),
        'rsm': trial.suggest_float('cat_rsm', 0.1, 1.0),  # Feature fraction equivalent
    }
    # Add fixed params
    params.update({
        'loss_function': 'MAE',  # MAE works well for RMSLE
        'iterations': 2000,
        'early_stopping_rounds': 200,
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False
    })

    model = CatBoostRegressor(**params)
    model.fit(
        train_split_df[FEATURES], train_split_df[TARGET],
        eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
        use_best_model=True,
        verbose=False
    )
    preds = model.predict(valid_df[FEATURES])
    score = rmsle(valid_df[TARGET], preds)
    return score

def objective_meta(trial):
    """Optuna objective function for the meta-model using optimized base models."""
    # Get the best LightGBM and CatBoost models from their respective studies
    try:
        lgbm_study = optuna.load_study(study_name=f"{OPTUNA_STUDY_NAME}_lgbm", storage=OPTUNA_DB)
        cat_study = optuna.load_study(study_name=f"{OPTUNA_STUDY_NAME}_cat", storage=OPTUNA_DB)
        lgbm_params = lgbm_study.best_params
        cat_params = cat_study.best_params
    except Exception as e:
        logging.error(f"Error loading base model studies: {e}")
        # Use default parameters if studies are not found
        lgbm_params = {}
        cat_params = {}
    
    # Train base models with K-fold cross-validation to get OOF predictions
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize arrays for out-of-fold predictions
    oof_lgb = np.zeros(len(train_split_df))
    oof_cat = np.zeros(len(train_split_df))
    valid_lgb = np.zeros(len(valid_df))
    valid_cat = np.zeros(len(valid_df))
      # Prepare LightGBM parameters
    lgbm_model_params = {
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
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
        'metric': 'None'  # Crucial when using feval - this must be 'None' (not None)
    }
    # Update with optimized parameters
    for key in lgbm_params:
        # Remove 'lgbm_' prefix from parameter names
        param_name = key.replace('lgbm_', '')
        lgbm_model_params[param_name] = lgbm_params[key]
    
    # Prepare CatBoost parameters
    cat_model_params = {
        'loss_function': 'MAE',  # MAE works well for RMSLE
        'iterations': 2000,
        'early_stopping_rounds': 200,
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False
    }
    # Update with optimized parameters
    for key in cat_params:
        # Remove 'cat_' prefix from parameter names
        param_name = key.replace('cat_', '')
        cat_model_params[param_name] = cat_params[key]
    
    # Train base models and get predictions
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_split_df)):
        X_train, y_train = train_split_df.iloc[train_idx][FEATURES], train_split_df.iloc[train_idx][TARGET]
        X_val, y_val = train_split_df.iloc[val_idx][FEATURES], train_split_df.iloc[val_idx][TARGET]
        
        print(f"DEBUG: Training fold {fold+1}/{n_folds}")
        print(f"DEBUG: Train indices: {train_idx}")
        print(f"DEBUG: Validation indices: {val_idx}")
        print(f"DEBUG: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"DEBUG: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"DEBUG: lgbm_model_params: {lgbm_model_params}")

        # Train LightGBM
        lgb_model = LGBMRegressor(**lgbm_model_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=lgb_rmsle,
            callbacks=[early_stopping_with_overfit(stopping_rounds=200, overfit_rounds=15)]
        )
        
        # Get OOF and validation predictions
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        valid_lgb += lgb_model.predict(valid_df[FEATURES]) / n_folds
        
        # Train CatBoost
        cat_model = CatBoostRegressor(**cat_model_params)
        cat_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            use_best_model=True,
            verbose=False
        )
        
        # Get OOF and validation predictions
        oof_cat[val_idx] = cat_model.predict(X_val)
        valid_cat += cat_model.predict(valid_df[FEATURES]) / n_folds
    
    # Create meta-features for meta-model
    meta_features_train = pd.DataFrame({
        'lgb_pred': oof_lgb,
        'cat_pred': oof_cat,
        # Error features
        'abs_lgb_error': np.abs(train_split_df[TARGET].values - oof_lgb),
        'abs_cat_error': np.abs(train_split_df[TARGET].values - oof_cat),
        # Model agreement features
        'pred_diff': np.abs(oof_lgb - oof_cat),
        'pred_mean': (oof_lgb + oof_cat) / 2,
        'pred_product': oof_lgb * oof_cat,
        # Important original features
        'checkout_price': train_split_df['checkout_price'].values,
        'base_price': train_split_df['base_price'].values,
        'discount': train_split_df['discount'].values,
        'weekofyear': train_split_df['weekofyear'].values
    })
    
    meta_features_valid = pd.DataFrame({
        'lgb_pred': valid_lgb,
        'cat_pred': valid_cat,
        # Use average errors from training
        'abs_lgb_error': meta_features_train['abs_lgb_error'].mean(),
        'abs_cat_error': meta_features_train['abs_cat_error'].mean(),
        # Model agreement features
        'pred_diff': np.abs(valid_lgb - valid_cat),
        'pred_mean': (valid_lgb + valid_cat) / 2,
        'pred_product': valid_lgb * valid_cat,
        # Important original features
        'checkout_price': valid_df['checkout_price'].values,
        'base_price': valid_df['base_price'].values,
        'discount': valid_df['discount'].values,
        'weekofyear': valid_df['weekofyear'].values
    })
    
    # Tune meta-model hyperparameters
    meta_model_type = trial.suggest_categorical('meta_model_type', ['ridge', 'linear', 'lgbm'])
    
    if meta_model_type == 'ridge':
        alpha = trial.suggest_float('meta_alpha', 0.01, 10.0, log=True)
        meta_model = Ridge(alpha=alpha)
    elif meta_model_type == 'linear':
        meta_model = LinearRegression()
    else:  # lgbm
        lgbm_meta_params = {
            'objective': 'regression_l1',
            'boosting_type': 'gbdt',
            'n_estimators': 100,
            'learning_rate': trial.suggest_float('meta_learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('meta_num_leaves', 4, 31),
            'max_depth': trial.suggest_int('meta_max_depth', 2, 7),
            'seed': SEED,
            'verbose': -1
        }
        meta_model = LGBMRegressor(**lgbm_meta_params)
    
    # Train meta-model
    meta_model.fit(meta_features_train, train_split_df[TARGET])
    
    # Make meta-model predictions
    valid_meta = meta_model.predict(meta_features_valid)
    valid_meta = np.clip(valid_meta, 0, None)  # Clip negative predictions
    
    # Calculate meta-model performance
    score = rmsle(valid_df[TARGET], valid_meta)
    return score

def objective(trial):
    """Legacy Optuna objective function - kept for backward compatibility."""
    return objective_lgbm(trial)








# --- Optuna Hyperparameter Tuning ---
logging.info("Starting comprehensive Optuna hyperparameter tuning for all models...")

# --- LightGBM Tuning ---
logging.info("Tuning LightGBM model...")
try:
    lgbm_study = optuna.load_study(study_name=f"{OPTUNA_STUDY_NAME}_lgbm", storage=OPTUNA_DB)
    logging.info(f"Loaded existing LightGBM Optuna study from {OPTUNA_DB}")
except Exception:
    lgbm_study = optuna.create_study(
        direction="minimize", 
        study_name=f"{OPTUNA_STUDY_NAME}_lgbm", 
        storage=OPTUNA_DB, 
        sampler=optuna.samplers.TPESampler(constant_liar=True)
    )
    logging.info(f"Created new LightGBM Optuna study at {OPTUNA_DB}")

# Run LightGBM optimization
lgbm_study.optimize(objective_lgbm, n_trials=OPTUNA_TRIALS, timeout=1200)  # 20 minutes max
logging.info(f"LightGBM tuning completed. Best score: {lgbm_study.best_value:.5f}")
logging.info(f"Best LightGBM parameters: {lgbm_study.best_params}")

# --- CatBoost Tuning ---
logging.info("Tuning CatBoost model...")
try:
    cat_study = optuna.load_study(study_name=f"{OPTUNA_STUDY_NAME}_cat", storage=OPTUNA_DB)
    logging.info(f"Loaded existing CatBoost Optuna study from {OPTUNA_DB}")
except Exception:
    cat_study = optuna.create_study(
        direction="minimize", 
        study_name=f"{OPTUNA_STUDY_NAME}_cat", 
        storage=OPTUNA_DB, 
        sampler=optuna.samplers.TPESampler(constant_liar=True)
    )
    logging.info(f"Created new CatBoost Optuna study at {OPTUNA_DB}")

# Run CatBoost optimization
cat_study.optimize(objective_catboost, n_trials=OPTUNA_TRIALS, timeout=1200)  # 20 minutes max
logging.info(f"CatBoost tuning completed. Best score: {cat_study.best_value:.5f}")
logging.info(f"Best CatBoost parameters: {cat_study.best_params}")

# --- Meta-Model Tuning ---
logging.info("Tuning meta-model using optimized base models...")
try:
    meta_study = optuna.load_study(study_name=f"{OPTUNA_STUDY_NAME}_meta", storage=OPTUNA_DB)
    logging.info(f"Loaded existing meta-model Optuna study from {OPTUNA_DB}")
except Exception:
    meta_study = optuna.create_study(
        direction="minimize", 
        study_name=f"{OPTUNA_STUDY_NAME}_meta", 
        storage=OPTUNA_DB, 
        sampler=optuna.samplers.TPESampler(constant_liar=True)
    )
    logging.info(f"Created new meta-model Optuna study at {OPTUNA_DB}")

# Run meta-model optimization
meta_study.optimize(objective_meta, n_trials=max(5, OPTUNA_TRIALS), timeout=600)  # 10 minutes max, fewer trials
logging.info(f"Meta-model tuning completed. Best score: {meta_study.best_value:.5f}")
logging.info(f"Best meta-model parameters: {meta_study.best_params}")

# --- Maintain backward compatibility for legacy code ---
# Also create/update the default study for backward compatibility
# try:
#     study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
#     logging.info(f"Loaded existing legacy Optuna study from {OPTUNA_DB}")
# except Exception:
#     study = optuna.create_study(
#         direction="minimize", 
#         study_name=OPTUNA_STUDY_NAME, 
#         storage=OPTUNA_DB, 
#         sampler=optuna.samplers.TPESampler(constant_liar=True)
#     )
#     logging.info(f"Created new legacy Optuna study at {OPTUNA_DB}")

# # Run a single trial to maintain backward compatibility
# study.optimize(objective, n_trials=1)
# best_params = study.best_params
# logging.info(f"Best Optuna params: {best_params}")
# logging.info(f"Best validation RMSLE: {study.best_value:.5f}")

# --- Final Model Training ---
logging.info("Training advanced ensemble model with stacking architecture...")

# Train the ensemble stacking model
ensemble_result = train_stacking_ensemble(
    train_df=train_df, 
    valid_df=valid_df, 
    features=FEATURES, 
    target=TARGET, 
    n_folds=5  # 5-fold cross-validation
)

# Create diagnostic plots
plot_ensemble_diagnostics(ensemble_result, train_df, valid_df, TARGET)

# Track and visualize learning curves
track_learning_curves(
    lgb_models=ensemble_result['models']['lgb'],
    cat_models=ensemble_result['models']['cat'],
    valid_df=valid_df,
    features=FEATURES,
    target=TARGET
)

# --- Recursive Prediction ---
logging.info("Starting recursive prediction on the test set using ensemble model...")
# Prepare the combined data history (training data + test structure)
# We need the structure of test_df but will fill num_orders recursively
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

# Extract test weeks
test_weeks = sorted(test_df['week'].unique())

# Perform recursive prediction using the ensemble model
predictions_df = ensemble_predict_recursive(
    ensemble_result=ensemble_result,
    history_df=history_df,
    test_weeks=test_weeks,
    features=FEATURES,
    target=TARGET
)

# Extract final predictions for the original test set IDs
final_predictions_df = predictions_df.loc[predictions_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int) # Final conversion to int
final_predictions_df['id'] = final_predictions_df['id'].astype(int)

# --- Create Submission File ---
submission_path = f"{SUBMISSION_FILE_PREFIX}_ensemble_stacking.csv"
final_predictions_df.to_csv(submission_path, index=False)
logging.info(f"Ensemble stacking submission file saved to {submission_path}")

# --- Compare Single Models vs Ensemble ---
logging.info("Comparing model performances in validation:")
performance_comparison = pd.DataFrame({
    'Model': ['LightGBM', 'CatBoost', 'Simple Average', 'Stacking Ensemble'],
    'Validation RMSLE': [
        ensemble_result['scores']['valid']['lgb'],
        ensemble_result['scores']['valid']['cat'],
        ensemble_result['scores']['valid']['avg'],
        ensemble_result['scores']['valid']['meta']
    ]
})

# Calculate relative improvement of ensemble over best base model
best_base_rmsle = min(ensemble_result['scores']['valid']['lgb'], ensemble_result['scores']['valid']['cat'])
ensemble_rmsle = ensemble_result['scores']['valid']['meta']
relative_improvement = (best_base_rmsle - ensemble_rmsle) / best_base_rmsle * 100

performance_comparison.to_csv("ensemble_performance_comparison.csv", index=False)
logging.info(f"Performance comparison:\n{performance_comparison}")
logging.info(f"Ensemble provides {relative_improvement:.2f}% improvement over best base model")

# --- Segment-wise Performance Analysis ---
logging.info("Analyzing model performance across different order volume segments...")

# Create segments based on order volume
valid_targets = valid_df[TARGET].values
segment_bins = [0, 10, 25, 50, 100, 500, float('inf')]
segment_labels = ['0-10', '11-25', '26-50', '51-100', '101-500', '500+']
valid_segments = pd.cut(valid_targets, bins=segment_bins, labels=segment_labels)

# Get predictions
valid_lgb_preds = ensemble_result['predictions']['valid']['lgb']
valid_cat_preds = ensemble_result['predictions']['valid']['cat']
valid_avg_preds = ensemble_result['predictions']['valid']['avg']
valid_meta_preds = ensemble_result['predictions']['valid']['meta']

# Initialize segment analysis dataframe
segment_analysis = []

# Calculate metrics for each segment
for segment in segment_labels:
    mask = valid_segments == segment
    if np.sum(mask) > 0:  # Only analyze segments with data
        segment_targets = valid_targets[mask]
        segment_lgb_preds = valid_lgb_preds[mask]
        segment_cat_preds = valid_cat_preds[mask]
        segment_avg_preds = valid_avg_preds[mask]
        segment_meta_preds = valid_meta_preds[mask]
        
        # Calculate metrics
        segment_analysis.append({
            'Segment': segment,
            'Count': np.sum(mask),
            'Percentage': np.sum(mask) / len(valid_targets) * 100,
            'Avg_Order_Volume': np.mean(segment_targets),
            'LGB_RMSLE': rmsle(segment_targets, segment_lgb_preds),
            'CAT_RMSLE': rmsle(segment_targets, segment_cat_preds),
            'AVG_RMSLE': rmsle(segment_targets, segment_avg_preds),
            'META_RMSLE': rmsle(segment_targets, segment_meta_preds),
            'META_vs_Best_Improvement': (min(rmsle(segment_targets, segment_lgb_preds), 
                                            rmsle(segment_targets, segment_cat_preds)) - 
                                         rmsle(segment_targets, segment_meta_preds)) / 
                                        min(rmsle(segment_targets, segment_lgb_preds), 
                                            rmsle(segment_targets, segment_cat_preds)) * 100 if min(rmsle(segment_targets, segment_lgb_preds), 
                                                                                                 rmsle(segment_targets, segment_cat_preds)) > 0 else 0
        })

# Convert to DataFrame and save
segment_analysis_df = pd.DataFrame(segment_analysis)
segment_analysis_df.to_csv("ensemble_segment_analysis.csv", index=False)
logging.info(f"Segment analysis saved to ensemble_segment_analysis.csv")

# --- Visualize Segment Performance ---
try:
    plt.figure(figsize=(14, 10))
    
    # Plot RMSLE by segment
    plt.subplot(2, 1, 1)
    segments = segment_analysis_df['Segment']
    x = np.arange(len(segments))
    width = 0.2
    
    plt.barh(x - width/2, segment_analysis_df['LGB_RMSLE'], width, label='LightGBM')
    plt.barh(x + width/2, segment_analysis_df['CAT_RMSLE'], width, label='CatBoost')
    plt.barh(x + width*1.5, segment_analysis_df['AVG_RMSLE'], width, label='Simple Avg')
    plt.barh(x + width*2.5, segment_analysis_df['META_RMSLE'], width, label='Meta-Model')
    
    plt.xlabel('RMSLE')
    plt.ylabel('Order Volume Segment')
    plt.title('Model Performance by Order Volume Segment')
    plt.yticks(x, segments)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    
    # Plot improvement percentage by segment
    plt.subplot(2, 1, 2)
    plt.bar(x, segment_analysis_df['META_vs_Best_Improvement'], color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('Order Volume Segment')
    plt.ylabel('Improvement %')
    plt.title('Meta-Model Improvement Over Best Base Model by Segment')
    plt.xticks(x, segments)
    for i, v in enumerate(segment_analysis_df['META_vs_Best_Improvement']):
        plt.text(i, v + 0.5 if v >= 0 else v - 2, f"{v:.1f}%", ha='center')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ensemble_segment_performance.png")
    plt.close()
    
    # Create summary report
    with open("ensemble_model_report.md", "w") as f:
        f.write("# Ensemble Model Performance Report\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Performance\n\n")
        f.write("| Model | Validation RMSLE |\n")
        f.write("|-------|----------------:|\n")
        for i, row in performance_comparison.iterrows():
            f.write(f"| {row['Model']} | {row['Validation RMSLE']:.5f} |\n")
        
        f.write(f"\nEnsemble provides **{relative_improvement:.2f}%** improvement over best base model\n\n")
        
        f.write("## Segment Analysis\n\n")
        f.write("| Segment | Count | % of Data | Avg Volume | LightGBM | CatBoost | Simple Avg | Meta-Model | Improvement |\n")
        f.write("|---------|------:|----------:|-----------:|---------:|---------:|-----------:|-----------:|------------:|\n")
        for i, row in segment_analysis_df.iterrows():
            f.write(f"| {row['Segment']} | {row['Count']:.0f} | {row['Percentage']:.1f}% | {row['Avg_Order_Volume']:.1f} | {row['LGB_RMSLE']:.5f} | {row['CAT_RMSLE']:.5f} | {row['AVG_RMSLE']:.5f} | {row['META_RMSLE']:.5f} | {row['META_vs_Best_Improvement']:.2f}% |\n")
        
        f.write("\n\n## Key Findings\n\n")
        
        # Identify segments where ensemble performs best/worst
        best_segment = segment_analysis_df.loc[segment_analysis_df['META_vs_Best_Improvement'].idxmax()]
        worst_segment = segment_analysis_df.loc[segment_analysis_df['META_vs_Best_Improvement'].idxmin()]
        
        f.write(f"- Ensemble performs best in the **{best_segment['Segment']}** orders segment with **{best_segment['META_vs_Best_Improvement']:.2f}%** improvement\n")
        f.write(f"- Ensemble performs worst in the **{worst_segment['Segment']}** orders segment with **{worst_segment['META_vs_Best_Improvement']:.2f}%** improvement\n")
        
        # Identify which model performs best in each segment
        f.write("\n### Best Model by Segment\n\n")
        for i, row in segment_analysis_df.iterrows():
            models = ['LightGBM', 'CatBoost', 'Simple Average', 'Meta-Model']
            scores = [row['LGB_RMSLE'], row['CAT_RMSLE'], row['AVG_RMSLE'], row['META_RMSLE']]
            best_model = models[np.argmin(scores)]
            f.write(f"- **{row['Segment']}**: {best_model} (RMSLE: {min(scores):.5f})\n")
        
        f.write("\n\n## Visualizations\n\n")
        f.write("### Model Performance by Segment\n")
        f.write("![Model Performance by Segment](ensemble_segment_performance.png)\n\n")
        f.write("### Error Distribution\n")
        f.write("![Error Distribution](ensemble_validation/error_distribution.png)\n\n")
        f.write("### Error by Volume\n")
        f.write("![Error by Volume](ensemble_validation/error_by_volume.png)\n\n")
        
    logging.info(f"Ensemble model report generated: ensemble_model_report.md")
    
    # --- Feature-wise Error Correction Analysis ---
    logging.info("Analyzing feature-specific error correction patterns...")
    
    # Perform feature-wise error analysis
    feature_error_analysis = analyze_feature_error_correction(
        ensemble_result=ensemble_result,
        valid_df=valid_df,
        features=FEATURES,
        target=TARGET
    )
    
    # Create directory for feature analysis
    feature_dir = "feature_analysis"
    os.makedirs(feature_dir, exist_ok=True)
    
    # Create visualizations for each feature
    for feature, analysis_df in feature_error_analysis.items():
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot RMSLE by feature value bin
            plt.subplot(2, 1, 1)
            x = np.arange(len(analysis_df))
            width = 0.2
            
            plt.barh(x - width*1.5, analysis_df['LGB_RMSLE'], width, label='LightGBM')
            plt.barh(x - width/2, analysis_df['CAT_RMSLE'], width, label='CatBoost')
            plt.barh(x + width/2, analysis_df['AVG_RMSLE'], width, label='Simple Average')
            plt.barh(x + width*1.5, analysis_df['META_RMSLE'], width, label='Meta-Model')
            
            plt.xlabel(f'{feature} Bin')
            plt.ylabel('RMSLE')
            plt.title(f'Model Performance by {feature} Value')
            plt.xticks(x, analysis_df['Bin'], rotation=45)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Plot improvement percentage by feature value bin
            plt.subplot(2, 1, 2)
            plt.bar(x, analysis_df['Improvement_Pct'], color='green')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.xlabel(f'{feature} Bin')
            plt.ylabel('Improvement %')
            plt.title(f'Meta-Model Improvement Over Best Base Model by {feature}')
            plt.xticks(x, analysis_df['Bin'], rotation=45)
            
            for i, v in enumerate(analysis_df['Improvement_Pct']):
                plt.text(i, v + 0.5 if v >= 0 else v - 2, f"{v:.1f}%", ha='center')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{feature_dir}/{feature}_error_analysis.png")
            plt.close()
            
            # Save analysis to CSV
            analysis_df.to_csv(f"{feature_dir}/{feature}_error_analysis.csv", index=False)
            
        except Exception as e:
            logging.error(f"Error creating visualization for feature {feature}: {e}")
    
    # Create summary of feature improvements
    feature_improvements = []
    
    for feature, analysis_df in feature_error_analysis.items():
        if not analysis_df.empty:
            # Calculate weighted average improvement
            weighted_improvement = np.average(
                analysis_df['Improvement_Pct'], 
                weights=analysis_df['Count']
            )
            
            # Find bins with best and worst improvement
            best_bin_idx = analysis_df['Improvement_Pct'].idxmax()
            worst_bin_idx = analysis_df['Improvement_Pct'].idxmin()
            
            feature_improvements.append({
                'Feature': feature,
                'Avg_Improvement': weighted_improvement,
                'Best_Bin': analysis_df.loc[best_bin_idx, 'Bin'],
                'Best_Improvement': analysis_df.loc[best_bin_idx, 'Improvement_Pct'],
                'Worst_Bin': analysis_df.loc[worst_bin_idx, 'Bin'],
                'Worst_Improvement': analysis_df.loc[worst_bin_idx, 'Improvement_Pct']
            })
    
    # Create and save feature improvement summary
    feature_improvements_df = pd.DataFrame(feature_improvements)
    feature_improvements_df = feature_improvements_df.sort_values('Avg_Improvement', ascending=False)
    feature_improvements_df.to_csv(f"{feature_dir}/feature_improvement_summary.csv", index=False)
    
    # Visualize top 10 features by improvement
    plt.figure(figsize=(14, 7))
    top_features = feature_improvements_df.head(10)
    
    plt.barh(np.arange(len(top_features)), top_features['Avg_Improvement'], color='green')
    plt.yticks(np.arange(len(top_features)), top_features['Feature'])
    plt.xlabel('Average Improvement %')
    plt.title('Top 10 Features by Error Correction Improvement')
    plt.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(top_features['Avg_Improvement']):
        plt.text(v + 0.1, i, f"{v:.2f}%", va='center')
    
    plt.tight_layout()
    plt.savefig(f"{feature_dir}/top_features_by_improvement.png")
    plt.close()
    
    logging.info(f"Feature-wise error analysis saved to {feature_dir}/")
    
    # Append to report
    with open("ensemble_model_report.md", "a") as f:
        f.write("\n\n## Feature-wise Error Correction Analysis\n\n")
        f.write("### Top 10 Features by Error Correction Effectiveness\n\n")
        f.write("| Feature | Avg Improvement | Best Bin | Best Improvement | Worst Bin | Worst Improvement |\n")
        f.write("|---------|---------------:|----------|------------------|-----------|------------------:|\n")
        
        for _, row in top_features.iterrows():
            f.write(f"| {row['Feature']} | {row['Avg_Improvement']:.2f}% | {row['Best_Bin']} | {row['Best_Improvement']:.2f}% | {row['Worst_Bin']} | {row['Worst_Improvement']:.2f}% |\n")
        
        f.write("\n![Top Features by Improvement](feature_analysis/top_features_by_improvement.png)\n\n")
        
    logging.info("Feature analysis added to ensemble model report")
    
    # --- Ensemble Stability Analysis ---
    logging.info("Analyzing ensemble stability and robustness...")
    
    # Perform stability analysis
    stability_df = analyze_ensemble_stability(
        ensemble_result=ensemble_result,
        valid_df=valid_df,
        target=TARGET
    )
    
    # Calculate overall stability metrics
    avg_error_improvement = stability_df['Error_Improvement_Pct'].mean()
    avg_variance_reduction = stability_df['Variance_Reduction_Pct'].mean()
    
    logging.info(f"Average error improvement: {avg_error_improvement:.2f}%")
    logging.info(f"Average error variance reduction: {avg_variance_reduction:.2f}%")
    
    # Append to report
    with open("ensemble_model_report.md", "a") as f:
        f.write("\n\n## Ensemble Stability Analysis\n\n")
        f.write(f"The ensemble model shows an average error improvement of **{avg_error_improvement:.2f}%** and ")
        f.write(f"error variance reduction of **{avg_variance_reduction:.2f}%** compared to the best base model.\n\n")
        
        f.write("### Stability by Model Disagreement\n\n")
        f.write("When base models disagree more, the meta-model provides the following benefits:\n\n")
        f.write("![Disagreement Analysis](stability_analysis/disagreement_analysis.png)\n\n")
        f.write("![RMSLE by Disagreement](stability_analysis/rmsle_by_disagreement.png)\n\n")
        
        # Find where meta-model provides most benefit
        max_improvement_idx = stability_df['Error_Improvement_Pct'].idxmax()
        max_var_reduction_idx = stability_df['Variance_Reduction_Pct'].idxmax()
        
        if not stability_df.empty and max_improvement_idx is not None and max_var_reduction_idx is not None:
            f.write("### Key Stability Insights\n\n")
            f.write(f"- Highest error improvement ({stability_df.loc[max_improvement_idx, 'Error_Improvement_Pct']:.2f}%) ")
            f.write(f"occurs when models disagree by {stability_df.loc[max_improvement_idx, 'Avg_Disagreement']:.2f} units\n")
            f.write(f"- Highest variance reduction ({stability_df.loc[max_var_reduction_idx, 'Variance_Reduction_Pct']:.2f}%) ")
            f.write(f"occurs when models disagree by {stability_df.loc[max_var_reduction_idx, 'Avg_Disagreement']:.2f} units\n")
            logging.info("Stability analysis added to ensemble model report")

except Exception as e:
    logging.error(f"Error during analysis and visualization: {e}")    
    
# --- SHAP Analysis ---
logging.info("Calculating SHAP values for ensemble models...")
try:
    # Sample data for SHAP to keep computation reasonable
    if len(train_df) > N_SHAP_SAMPLES:
        shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
    else:
        shap_sample = train_df.copy()

    # Create folder for SHAP plots
    shap_dir = "ensemble_shap"
    os.makedirs(shap_dir, exist_ok=True)
    
    # Analyze LightGBM model (first fold)
    logging.info("Analyzing LightGBM model with SHAP...")
    lgb_model = ensemble_result['models']['lgb'][0]
    lgb_explainer = shap.TreeExplainer(lgb_model)
    lgb_shap_values = lgb_explainer.shap_values(shap_sample[FEATURES])
    
    # Analyze CatBoost model (first fold)
    logging.info("Analyzing CatBoost model with SHAP...")
    cat_model = ensemble_result['models']['cat'][0]
    cat_explainer = shap.TreeExplainer(cat_model)
    cat_shap_values = cat_explainer.shap_values(shap_sample[FEATURES])
    
    # Save SHAP values and importance for both models
    lgb_shap_values_df = pd.DataFrame(lgb_shap_values, columns=FEATURES)
    lgb_shap_values_df.to_csv(f"{shap_dir}/{SHAP_FILE_PREFIX}_lgb_values.csv", index=False)
    
    cat_shap_values_df = pd.DataFrame(cat_shap_values, columns=FEATURES)
    cat_shap_values_df.to_csv(f"{shap_dir}/{SHAP_FILE_PREFIX}_cat_values.csv", index=False)
    
    # Calculate feature importance based on SHAP values
    lgb_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(lgb_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    lgb_importance_df.to_csv(f"{shap_dir}/{SHAP_FILE_PREFIX}_lgb_importances.csv", index=False)
    
    cat_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(cat_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    cat_importance_df.to_csv(f"{shap_dir}/{SHAP_FILE_PREFIX}_cat_importances.csv", index=False)
    
    # Create combined importance dataframe to see how models differ
    combined_importance = pd.merge(
        lgb_importance_df.rename(columns={'mean_abs_shap': 'lgb_importance'}),
        cat_importance_df.rename(columns={'mean_abs_shap': 'cat_importance'}),
        on='feature'
    )
    combined_importance['avg_importance'] = (combined_importance['lgb_importance'] + combined_importance['cat_importance']) / 2
    combined_importance['importance_diff'] = np.abs(combined_importance['lgb_importance'] - combined_importance['cat_importance'])
    combined_importance = combined_importance.sort_values('avg_importance', ascending=False)
    combined_importance.to_csv(f"{shap_dir}/{SHAP_FILE_PREFIX}_combined_importances.csv", index=False)

    # Generate SHAP plots
    logging.info("Generating SHAP plots for ensemble models...")
    
    # LightGBM Summary Plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(lgb_shap_values, shap_sample[FEATURES], show=False)
    plt.title('LightGBM SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{shap_dir}/{SHAP_FILE_PREFIX}_lgb_summary.png")
    plt.close()
    
    # CatBoost Summary Plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(cat_shap_values, shap_sample[FEATURES], show=False)
    plt.title('CatBoost SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{shap_dir}/{SHAP_FILE_PREFIX}_cat_summary.png")
    plt.close()
    
    # Top 20 Feature Comparison (Bar Plot)
    plt.figure(figsize=(12, 10))
    top_features = combined_importance.head(20)['feature'].tolist()
    top_importance = combined_importance[combined_importance['feature'].isin(top_features)].copy()
    
    # Sort by average importance for consistent ordering
    top_importance = top_importance.sort_values('avg_importance')
    
    # Plot for comparison
    bar_width = 0.35
    y_pos = np.arange(len(top_features))
    
    plt.barh(y_pos - bar_width/2, top_importance['lgb_importance'], bar_width, label='LightGBM')
    plt.barh(y_pos + bar_width/2, top_importance['cat_importance'], bar_width, label='CatBoost')
    
    plt.yticks(y_pos, top_importance['feature'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 20 Feature Importance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{shap_dir}/{SHAP_FILE_PREFIX}_model_comparison.png")
    plt.close()
    
    # Feature Importance Correlation Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(combined_importance['lgb_importance'], combined_importance['cat_importance'], alpha=0.7)
    
    # Add identity line
    max_val = max(combined_importance['lgb_importance'].max(), combined_importance['cat_importance'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    # Label outlier points (features where models disagree most)
    for _, row in combined_importance.nlargest(10, 'importance_diff').iterrows():
        plt.annotate(row['feature'], 
                    (row['lgb_importance'], row['cat_importance']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('LightGBM Feature Importance')
    plt.ylabel('CatBoost Feature Importance')
    plt.title('Feature Importance Correlation Between Models')
    plt.tight_layout()
    plt.savefig(f"{shap_dir}/{SHAP_FILE_PREFIX}_importance_correlation.png")
    plt.close()
    
    logging.info(f"SHAP analysis saved to {shap_dir}/")

except Exception as e:
    logging.error(f"Error during SHAP analysis: {e}")

# --- Validation Plot ---
logging.info("Generating ensemble validation comparison plots...")
try:
    # Extract predictions
    lgb_preds = ensemble_result['predictions']['valid']['lgb']
    cat_preds = ensemble_result['predictions']['valid']['cat']
    avg_preds = ensemble_result['predictions']['valid']['avg']
    meta_preds = ensemble_result['predictions']['valid']['meta']
    
    # Create a directory for validation plots
    valid_dir = "ensemble_validation"
    os.makedirs(valid_dir, exist_ok=True)
    
    # Create a combined plot of all models
    plt.figure(figsize=(15, 10))
    
    # Individual scatter plots
    plt.subplot(2, 2, 1)
    plt.scatter(valid_df[TARGET], lgb_preds, alpha=0.5, s=7, color='blue')
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
             [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2)
    plt.xlabel("Actual Orders")
    plt.ylabel("Predicted Orders")
    lgb_rmsle_score = rmsle(valid_df[TARGET], lgb_preds)
    plt.title(f"LightGBM - RMSLE: {lgb_rmsle_score:.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(valid_df[TARGET], cat_preds, alpha=0.5, s=7, color='green')
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
             [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2)
    plt.xlabel("Actual Orders")
    plt.ylabel("Predicted Orders")
    cat_rmsle_score = rmsle(valid_df[TARGET], cat_preds)
    plt.title(f"CatBoost - RMSLE: {cat_rmsle_score:.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(valid_df[TARGET], avg_preds, alpha=0.5, s=7, color='orange')
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
             [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2)
    plt.xlabel("Actual Orders")
    plt.ylabel("Predicted Orders")
    avg_rmsle_score = rmsle(valid_df[TARGET], avg_preds)
    plt.title(f"Simple Average - RMSLE: {avg_rmsle_score:.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(valid_df[TARGET], meta_preds, alpha=0.5, s=7, color='red')
    plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
             [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2)
    plt.xlabel("Actual Orders")
    plt.ylabel("Predicted Orders")
    meta_rmsle_score = rmsle(valid_df[TARGET], meta_preds)
    plt.title(f"Meta-Model - RMSLE: {meta_rmsle_score:.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{valid_dir}/validation_comparison.png")
    plt.close()
    
    # Create error distribution comparison
    plt.figure(figsize=(12, 8))
    
    # Calculate errors
    lgb_errors = valid_df[TARGET] - lgb_preds
    cat_errors = valid_df[TARGET] - cat_preds
    avg_errors = valid_df[TARGET] - avg_preds
    meta_errors = valid_df[TARGET] - meta_preds
    
    # Plot error distributions
    plt.hist(lgb_errors, bins=50, alpha=0.5, label=f'LightGBM (σ={np.std(lgb_errors):.2f})', color='blue')
    plt.hist(cat_errors, bins=50, alpha=0.5, label=f'CatBoost (σ={np.std(cat_errors):.2f})', color='green')
    plt.hist(avg_errors, bins=50, alpha=0.5, label=f'Simple Avg (σ={np.std(avg_errors):.2f})', color='orange')
    plt.hist(meta_errors, bins=50, alpha=0.5, label=f'Meta-Model (σ={np.std(meta_errors):.2f})', color='red')
    
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{valid_dir}/error_distribution.png")
    plt.close()
    
    # Create error by order volume plot
    plt.figure(figsize=(14, 7))
    
    # Bin actual values and calculate mean absolute errors by bin
    bins = np.linspace(0, valid_df[TARGET].max(), 20)
    bin_indices = np.digitize(valid_df[TARGET], bins)
    
    bin_centers = []
    lgb_mae_by_bin = []
    cat_mae_by_bin = []
    avg_mae_by_bin = []
    meta_mae_by_bin = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 10:  # Only include bins with enough samples
            bin_centers.append((bins[i-1] + bins[i]) / 2)
            lgb_mae_by_bin.append(np.abs(lgb_errors[mask]).mean())
            cat_mae_by_bin.append(np.abs(cat_errors[mask]).mean())
            avg_mae_by_bin.append(np.abs(avg_errors[mask]).mean())
            meta_mae_by_bin.append(np.abs(meta_errors[mask]).mean())
    
    plt.plot(bin_centers, lgb_mae_by_bin, 'o-', label='LightGBM', color='blue')
    plt.plot(bin_centers, cat_mae_by_bin, 'o-', label='CatBoost', color='green')
    plt.plot(bin_centers, avg_mae_by_bin, 'o-', label='Simple Average', color='orange')
    plt.plot(bin_centers, meta_mae_by_bin, 'o-', label='Meta-Model', color='red')
    
    plt.xlabel('Order Volume')
    plt.ylabel('Mean Absolute Error')
    plt.title('Error by Order Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{valid_dir}/error_by_volume.png")
    plt.close()
    
    logging.info(f"Ensemble validation plots saved to {valid_dir}/")

except Exception as e:
    logging.error(f"Error during validation plotting: {e}")

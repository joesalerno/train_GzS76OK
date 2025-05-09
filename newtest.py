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
SEED = random.randint(0, 1000) # Random seed for reproducibility
LAG_WEEKS = [1, 2, 3, 5, 10] # Lags based on num_orders
ROLLING_WINDOWS = [2, 3, 5, 10, 14, 21] # Added 14 and 21
# Other features (not directly dependent on recursive prediction)
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 75 # Number of Optuna trials
OPTUNA_STUDY_NAME = "newertest"
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "newertest_submission"
SHAP_FILE_PREFIX = "shap_newertest"
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
    df_out['center_meal_orders_mean_div'] = df_out['center_orders_mean'] / df_out['meal_orders_mean'].replace(0, 1)
    
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
    
    # Extending polynomial features for rolling statistics (important in SHAP)
    for col in [f'num_orders_rolling_mean_{w}' for w in [3, 5, 14, 21] if f'num_orders_rolling_mean_{w}' in df_out.columns]:
        df_out[f'{col}_sq'] = df_out[col] ** 2
        df_out[f'{col}_sqrt'] = np.sqrt(df_out[col].clip(0))
    
    # Add polynomial features for important numeric columns
    for col in ['checkout_price', 'base_price', 'discount', 'discount_pct', 'price_diff',
                'center_orders_mean', 'meal_orders_mean']:
        if col in df_out.columns:
            df_out[f'{col}_sq'] = df_out[col] ** 2
    
    # Add polynomial features for lag variables (highly important in SHAP)
    for lag in [1, 2, 3, 5, 10]:
        lag_col = f'num_orders_lag_{lag}'
        if lag_col in df_out.columns:
            df_out[f'{lag_col}_sq'] = df_out[lag_col] ** 2
    
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

def add_binary_rolling_means(df, binary_cols=["emailer_for_promotion", "homepage_featured"], 
                       windows=[2, 3, 5, 7, 14, 21]):
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
    col.endswith("_sq") or 
    col.endswith("_sqrt") or
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
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 2000),
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

final_model = LGBMRegressor(**final_params)

# Train on the entire training dataset with eval set for detecting overfitting
train_size = int(0.9 * len(train_df))
train_indices = np.random.choice(len(train_df), train_size, replace=False)
eval_indices = np.array([i for i in range(len(train_df)) if i not in train_indices])

# Use a small eval set to detect overfitting during final model training
final_model.fit(
    train_df[FEATURES], train_df[TARGET], 
    eval_set=[
        (train_df.iloc[train_indices][FEATURES], train_df.iloc[train_indices][TARGET]),
        (train_df.iloc[eval_indices][FEATURES], train_df.iloc[eval_indices][TARGET])
    ],
    eval_metric=lgb_rmsle,
    callbacks=[early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=20, verbose=True)]
)

# --- Recursive Prediction ---
logging.info("Starting recursive prediction on the test set...")
# Prepare the combined data history (training data + test structure)
# We need the structure of test_df but will fill num_orders recursively
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

# Extract seasonality means from train_df for use in recursive prediction
weekofyear_means = train_df.groupby('weekofyear')['num_orders'].mean()
month_means = train_df.groupby('month')['num_orders'].mean()

test_weeks = sorted(test_df['week'].unique())

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

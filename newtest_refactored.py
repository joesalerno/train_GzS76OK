"""
Food delivery order forecasting with optimized feature selection.

This version addresses collinearity issues by:
1. Reducing the number of rolling windows to essential ones (short, medium, long-term)
2. Selectively creating polynomial features based on SHAP importance
3. Focusing on high-value interaction features
4. Using a more targeted feature selection approach
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import shap
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# --- Configuration ---
# Data paths
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"

# Random seed
SEED = random.randint(0, 1000)  # Random seed for remote execution

# Feature engineering window configurations
LAG_WEEKS = [1, 2, 3, 5, 10]  # Lags based on num_orders
ROLLING_WINDOWS = {
    'standard': [2, 5, 14],  # Standard windows for rolling features (short, medium, long-term)
    'stats': [2, 5, 14],  # Windows for basic statistics
    'binary': [2, 5, 14],  # Windows for binary features
    'promo_combine': [7],  # Window for combined promotional effects
    'sums': [4, 16]  # Windows for rolling sums (short and long-term)
}

# Binary column configurations
PROMO_COLUMNS = ["emailer_for_promotion", "homepage_featured"]
EWM_ALPHAS = [0.3, 0.7]  # Alpha values for exponentially weighted means

# Validation configuration
VALIDATION_WEEKS = 8  # Use last 8 weeks for validation

# Optuna configuration
OPTUNA_TRIALS = 75  # Number of Optuna trials
OPTUNA_STUDY_NAME = "ecc"
# Database configuration for Optuna
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"  # Uncomment for SQLite

# Output file configuration
SUBMISSION_FILE_PREFIX = "ecc_submission"
SHAP_FILE_PREFIX = "shap_ecc"
N_SHAP_SAMPLES = 2000

# Core data structure
GROUP_COLS = ["center_id", "meal_id"]
TARGET = "num_orders"

# Temporal features configuration
CYCLICAL_FEATURES = [
    ("weekofyear", 52),  # Week of year, period = 52 weeks
    ("month", 12)  # Month, period = 12 months
]

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Engineering Functions ---
def preprocess_data(df, meal_info, center_info):
    """Merges dataframes and sorts."""
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

def create_temporal_features(df):
    """Creates temporal features from week."""
    df_out = df.copy()
    
    # Week of year
    df_out["weekofyear"] = df_out["week"] % 52
    
    # Month (derived from week)
    df_out["month"] = ((df_out["week"] - 1) // 4) % 12 + 1
    
    # Add cyclical encoding for temporal features
    for col, period in CYCLICAL_FEATURES:
        df_out[f"{col}_sin"] = np.sin(2 * np.pi * df_out[col] / period)
        df_out[f"{col}_cos"] = np.cos(2 * np.pi * df_out[col] / period)
        
        # Add centered quadratic features
        mid_point = period / 2
        scale = mid_point ** 2
        df_out[f"{col}_centered_sq"] = ((df_out[col] - mid_point) ** 2) / scale
    
    return df_out

def create_lag_rolling_features(df, target_col=TARGET, lag_weeks=LAG_WEEKS, rolling_windows=None):
    """Creates lag and rolling window features for a given target column."""
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS['stats']
        
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Lags
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)

    # Rolling features (use shift(1) to avoid data leakage)
    shifted = group[target_col].shift(1)
    
    # Apply rolling means first (most important according to SHAP)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
    
    # Apply standard deviations for selected windows only (to reduce collinearity)
    # Limit to fewer windows for std as they tend to be more correlated
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)

    return df_out

def create_basic_features(df):
    """Creates basic features not dependent on recursive prediction."""
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Price features
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)  # Avoid division by zero
    df_out["price_diff"] = group["checkout_price"].diff()
    
    if all(c in df_out.columns for c in ['checkout_price', 'base_price']):
        df_out['price_ratio'] = df_out['checkout_price'] / df_out['base_price'].replace(0, np.nan)

    # Rolling sums for promo/featured columns
    for col in PROMO_COLUMNS:
        if col in df_out.columns:
            shifted = group[col].shift(1)
            for window in ROLLING_WINDOWS['sums'][:1]:  # Use just first window from sums list
                df_out[f"{col}_rolling_sum_{window}"] = (
                    shifted.rolling(window, min_periods=1).sum().reset_index(drop=True)
                )

    return df_out

def create_group_aggregates(df):
    """Creates aggregations by group levels."""
    df_out = df.copy()
    
    # Define aggregation targets and their functions
    agg_groups = ['center_id', 'meal_id']
    if 'category' in df_out.columns:
        agg_groups.append('category')
    
    agg_funcs = ['mean', 'median', 'std']
    
    # Create basic aggregations for each group
    for group in agg_groups:
        for func in agg_funcs:
            col_name = f"{group.split('_')[0]}_orders_{func}"
            df_out[col_name] = df_out.groupby(group)[TARGET].transform(func)
    
    # High-value cross aggregates
    df_out['center_meal_orders_mean_prod'] = df_out['center_orders_mean'] * df_out['meal_orders_mean']
    df_out['center_meal_orders_median_prod'] = df_out['center_orders_median'] * df_out['meal_orders_median']
    df_out['center_meal_orders_mean_div'] = df_out['center_orders_mean'] / df_out['meal_orders_mean'].replace(0, 1)
    df_out['center_meal_orders_std_prod'] = df_out['center_orders_std'] * df_out['meal_orders_std']
    
    # Weighted center-meal features
    center_total_orders = df_out.groupby('center_id')[TARGET].transform('sum')
    meal_total_orders = df_out.groupby('meal_id')[TARGET].transform('sum')
    df_out['center_meal_ratio'] = df_out['meal_orders_mean'] / df_out['center_orders_mean'].replace(0, 1)
    df_out['center_meal_weighted'] = (
        df_out['center_meal_orders_mean_prod'] / 
        (center_total_orders + meal_total_orders).replace(0, 1)
    )
    
    return df_out

def add_seasonality_features(df, weekofyear_means=None, month_means=None, is_train=True):
    """Adds seasonality features based on weekly and monthly patterns."""
    df_out = df.copy()
    
    if is_train:
        # Calculate means from training data
        weekofyear_means = df_out.groupby('weekofyear')[TARGET].mean()
        month_means = df_out.groupby('month')[TARGET].mean()
    else:
        # Use pre-calculated means from training
        if weekofyear_means is None or month_means is None:
            raise ValueError("When is_train=False, weekofyear_means and month_means must be provided")
    
    # Map the means back to the dataframe
    df_out['mean_orders_by_weekofyear'] = df_out['weekofyear'].map(weekofyear_means)
    df_out['mean_orders_by_month'] = df_out['month'].map(month_means)
    
    return df_out, weekofyear_means, month_means

def add_rolling_features_for_binary_cols(df, binary_cols=None):
    """Creates rolling features for binary columns like promotions."""
    if binary_cols is None:
        binary_cols = PROMO_COLUMNS
        
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)
    
    for col in binary_cols:
        if col in df_out.columns:
            # Shift by 1 to avoid data leakage
            shifted = group[col].shift(1)
            
            # Standard windows - limited selection based on SHAP importance
            for window in ROLLING_WINDOWS['binary']:
                df_out[f"{col}_rolling_mean_{window}"] = (
                    shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
                )
            
            # Cumulative sums for key windows - use only limited windows
            for window in ROLLING_WINDOWS['sums']:
                df_out[f"{col}_rolling_sum_{window}"] = (
                    shifted.rolling(window, min_periods=1).sum().reset_index(drop=True)
                )
            
            # Exponentially weighted means - high SHAP importance with low collinearity
            for alpha in EWM_ALPHAS:
                df_out[f"{col}_ewm_alpha_{alpha}"] = (
                    shifted.ewm(alpha=alpha, min_periods=1).mean().reset_index(drop=True)
                )
    
    # Combined promotional effect
    if all(col in df_out.columns for col in PROMO_COLUMNS):
        df_out["emailer_homepage_combined"] = df_out["emailer_for_promotion"] * df_out["homepage_featured"]
        
        # Rolling means for combined effect - limit to one key window
        combined_shifted = group["emailer_homepage_combined"].shift(1)
        for window in ROLLING_WINDOWS['promo_combine']:
            df_out[f"emailer_homepage_combined_rolling_mean_{window}"] = (
                combined_shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
            )
    
    return df_out

def create_polynomial_features(df):
    """Creates polynomial features for key numeric columns."""
    df_out = df.copy()
    
    # Polynomial features for rolling means (focus on most important ones from SHAP)
    rolling_windows = ROLLING_WINDOWS['standard']
    rolling_cols = [f'num_orders_rolling_mean_{w}' for w in rolling_windows if f'num_orders_rolling_mean_{w}' in df_out.columns]
    
    # Limit polynomial features to the most important rolling windows only
    # This helps reduce collinearity problems
    for col in rolling_cols:
        df_out[f'{col}_sq'] = df_out[col] ** 2
        df_out[f'{col}_sqrt'] = np.sqrt(df_out[col].clip(0))
    
    # Polynomial features for key numeric features (focus on those with high SHAP values)
    # These tend to be less collinear with each other
    high_importance_cols = [
        'checkout_price', 'base_price', 'discount', 'discount_pct', 'price_diff'
    ]
    
    for col in high_importance_cols:
        if col in df_out.columns:
            df_out[f'{col}_sq'] = df_out[col] ** 2
    
    # Polynomial features for lag variables (most important lag features)
    for lag in [1, 2, 5]:  # Limited selection based on SHAP importance
        lag_col = f'num_orders_lag_{lag}'
        if lag_col in df_out.columns:
            df_out[f'{lag_col}_sq'] = df_out[lag_col] ** 2
    
    return df_out

def create_interaction_features(df):
    """Creates feature interactions based on domain knowledge and SHAP analysis."""
    df_out = df.copy()
    
    # Define interaction groups as tuples of features
    interactions = {
        # Price and promotional interactions
        "price_diff_x_emailer": ("price_diff", "emailer_for_promotion"),
        "lag1_x_emailer": ("num_orders_lag_1", "emailer_for_promotion"),
        "price_diff_x_home": ("price_diff", "homepage_featured"),
        "lag1_x_home": ("num_orders_lag_1", "homepage_featured"),
        
        # Rolling mean interactions with promotions
        "rolling_mean_2_x_emailer": ("num_orders_rolling_mean_2", "emailer_for_promotion"),
        "rolling_mean_2_x_home": ("num_orders_rolling_mean_2", "homepage_featured"),
        "rolling_mean_3_x_emailer": ("num_orders_rolling_mean_3", "emailer_for_promotion"),
        "rolling_mean_5_x_emailer": ("num_orders_rolling_mean_5", "emailer_for_promotion"),
        
        # Meal/center aggregates interactions
        "meal_mean_x_discount": ("meal_orders_mean", "discount"),
        "center_mean_x_discount": ("center_orders_mean", "discount"),
        "discount_pct_x_center_mean": ("discount_pct", "center_orders_mean"),
        "base_price_x_homepage": ("base_price", "homepage_featured"),
        
        # Lag and rolling interactions
        "lag1_x_rolling_mean_2": ("num_orders_lag_1", "num_orders_rolling_mean_2"),
        "lag1_x_rolling_mean_3": ("num_orders_lag_1", "num_orders_rolling_mean_3"),
        "rolling_mean_2_x_rolling_mean_3": ("num_orders_rolling_mean_2", "num_orders_rolling_mean_3"),
        "lag1_x_lag2": ("num_orders_lag_1", "num_orders_lag_2"),
        "lag1_x_lag3": ("num_orders_lag_1", "num_orders_lag_3"),
        "lag2_x_lag3": ("num_orders_lag_2", "num_orders_lag_3"),
        "lag2_x_rolling_mean_2": ("num_orders_lag_2", "num_orders_rolling_mean_2"),
        "lag2_x_rolling_mean_3": ("num_orders_lag_2", "num_orders_rolling_mean_3"),
        
        # Seasonality interactions
        "lag1_x_weekofyear_sin": ("num_orders_lag_1", "weekofyear_sin"),
        "lag1_x_month_sin": ("num_orders_lag_1", "month_sin"),
        "mean_by_weekofyear_x_checkout": ("mean_orders_by_weekofyear", "checkout_price"),
        "mean_by_month_x_checkout": ("mean_orders_by_month", "checkout_price"),
        "mean_by_month_x_discount": ("mean_orders_by_month", "discount"),
        
        # Price based interactions
        "base_price_x_discount_pct": ("base_price", "discount_pct"),
    }
    
    # Triple interactions (separate dictionary for clarity)
    triple_interactions = {
        "checkout_x_homepage_x_discount": ("checkout_price", "homepage_featured", "discount"),
        "checkout_x_homepage_x_month_mean": ("checkout_price", "homepage_featured", "mean_orders_by_month"),
        "center_mean_x_meal_mean_x_discount": ("center_orders_mean", "meal_orders_mean", "discount"),
        "rolling_mean_2_x_rolling_mean_3_x_emailer": ("num_orders_rolling_mean_2", "num_orders_rolling_mean_3", "emailer_for_promotion"),
    }
    
    # Add all two-feature interactions
    for name, (feat1, feat2) in interactions.items():
        if feat1 in df_out.columns and feat2 in df_out.columns:
            df_out[name] = df_out[feat1] * df_out[feat2]
        else:
            logging.debug(f"Skipping interaction '{name}' because base feature(s) missing.")
            df_out[name] = 0  # Add column with default value if base features missing
    
    # Add all three-feature interactions
    for name, (feat1, feat2, feat3) in triple_interactions.items():
        if feat1 in df_out.columns and feat2 in df_out.columns and feat3 in df_out.columns:
            df_out[name] = df_out[feat1] * df_out[feat2] * df_out[feat3]
        else:
            logging.debug(f"Skipping triple interaction '{name}' because base feature(s) missing.")
            df_out[name] = 0
    
    # Add specialized polynomial interactions
    if all(c in df_out.columns for c in ['base_price', 'discount_pct']):
        df_out['base_price_poly2_discount_pct'] = df_out['base_price'] * (df_out['discount_pct'] ** 2)
    
    if all(c in df_out.columns for c in ['homepage_featured', 'discount']):
        df_out['homepage_featured_poly2_discount'] = df_out['homepage_featured'] * (df_out['discount'] ** 2)
    
    if all(c in df_out.columns for c in ['mean_orders_by_weekofyear', 'checkout_price']):
        df_out['seasonal_week_x_price'] = df_out['mean_orders_by_weekofyear'] * df_out['checkout_price']
    
    if all(c in df_out.columns for c in ['center_orders_mean', 'meal_orders_mean']):
        df_out['center_orders_mean_poly2_meal_orders_mean'] = (
            df_out['center_orders_mean'] * (df_out['meal_orders_mean'] ** 2)
        )
        df_out['meal_orders_mean_poly2_center_orders_mean'] = (
            df_out['meal_orders_mean'] * (df_out['center_orders_mean'] ** 2)
        )
    
    return df_out

def apply_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None):
    """Applies all feature engineering steps consistently."""
    df_out = df.copy()
    
    # Apply basic feature transformations
    df_out = create_temporal_features(df_out)
    df_out = create_basic_features(df_out)
    
    # Apply lag features only if training or if num_orders exists
    if is_train or TARGET in df_out.columns:
        df_out = create_lag_rolling_features(df_out)
    
    # Apply group aggregations if the target column exists
    if TARGET in df_out.columns:
        df_out = create_group_aggregates(df_out)
    
    # Add seasonality features
    df_out, new_weekofyear_means, new_month_means = add_seasonality_features(
        df_out, weekofyear_means, month_means, is_train
    )
    
    # Add binary column rolling features
    df_out = add_rolling_features_for_binary_cols(df_out)
    
    # Add polynomial and interaction features
    df_out = create_polynomial_features(df_out)
    df_out = create_interaction_features(df_out)
    
    # Return the transformed dataframe and seasonality means
    if is_train:
        return df_out, new_weekofyear_means, new_month_means
    else:
        return df_out

# --- Feature Selection Function ---
def get_feature_list(df):
    """
    Generate a feature list based on dataframe columns, prioritizing features
    that are less likely to cause collinearity issues.
    """
    feature_groups = {
        'base': [
            "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion",
            "discount", "discount_pct", "price_diff", "weekofyear", "price_ratio"
        ],
        'temporal': [
            "weekofyear_sin", "weekofyear_cos", "month_sin", "month_cos",
            "weekofyear_centered_sq", "month_centered_sq"
        ],
        'seasonality': [
            "mean_orders_by_weekofyear", "mean_orders_by_month"
        ],
        'lag_pattern': [f"{TARGET}_lag_{lag}" for lag in LAG_WEEKS],
        # Use only selected windows with higher SHAP importance
        'rolling_mean': [
            f"{TARGET}_rolling_mean_{w}" for w in ROLLING_WINDOWS['standard']
        ],
        'rolling_std': [
            f"{TARGET}_rolling_std_{w}" for w in ROLLING_WINDOWS['stats']
        ],
        # Limit binary features to prevent collinearity
        'binary_rolling': [
            f"{col}_rolling_mean_{w}" 
            for col in PROMO_COLUMNS 
            for w in ROLLING_WINDOWS['binary']
        ],
        'promo_rolling_sums': [
            f"{col}_rolling_sum_{w}" 
            for col in PROMO_COLUMNS 
            for w in ROLLING_WINDOWS['sums']
        ],
        'ewm_features': [
            f"{col}_ewm_alpha_{alpha}" 
            for col in PROMO_COLUMNS 
            for alpha in EWM_ALPHAS
        ],
        'combined_promo': ["emailer_homepage_combined"] + [
            f"emailer_homepage_combined_rolling_mean_{w}" for w in ROLLING_WINDOWS['promo_combine']
        ],
        # Prioritize most important aggregations
        'aggregations': [
            "center_orders_mean", "meal_orders_mean", "center_orders_std", 
            "meal_orders_std", "center_orders_median", "meal_orders_median"
        ],
        'center_meal_interactions': [
            "center_meal_orders_mean_prod", "center_meal_orders_mean_div",
            "center_meal_ratio", "center_meal_weighted"
        ],
        'one_hot': [
            col for col in df.columns if any(col.startswith(prefix) for prefix in [
                "category_", "cuisine_", "center_type_"
            ])
        ]
    }
    
    # Key interaction features based on SHAP values
    high_value_interactions = [
        "lag1_x_rolling_mean_2", "lag1_x_rolling_mean_3",
        "price_diff_x_emailer", "price_diff_x_home",
        "lag1_x_emailer", "lag1_x_home",
        "lag1_x_lag2", "rolling_mean_2_x_rolling_mean_3",
        "rolling_mean_2_x_emailer", "rolling_mean_2_x_home",
        "meal_mean_x_discount", "center_mean_x_discount",
        "lag1_x_weekofyear_sin", "lag1_x_month_sin",
        "mean_by_weekofyear_x_checkout", "mean_by_month_x_checkout",
        "mean_by_month_x_discount", "base_price_x_discount_pct"
    ]
    
    # Check which high-value interactions exist in the dataframe
    feature_groups['high_value_interactions'] = [
        feat for feat in high_value_interactions if feat in df.columns
    ]
    
    # Highly important triple interactions
    triple_interactions = [
        "checkout_x_homepage_x_discount",
        "checkout_x_homepage_x_month_mean",
        "center_mean_x_meal_mean_x_discount",
        "rolling_mean_2_x_rolling_mean_3_x_emailer"
    ]
    
    feature_groups['triple_interactions'] = [
        feat for feat in triple_interactions if feat in df.columns
    ]
    
    # Key polynomial features based on SHAP analysis
    key_polynomial_features = [
        "num_orders_lag_1_sq", "rolling_mean_2_sq", "rolling_mean_2_sqrt",
        "checkout_price_sq", "base_price_sq", "discount_pct_sq", "price_diff_sq"
    ]
    
    feature_groups['key_polynomials'] = [
        feat for feat in key_polynomial_features if feat in df.columns
    ]
    
    # Combine all feature groups and filter to only existing columns
    all_features = []
    for group, features in feature_groups.items():
        all_features.extend([f for f in features if f in df.columns])
    
    # Remove any duplicates while preserving order
    all_features = list(dict.fromkeys(all_features))
    
    # Filter out target or ID columns
    all_features = [f for f in all_features if f != TARGET and f != 'id']
    
    return all_features

# --- Model Utility Functions ---
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)  # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False  # lower is better

def get_lgbm_params(trial=None):
    """Get LightGBM parameters, either default or from Optuna trial."""
    params = {
        'objective': 'regression_l1',  # MAE objective often works well for RMSLE
        'metric': 'None',  # Use custom metric
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
        'min_data_in_leaf': 20,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    if trial:
        # Override with optimized parameters
        params.update({
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 512),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1000.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1000.0, log=True),
        })
    
    return params

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

def run_optuna_trials(train_df, valid_df, features):
    """
    Run Optuna trials to find optimal hyperparameters.
    Returns the best parameters found.
    """
    def objective(trial):
        """Optuna objective function."""
        params = get_lgbm_params(trial)
        
        model = LGBMRegressor(**params)
        model.fit(
            train_df[features], train_df[TARGET],
            eval_set=[
                (train_df[features], train_df[TARGET]),
                (valid_df[features], valid_df[TARGET])
            ],
            eval_metric=lgb_rmsle,
            callbacks=[
                optuna.integration.LightGBMPruningCallback(trial, 'rmsle'),
                early_stopping_with_overfit(stopping_rounds=200, overfit_rounds=15, verbose=False)
            ]
        )
        preds = model.predict(valid_df[features])
        score = rmsle(valid_df[TARGET], preds)
        return score

    # Create or load Optuna study
    try:
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
        logging.info(f"Loaded existing Optuna study from {OPTUNA_DB}")
    except Exception:
        study = optuna.create_study(
            direction="minimize", 
            study_name=OPTUNA_STUDY_NAME, 
            storage=OPTUNA_DB,
            sampler=optuna.samplers.TPESampler(constant_liar=True)
        )
        logging.info(f"Created new Optuna study at {OPTUNA_DB}")
    
    # Run optimization
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=1800)  # 30 minute timeout
    
    best_params = study.best_params
    logging.info(f"Best Optuna params: {best_params}")
    logging.info(f"Best validation RMSLE: {study.best_value:.5f}")
    
    return best_params

def train_final_model(train_df, best_params):
    """
    Train the final model using the best parameters found by Optuna.
    Returns the trained model, validation data, and validation predictions.
    """
    # Get feature list
    features = get_feature_list(train_df)
    
    # Prepare final model parameters
    final_params = get_lgbm_params()  # Get default params
    final_params.update(best_params)  # Override with best params
    final_params['n_estimators'] = 3000  # Increase slightly for final training
    
    # Create a time-based validation split for final training
    train_weeks = sorted(train_df['week'].unique())
    n_val_weeks = max(3, int(0.15 * len(train_weeks)))
    val_weeks = train_weeks[-n_val_weeks:]
    train_weeks = train_weeks[:-n_val_weeks]
    
    train_final = train_df[train_df['week'].isin(train_weeks)].copy()
    val_final = train_df[train_df['week'].isin(val_weeks)].copy()
    
    logging.info(f"Final training: {len(train_final)} samples, validation: {len(val_final)} samples")
    
    # Train with early stopping
    final_model = LGBMRegressor(**final_params)
    final_model.fit(
        train_final[features], train_final[TARGET], 
        eval_set=[
            (train_final[features], train_final[TARGET]),
            (val_final[features], val_final[TARGET])
        ],
        eval_metric=lgb_rmsle,
        callbacks=[early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=15, verbose=True)]
    )
    
    # Evaluate on validation
    val_preds = final_model.predict(val_final[features])
    val_preds = np.clip(val_preds, 0, None)
    val_rmsle_score = rmsle(val_final[TARGET], val_preds)
    logging.info(f"Final model validation RMSLE: {val_rmsle_score:.5f}")
    
    return final_model, val_final, val_preds, features

def generate_predictions(final_model, train_df, test_df, features, weekofyear_means, month_means):
    """
    Generate recursive predictions for the test set with error correction.
    Returns a DataFrame with the predictions.
    """
    logging.info("Starting recursive prediction with error correction on the test set...")
    
    # Prepare for recursive prediction
    history_df = pd.concat([train_df, test_df], ignore_index=True)
    history_df = history_df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    
    # Create a validation set from the most recent training data to calibrate our predictions
    train_weeks = sorted(train_df['week'].unique())
    validation_weeks = train_weeks[-VALIDATION_WEEKS:]
    validation_df = train_df[train_df['week'].isin(validation_weeks)].copy()
    
    # Generate validation predictions to measure error patterns
    validation_features = validation_df[features]
    validation_preds = final_model.predict(validation_features)
    validation_true = validation_df[TARGET].values
      # Calculate errors and error statistics
    val_errors, error_stats = calculate_prediction_errors(validation_true, validation_preds)
    logging.info(f"Validation error statistics: {error_stats}")
    
    # Find optimal scaling factor to minimize RMSLE
    rmsle_scaling_factor = optimize_rmsle_correction(validation_true, validation_preds)
    
    # Build a simple error correction model to predict systematic errors
    error_model, error_model_features = build_error_correction_model(
        validation_df, features, validation_true, validation_preds
    )
      # Create validation error correction report
    validation_report = create_error_correction_report(validation_preds, validation_preds * rmsle_scaling_factor, validation_true)
    logging.info(f"Validation error correction report: {validation_report}")
      # Tracking variables for error correction
    error_decay_rate = 0.8  # How quickly to decay error correction (0-1)
    cumulative_error = 0    # Track cumulative error for monitoring
    weekly_correction_stats = []  # Track correction statistics by week
    
    # Track the number of weeks we've predicted for error correction adjustment
    weeks_predicted = 0
    
    # Predict week by week with error correction
    test_weeks = sorted(test_df['week'].unique())
    for week_num in test_weeks:
        logging.info(f"Predicting for week {week_num}...")
        
        # Identify rows for current week
        current_week_mask = history_df['week'] == week_num
        
        try:
            # Re-apply feature engineering with latest predictions
            history_df = apply_feature_engineering(
                history_df, 
                is_train=False,
                weekofyear_means=weekofyear_means,
                month_means=month_means
            )
            
            # Get features for current week
            current_features = history_df.loc[current_week_mask, features].copy()
            
            # Handle missing columns
            missing_cols = [col for col in features if col not in current_features.columns]
            if missing_cols:
                logging.warning(f"Missing {len(missing_cols)} columns during prediction. Filling with 0.")
                for col in missing_cols:
                    current_features[col] = 0
            
            # Ensure correct feature order
            current_features = current_features[features]
            
            # Get original predictions from the model
            base_preds = final_model.predict(current_features)
            
            # Get correction factors based on seasonality
            correction_factors = get_error_correction_factors(
                history_df, week_num, weekofyear_means, month_means
            )
            
            # Apply error model correction if available
            error_corrections = np.zeros_like(base_preds)
            if error_model is not None and all(f in current_features.columns for f in error_model_features):
                try:
                    # Predict errors based on current features
                    error_corrections = error_model.predict(current_features[error_model_features])
                    # Decay error corrections over time to prevent overcorrection
                    error_corrections *= error_decay_rate ** weeks_predicted
                except Exception as e:
                    logging.warning(f"Error using error model: {e}. Using fallback correction.")
              # Apply mean error correction (with decay over time)
            mean_correction = error_stats['mean'] * (error_decay_rate ** weeks_predicted)
            
            # Apply a blend of different correction methods
            corrected_preds = base_preds.copy()
            
            # First apply the optimal RMSLE scaling factor
            corrected_preds *= rmsle_scaling_factor
            
            # Error model correction weight (only if model exists)
            if error_model is not None:
                corrected_preds += error_corrections * 0.3  # 30% weight to model-based correction
            
            # Apply mean error correction with lower weight
            corrected_preds += mean_correction * 0.2  # 20% weight to mean correction
            
            # For later weeks, revert more to a blend with seasonal factors
            seasonal_blend_weight = min(0.4, 0.1 * weeks_predicted)  # Gradually increase up to 40%
            if 'similar_weekofyear_mean' in correction_factors and 'similar_month_mean' in correction_factors:
                # Create a seasonal baseline prediction
                seasonality_baseline = (
                    correction_factors['similar_weekofyear_mean'] * 0.6 + 
                    correction_factors['similar_month_mean'] * 0.4
                )
                # Blend with seasonal factors for later weeks
                corrected_preds = (
                    (1 - seasonal_blend_weight) * corrected_preds + 
                    seasonal_blend_weight * seasonality_baseline
                )
            
            # Ensure predictions are non-negative and rounded
            corrected_preds = np.clip(corrected_preds, 0, None).round().astype(float)
              # Log correction statistics
            avg_base = np.mean(base_preds)
            avg_corrected = np.mean(corrected_preds)
            correction_diff = avg_corrected - avg_base
            logging.info(
                f"Week {week_num} corrections: " +
                f"Base avg={avg_base:.2f}, Corrected avg={avg_corrected:.2f}, " +
                f"Diff={correction_diff:.2f} ({100*correction_diff/max(1,avg_base):.1f}%)"
            )
            
            # Save weekly correction stats
            weekly_correction_stats.append({
                'week': week_num,
                'base_mean': avg_base,
                'corrected_mean': avg_corrected,
                'abs_correction': np.mean(np.abs(corrected_preds - base_preds)),
                'rel_correction_pct': 100*correction_diff/max(1,avg_base),
                'n_samples': len(base_preds)
            })
            
            # Update predictions in history_df for next iteration
            history_df.loc[current_week_mask, TARGET] = corrected_preds
            
            # Update tracking variables
            weeks_predicted += 1
            
        except Exception as e:
            logging.error(f"Error during prediction for week {week_num}: {e}")
            # If prediction fails for a week, use last week's mean or fallback to base predictions
            try:
                # Try to get predictions from model at minimum
                base_preds = final_model.predict(current_features)
                history_df.loc[current_week_mask, TARGET] = np.clip(base_preds, 0, None).round()
            except:
                # Last resort - set to last known mean
                last_week_mean = history_df[history_df['week'] < week_num][TARGET].mean()
                if pd.isna(last_week_mean):
                    last_week_mean = 0
                history_df.loc[current_week_mask, TARGET] = last_week_mean    # Create submission file
    final_predictions_df = history_df.loc[history_df['id'].isin(test_df['id']), ['id', TARGET]].copy()
    final_predictions_df[TARGET] = final_predictions_df[TARGET].round().astype(int)
    final_predictions_df['id'] = final_predictions_df['id'].astype(int)
    
    # Save weekly correction statistics
    if weekly_correction_stats:
        correction_stats_df = pd.DataFrame(weekly_correction_stats)
        correction_stats_df.to_csv(f"{SUBMISSION_FILE_PREFIX}_weekly_correction_stats.csv", index=False)
        logging.info(f"Weekly correction statistics saved to {SUBMISSION_FILE_PREFIX}_weekly_correction_stats.csv")
        
        # Create a plot of weekly corrections
        try:
            plt.figure(figsize=(12, 6))
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
            plt.figure(figsize=(12, 6))
            plt.bar(correction_stats_df['week'], correction_stats_df['rel_correction_pct'])
            plt.xlabel('Week')
            plt.ylabel('Correction (%)')
            plt.title('Relative Correction Percentage by Week')
            plt.grid(True)
            plt.savefig(f"{SUBMISSION_FILE_PREFIX}_correction_percentage.png")
            plt.close()
            
            logging.info("Weekly correction plots created")
        except Exception as e:
            logging.error(f"Error creating correction plots: {e}")
    
    return final_predictions_df

def generate_ensemble_predictions(final_model, train_df, test_df, features, weekofyear_means, month_means):
    """
    Generate ensemble predictions with multiple error correction strategies.
    This improves robustness by combining different approaches.
    Returns a DataFrame with the ensemble predictions.
    """
    logging.info("Generating ensemble predictions with multiple error correction strategies...")
    
    # Strategy 1: Standard predictions with error correction
    predictions_df1 = generate_predictions(final_model, train_df, test_df, features, weekofyear_means, month_means)
    
    # Strategy 2: Generate predictions with a more conservative error correction
    # Create a modified version of the model with more regularization
    conservative_params = final_model.get_params()
    conservative_params['lambda_l1'] = conservative_params.get('lambda_l1', 0.1) * 2
    conservative_params['lambda_l2'] = conservative_params.get('lambda_l2', 0.1) * 2
    conservative_model = LGBMRegressor(**conservative_params)
    
    # Transfer learned model parameters
    conservative_model._Booster = final_model._Booster
    
    # Backup the original logging level temporarily
    original_log_level = logging.getLogger().level
    
    # Reduce logging output for secondary strategies
    logging.getLogger().setLevel(logging.WARNING)
    
    # Generate predictions with the conservative model
    predictions_df2 = generate_predictions(conservative_model, train_df, test_df, features, weekofyear_means, month_means)
    
    # Strategy 3: Seasonality-focused predictions
    # For this approach, we'll create a version that emphasizes seasonality patterns
    # by modifying the feature set
    seasonality_features = [
        f for f in features if any(term in f for term in [
            'weekofyear', 'month', 'mean_orders_by', '_sin', '_cos',
            'centered_sq'
        ])
    ]
    
    # Include essential features regardless of seasonality
    essential_features = [
        f for f in features if any(term in f for term in [
            'checkout_price', 'base_price', 'discount', 
            'center_orders_mean', 'meal_orders_mean',
            'emailer_for_promotion', 'homepage_featured'
        ])
    ]
    
    # Combine seasonal and essential features
    seasonality_focus_features = list(set(seasonality_features + essential_features))
    
    # Skip this strategy if we don't have enough features
    seasonality_predictions = None
    if len(seasonality_focus_features) >= len(features) * 0.5:
        try:
            # Create model with seasonality focus
            seasonality_model = LGBMRegressor(**final_model.get_params())
            seasonality_model._Booster = final_model._Booster
            
            # Generate seasonality-focused predictions
            predictions_df3 = generate_predictions(
                seasonality_model, train_df, test_df, 
                seasonality_focus_features, weekofyear_means, month_means
            )
            seasonality_predictions = dict(zip(predictions_df3['id'], predictions_df3[TARGET]))
        except Exception as e:
            logging.warning(f"Seasonality strategy failed: {e}")
            seasonality_predictions = None
    
    # Strategy 4: Lag-weighted approach
    # This strategy focuses more on recent time patterns
    lag_features = [
        f for f in features if any(term in f for term in [
            'lag_', 'rolling_mean', 'rolling_std'
        ])
    ]
    
    # Include essential features
    lag_focus_features = list(set(lag_features + essential_features))
    
    # Skip this strategy if we don't have enough features
    lag_predictions = None
    if len(lag_focus_features) >= len(features) * 0.5:
        try:
            # Create model with lag focus
            lag_model = LGBMRegressor(**final_model.get_params())
            lag_model._Booster = final_model._Booster
            
            # Generate lag-focused predictions
            predictions_df4 = generate_predictions(
                lag_model, train_df, test_df, 
                lag_focus_features, weekofyear_means, month_means
            )
            lag_predictions = dict(zip(predictions_df4['id'], predictions_df4[TARGET]))
        except Exception as e:
            logging.warning(f"Lag strategy failed: {e}")
            lag_predictions = None
    
    # Restore original logging level
    logging.getLogger().setLevel(original_log_level)
    
    # Create a mapping from test ids to their predictions
    id_to_pred1 = dict(zip(predictions_df1['id'], predictions_df1[TARGET]))
    id_to_pred2 = dict(zip(predictions_df2['id'], predictions_df2[TARGET]))
    
    # Create ensemble predictions DataFrame
    ensemble_df = predictions_df1.copy()
    
    # Average the predictions with weights based on available strategies
    ensemble_df[TARGET] = ensemble_df.apply(
        lambda row: calculate_weighted_ensemble_prediction(
            row['id'], id_to_pred1, id_to_pred2, 
            seasonality_predictions, lag_predictions
        ),
        axis=1
    )
    
    logging.info("Ensemble prediction completed")
    return ensemble_df

def calculate_weighted_ensemble_prediction(id_value, strategy1, strategy2, strategy3=None, strategy4=None):
    """
    Calculate weighted ensemble prediction based on available strategies.
    Returns an integer prediction.
    """
    # Initialize weights and values
    weights = [0.4, 0.3, 0.15, 0.15]  # Default weights for 4 strategies
    values = [
        strategy1.get(id_value, 0), 
        strategy2.get(id_value, 0),
        strategy3.get(id_value, 0) if strategy3 is not None else None,
        strategy4.get(id_value, 0) if strategy4 is not None else None
    ]
    
    # Remove None values and adjust weights
    valid_values = [v for v in values if v is not None]
    valid_weights = weights[:len(valid_values)]
    
    # Normalize weights to sum to 1
    valid_weights = [w / sum(valid_weights) for w in valid_weights]
    
    # Calculate weighted average
    weighted_pred = sum(w * v for w, v in zip(valid_weights, valid_values))
    
    # Round to integer
    return int(round(weighted_pred))

def generate_shap_analysis(final_model, train_df, features):
    """Generate SHAP analysis plots and save feature importance."""
    logging.info("Calculating SHAP values...")
    try:
        # Sample data for SHAP to keep computation reasonable
        if len(train_df) > N_SHAP_SAMPLES:
            shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
        else:
            shap_sample = train_df.copy()
            
        # Create SHAP explainer and calculate values
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(shap_sample[features])
        
        # Save SHAP values and importance
        shap_values_df = pd.DataFrame(shap_values, columns=features)
        shap_values_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_values.csv", index=False)
        
        shap_importance_df = pd.DataFrame({
            'feature': features,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        shap_importance_df.to_csv(f"{SHAP_FILE_PREFIX}_optuna_feature_importances.csv", index=False)
        
        # Generate SHAP plots
        logging.info("Generating SHAP plots...")
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, shap_sample[features], show=False)
        plt.tight_layout()
        plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_summary.png")
        plt.close()
        
        # Importance Bar Plot (Top 20)
        plt.figure(figsize=(10, 8))
        shap_importance_df.head(20).plot(kind='barh', x='feature', y='mean_abs_shap', legend=False)
        plt.gca().invert_yaxis()  # Display most important at the top
        plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
        plt.title('Top 20 SHAP Feature Importances (Recursive Optuna Model)')
        plt.tight_layout()
        plt.savefig(f"{SHAP_FILE_PREFIX}_optuna_top20_importance.png")
        plt.close()
        
        logging.info("SHAP analysis saved.")
        
    except Exception as e:
        logging.error(f"Error during SHAP analysis: {e}")

def create_validation_plot(valid_df, val_preds):
    """Create and save validation plot."""
    logging.info("Generating validation plot...")
    try:
        plt.figure(figsize=(15, 6))
        plt.scatter(valid_df[TARGET], val_preds, alpha=0.5, s=10)
        plt.plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
                 [valid_df[TARGET].min(), valid_df[TARGET].max()], 
                 'r--', lw=2, label='Ideal')
        plt.xlabel("Actual Orders (Validation Set)")
        plt.ylabel("Predicted Orders (Validation Set)")
        plt.title(f"Actual vs. Predicted Orders - RMSLE: {rmsle(valid_df[TARGET], val_preds):.4f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("validation_actual_vs_predicted.png")
        plt.close()
        logging.info("Validation plot saved.")
        
    except Exception as e:
        logging.error(f"Error during plotting: {e}")

def calculate_prediction_errors(true_values, predicted_values):
    """
    Calculate prediction errors from validation data.
    Returns error statistics that can be used for error correction.
    """
    errors = true_values - predicted_values
    
    # Compute error statistics
    error_stats = {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'mae': np.mean(np.abs(errors)),
        'mape': np.mean(np.abs(errors / true_values.replace(0, 1))) * 100
    }
    
    return errors, error_stats

def build_error_correction_model(train_df, features, y_true, y_pred):
    """
    Build a simple model to predict errors based on features.
    This can help correct systematic biases in the predictions.
    """
    # Calculate errors
    errors = y_true - y_pred
    
    # Select a subset of important features (to prevent overfitting)
    # Prioritize time features and raw values that may correlate with errors
    error_model_features = [
        f for f in features if any(term in f for term in [
            'lag_', 'rolling_mean', 'weekofyear', 'month', 'mean_orders_by', 
            'checkout_price', 'center_', 'meal_', 'discount'
        ])
    ][:20]  # Limit to most relevant features
    
    if len(error_model_features) == 0:
        return None, []
        
    # Create a simple linear model to predict errors
    error_model = LinearRegression()
    
    try:
        error_model.fit(train_df[error_model_features], errors)
        return error_model, error_model_features
    except Exception as e:
        logging.warning(f"Error creating error correction model: {e}")
        return None, []

def get_error_correction_factors(history_df, predicted_week, weekofyear_means, month_means):
    """
    Calculate correction factors based on seasonality and past errors.
    These factors can be used to adjust future predictions.
    """
    # Get current week/month
    current_week_data = history_df[history_df['week'] == predicted_week]
    if len(current_week_data) == 0:
        return {}
        
    weekofyear = current_week_data['weekofyear'].values[0]
    month = current_week_data['month'].values[0]
    
    # Identify previous weeks with similar seasonality
    similar_weekofyear_mask = (
        (history_df['weekofyear'] == weekofyear) & 
        (history_df['week'] < predicted_week) &
        (~history_df[TARGET].isna())
    )
    
    similar_month_mask = (
        (history_df['month'] == month) & 
        (history_df['week'] < predicted_week) &
        (~history_df[TARGET].isna())
    )
    
    # Get average error rate from similar seasonal periods
    correction_factors = {}
    
    if weekofyear_means is not None and weekofyear in weekofyear_means.index:
        correction_factors['weekofyear_mean'] = weekofyear_means[weekofyear]
    else:
        correction_factors['weekofyear_mean'] = 0
        
    if month_means is not None and month in month_means.index:
        correction_factors['month_mean'] = month_means[month]
    else:
        correction_factors['month_mean'] = 0
    
    # Get statistics from similar weeks of year (if available)
    similar_weekofyear_data = history_df[similar_weekofyear_mask]
    if len(similar_weekofyear_data) > 0:
        correction_factors['similar_weekofyear_mean'] = similar_weekofyear_data[TARGET].mean()
    else:
        correction_factors['similar_weekofyear_mean'] = correction_factors['weekofyear_mean']
        
    # Get statistics from similar months (if available)
    similar_month_data = history_df[similar_month_mask]
    if len(similar_month_data) > 0:
        correction_factors['similar_month_mean'] = similar_month_data[TARGET].mean()
    else:
        correction_factors['similar_month_mean'] = correction_factors['month_mean']
        
    return correction_factors

def optimize_rmsle_correction(true_values, predicted_values):
    """
    Find an optimal scaling factor to minimize RMSLE.
    This helps correct systematic biases in the predictions.
    """
    def rmsle_with_scaling(scaling_factor):
        scaled_preds = predicted_values * scaling_factor
        return rmsle(true_values, scaled_preds)
    
    # Try a range of scaling factors to find the best one
    best_scaling = 1.0
    best_score = rmsle(true_values, predicted_values)
    
    # Grid search for optimal scaling factor
    for scaling in np.linspace(0.8, 1.2, 41):  # Test scale factors from 0.8 to 1.2 in steps of 0.01
        score = rmsle_with_scaling(scaling)
        if score < best_score:
            best_score = score;
            best_scaling = scaling
    
    logging.info(f"Optimal RMSLE scaling factor: {best_scaling:.4f} improves score from {rmsle(true_values, predicted_values):.4f} to {best_score:.4f}")
    return best_scaling

# --- Main Execution Flow ---
def main():
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
    df = preprocess_data(df, meal_info, center_info)
    test = preprocess_data(test, meal_info, center_info)
    
    # Add placeholder for num_orders in test for alignment
    if TARGET not in test.columns:
        test[TARGET] = np.nan
    
    # --- One-hot encoding for categorical features ---
    logging.info("Applying one-hot encoding...")
    df_full = pd.concat([df, test], ignore_index=True)
    
    # Add basic features before one-hot encoding
    df_full = create_basic_features(df_full)
    df_full = create_temporal_features(df_full)
    
    # Apply one-hot encoding to categorical features
    cat_cols = [col for col in ["category", "cuisine", "center_type"] if col in df_full.columns]
    if cat_cols:
        df_full = pd.get_dummies(df_full, columns=cat_cols, dummy_na=False)
    
    # Split back to train and test
    train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
    test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()
    
    # --- Apply Feature Engineering ---
    logging.info("Applying feature engineering...")
    train_df, weekofyear_means, month_means = apply_feature_engineering(train_df, is_train=True)
    test_df = apply_feature_engineering(
        test_df, 
        is_train=False, 
        weekofyear_means=weekofyear_means, 
        month_means=month_means
    )
    
    # Clean up any rows with missing target values
    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    
    # --- Generate Feature List ---
    features = get_feature_list(train_df)
    logging.info(f"Using {len(features)} features")
    
    # --- Create Train/Validation Split ---
    max_week = train_df["week"].max()
    valid_df = train_df[train_df["week"] > max_week - VALIDATION_WEEKS].copy()
    train_split_df = train_df[train_df["week"] <= max_week - VALIDATION_WEEKS].copy()
    
    logging.info(f"Train split shape: {train_split_df.shape}, Validation shape: {valid_df.shape}")
    
    # --- Optuna Hyperparameter Tuning ---
    best_params = run_optuna_trials(train_split_df, valid_df, features)    # --- Train Final Model ---
    final_model, val_final, val_preds, features = train_final_model(train_df, best_params)
    
    # --- Evaluate Error Correction Strategies ---
    error_correction_report = evaluate_error_correction(
        final_model, val_final, features, weekofyear_means, month_means
    )
    
    # Log the best error correction strategy
    if not error_correction_report.empty:
        best_strategy = error_correction_report.iloc[0]['strategy']
        best_improvement = error_correction_report.iloc[0]['pct_improvement']
        logging.info(f"Best error correction strategy: {best_strategy} with {best_improvement:.2f}% improvement")
    
    # --- Generate Predictions ---
    # Standard predictions with error correction
    final_predictions_df = generate_predictions(
        final_model, train_df, test_df, features, weekofyear_means, month_means
    )
    
    # Save standard submission
    standard_submission_path = f"{SUBMISSION_FILE_PREFIX}_ec_standard.csv"
    final_predictions_df.to_csv(standard_submission_path, index=False)
    logging.info(f"Standard error-corrected submission saved to {standard_submission_path}")
    
    # --- Generate Ensemble Predictions with Multiple Error Correction Strategies ---
    ensemble_predictions_df = generate_ensemble_predictions(
        final_model, train_df, test_df, features, weekofyear_means, month_means
    )
    
    # Save ensemble submission
    ensemble_submission_path = f"{SUBMISSION_FILE_PREFIX}_ec_ensemble.csv"
    ensemble_predictions_df.to_csv(ensemble_submission_path, index=False)
    logging.info(f"Ensemble submission saved to {ensemble_submission_path}")
    
    # --- SHAP Analysis ---
    generate_shap_analysis(final_model, train_df, features)
    
    # --- Validation Plot ---
    create_validation_plot(val_final, val_preds)
    
    # --- Evaluate Error Correction Strategies ---
    evaluate_error_correction(final_model, val_final, features, weekofyear_means, month_means)
    
    logging.info("Script finished.")

if __name__ == "__main__":
    main()

def create_error_correction_report(original_preds, corrected_preds, true_values=None):
    """
    Create a report on the impact of error correction.
    If true values are provided, also includes accuracy metrics.
    """
    report = {
        'n_samples': len(original_preds),
        'original_mean': np.mean(original_preds),
        'corrected_mean': np.mean(corrected_preds),
        'mean_abs_correction': np.mean(np.abs(corrected_preds - original_preds)),
        'mean_correction': np.mean(corrected_preds - original_preds),
        'correction_percentage': np.mean((corrected_preds - original_preds) / np.maximum(original_preds, 1)) * 100,
        'max_correction': np.max(corrected_preds - original_preds),
        'min_correction': np.min(corrected_preds - original_preds),
    }
    
    # Add accuracy metrics if true values are provided
    if true_values is not None:
        report.update({
            'original_rmsle': rmsle(true_values, original_preds),
            'corrected_rmsle': rmsle(true_values, corrected_preds),
            'rmsle_improvement': rmsle(true_values, original_preds) - rmsle(true_values, corrected_preds),
            'original_mae': np.mean(np.abs(true_values - original_preds)),
            'corrected_mae': np.mean(np.abs(true_values - corrected_preds)),
            'mae_improvement': np.mean(np.abs(true_values - original_preds)) - np.mean(np.abs(true_values - corrected_preds)),
        })
    
    return report

def evaluate_error_correction(final_model, validation_df, features, weekofyear_means, month_means):
    """
    Evaluate different error correction strategies on validation data.
    This helps us understand which strategies are most effective.
    """
    logging.info("Evaluating error correction strategies on validation data...")
    
    # Extract validation data
    X_valid = validation_df[features]
    y_valid = validation_df[TARGET].values
    
    # Get raw predictions from the model (no error correction)
    raw_preds = final_model.predict(X_valid)
    raw_preds = np.clip(raw_preds, 0, None)
    base_rmsle = rmsle(y_valid, raw_preds)
    logging.info(f"Base model RMSLE (no error correction): {base_rmsle:.5f}")
    
    # Evaluate RMSLE scaling
    scaling_factor = optimize_rmsle_correction(y_valid, raw_preds)
    scaled_preds = raw_preds * scaling_factor
    scaled_rmsle = rmsle(y_valid, scaled_preds)
    logging.info(f"RMSLE with scaling factor ({scaling_factor:.4f}): {scaled_rmsle:.5f} (improvement: {base_rmsle - scaled_rmsle:.5f})")
    
    # Build error correction model
    error_model, error_features = build_error_correction_model(validation_df, features, y_valid, raw_preds)
    
    if error_model is not None and error_features:
        # Apply error model correction
        error_preds = raw_preds + error_model.predict(validation_df[error_features])
        error_preds = np.clip(error_preds, 0, None)
        error_rmsle = rmsle(y_valid, error_preds)
        logging.info(f"RMSLE with error model: {error_rmsle:.5f} (improvement: {base_rmsle - error_rmsle:.5f})")
        
        # Combine scaling and error model
        combined_preds = scaled_preds + 0.5 * error_model.predict(validation_df[error_features])
        combined_preds = np.clip(combined_preds, 0, None)
        combined_rmsle = rmsle(y_valid, combined_preds)
        logging.info(f"RMSLE with combined correction: {combined_rmsle:.5f} (improvement: {base_rmsle - combined_rmsle:.5f})")
    
    # Evaluate seasonality-based correction
    correction_factors = get_error_correction_factors(validation_df, validation_df['week'].max(), weekofyear_means, month_means)
    
    if 'similar_weekofyear_mean' in correction_factors and 'similar_month_mean' in correction_factors:
        # Create seasonal baseline predictions
        seasonality_baseline = (
            correction_factors['similar_weekofyear_mean'] * 0.6 + 
            correction_factors['similar_month_mean'] * 0.4
        )
        
        # Blend with raw predictions (20% seasonal)
        seasonal_blend_preds = 0.8 * raw_preds + 0.2 * seasonality_baseline
        seasonal_blend_preds = np.clip(seasonal_blend_preds, 0, None)
        seasonal_rmsle = rmsle(y_valid, seasonal_blend_preds)
        logging.info(f"RMSLE with 20% seasonal blend: {seasonal_rmsle:.5f} (improvement: {base_rmsle - seasonal_rmsle:.5f})")
      # Create a comprehensive error correction report
    strategies = {
        'base': raw_preds,
        'scaling': scaled_preds
    }
    
    if error_model is not None:
        strategies['error_model'] = error_preds
        strategies['combined'] = combined_preds
    
    if 'similar_weekofyear_mean' in correction_factors:
        strategies['seasonal_blend'] = seasonal_blend_preds
    
    # Create report
    report_data = []
    for name, preds in strategies.items():
        score = rmsle(y_valid, preds)
        improvement = base_rmsle - score
        pct_improvement = 100 * improvement / base_rmsle
        report_data.append({
            'strategy': name,
            'rmsle': score,
            'improvement': improvement,
            'pct_improvement': pct_improvement
        })
    
    report_df = pd.DataFrame(report_data).sort_values('rmsle')
    report_df.to_csv(f"{SUBMISSION_FILE_PREFIX}_error_correction_evaluation.csv", index=False)
    logging.info(f"Error correction evaluation saved to {SUBMISSION_FILE_PREFIX}_error_correction_evaluation.csv")
    
    # Visualize the impact of the best error correction method
    if len(report_df) > 1:
        best_strategy = report_df.iloc[0]['strategy']
        best_preds = strategies[best_strategy]
        visualize_error_correction_impact(validation_df, raw_preds, best_preds)
    
    return report_df

def visualize_error_correction_impact(validation_df, base_preds, corrected_preds):
    """
    Create visualizations showing the impact of error correction on validation data.
    """
    try:
        # Prepare data
        y_true = validation_df[TARGET].values
        
        # Calculate error metrics
        base_errors = y_true - base_preds
        corrected_errors = y_true - corrected_preds
        
        # Plot 1: Error distribution comparison
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.hist(base_errors, bins=50, alpha=0.5, label='Base Errors')
        plt.hist(corrected_errors, bins=50, alpha=0.5, label='Corrected Errors')
        plt.xlabel('Error Value')
        plt.ylabel('Frequency')
        plt.title('Error Distribution: Base vs. Corrected')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Actual vs. Predicted scatter comparison
        plt.subplot(2, 2, 3)
        plt.scatter(y_true, base_preds, alpha=0.5, s=10)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Base Predictions\nRMSLE: {rmsle(y_true, base_preds):.4f}')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.scatter(y_true, corrected_preds, alpha=0.5, s=10)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Corrected Predictions\nRMSLE: {rmsle(y_true, corrected_preds):.4f}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{SUBMISSION_FILE_PREFIX}_error_correction_impact.png")
        plt.close()
        
        # Plot 3: Error by order volume
        plt.figure(figsize=(10, 6))
        
        # Group by order volume bins
        order_bins = np.linspace(0, np.percentile(y_true, 99), 10)
        bin_indices = np.digitize(y_true, order_bins)
        
        base_error_by_bin = [np.mean(np.abs(base_errors[bin_indices == i])) for i in range(1, len(order_bins) + 1)]
        corrected_error_by_bin = [np.mean(np.abs(corrected_errors[bin_indices == i])) for i in range(1, len(order_bins) + 1)]
        bin_centers = [(order_bins[i] + order_bins[i-1])/2 for i in range(1, len(order_bins))]
        
        plt.bar(bin_centers, base_error_by_bin, width=order_bins[1]-order_bins[0], alpha=0.5, label='Base Errors')
        plt.bar(bin_centers, corrected_error_by_bin, width=order_bins[1]-order_bins[0], alpha=0.5, label='Corrected Errors')
        plt.xlabel('Order Volume')
        plt.ylabel('Mean Absolute Error')
        plt.title('Error by Order Volume')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{SUBMISSION_FILE_PREFIX}_error_by_volume.png")
        plt.close()
        
        logging.info("Error correction impact visualizations created")
        
    except Exception as e:
        logging.error(f"Error creating error correction visualizations: {e}")
        import traceback
        traceback.print_exc()

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
# Advanced ensemble and meta-learning imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
# SEED = 42
SEED = random.randint(0, 1000) # Random seed for reproducibility
LAG_WEEKS = [1, 2, 3, 5, 10] # Lags based on num_orders
ROLLING_WINDOWS = [2, 3, 5, 10, 14, 21] # Added 14 and 21
# Advanced lag features with exponential decay weights
EXP_DECAY_LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] # More granular lags
DECAY_FACTOR = 0.85 # Exponential decay factor for weighted features
# Other features (not directly dependent on recursive prediction)
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
OPTUNA_TRIALS = 1 # Number of Optuna trials
OPTUNA_STUDY_NAME = "experimental"
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "experimental_submission"
SHAP_FILE_PREFIX = "shap_experimental"
N_SHAP_SAMPLES = 2000
# Advanced modeling configuration
ENSEMBLE_MODELS = 5  # Number of models in ensemble
USE_TARGET_ENCODING = True  # Enable target encoding
USE_UNCERTAINTY_FEATURES = True  # Enable uncertainty quantification

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

    # Standard Lags
    for lag in lag_weeks:
        df_out[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)

    # Advanced: Exponential decay weighted lags
    if target_col == 'num_orders':  # Only for target variable
        for i, lag in enumerate(EXP_DECAY_LAGS):
            weight = DECAY_FACTOR ** i
            df_out[f"{target_col}_exp_lag_{lag}"] = group[target_col].shift(lag) * weight
        
        # Weighted moving averages with exponential decay
        for window in [3, 5, 7, 10]:
            weights = np.array([DECAY_FACTOR ** i for i in range(window)])
            weights = weights / weights.sum()  # Normalize
            
            # Calculate weighted average manually
            weighted_sum = df_out[f"{target_col}_lag_1"] * 0  # Initialize
            for i, w in enumerate(weights):
                lag_col = f"{target_col}_lag_{i+1}"
                if lag_col in df_out.columns:
                    weighted_sum += df_out[lag_col].fillna(0) * w
            df_out[f"{target_col}_exp_ma_{window}"] = weighted_sum    # Rolling features (use shift(1) to avoid data leakage)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)
        
        # Advanced rolling statistics
        df_out[f"{target_col}_rolling_median_{window}"] = shifted.rolling(window, min_periods=1).median().reset_index(drop=True)
        
        # Only calculate skew for windows >= 3 (skew needs at least 3 points)
        if window >= 3:
            df_out[f"{target_col}_rolling_skew_{window}"] = shifted.rolling(window, min_periods=min(3, window)).skew().reset_index(drop=True)
        
        # Trend features - slope of linear regression over rolling window (need at least 2 points)
        if window >= 2:
            def rolling_trend(series, window):
                def trend_slope(x):
                    if len(x) < 2:
                        return 0
                    try:
                        slope, _, _, _, _ = stats.linregress(range(len(x)), x)
                        return slope
                    except:
                        return 0
                return series.rolling(window, min_periods=min(2, window)).apply(trend_slope, raw=False)
            
            df_out[f"{target_col}_rolling_trend_{window}"] = rolling_trend(shifted, window)

    return df_out

def create_other_features(df):
    """Creates features not directly dependent on recursive prediction."""
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)

    # Price features
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, 1e-10) # Avoid division by zero
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
    df_out['center_meal_orders_mean_div'] = df_out['center_orders_mean'] / df_out['meal_orders_mean'].replace(0, 1e-10)
    # TODO - Test if this is useful
    # df_out['center_meal_orders_median_div'] = df_out['center_orders_median'] / df_out['meal_orders_median'].replace(0, 1e-10)
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
        df_out['price_ratio'] = df_out['checkout_price'] / df_out['base_price'].replace(0, 1e-10)

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

def create_target_encoding_features(df, is_train=True, encoding_stats=None):
    """
    Creates target encoding features with cross-validation to prevent overfitting.
    Based on advanced guide recommendations for categorical feature enhancement.
    """
    df_out = df.copy()
    
    if not USE_TARGET_ENCODING or 'num_orders' not in df_out.columns:
        return df_out, encoding_stats
    
    categorical_cols = ['center_id', 'meal_id']
    if 'category' in df_out.columns:
        categorical_cols.append('category')
    if 'cuisine' in df_out.columns:
        categorical_cols.append('cuisine')
    
    if is_train:
        encoding_stats = {}
        # Calculate global mean for smoothing
        global_mean = df_out['num_orders'].mean()
        
        for col in categorical_cols:
            # Calculate category means with smoothing
            cat_stats = df_out.groupby(col)['num_orders'].agg(['mean', 'count']).reset_index()
            
            # Bayesian smoothing: blend category mean with global mean based on count
            min_samples_leaf = 10
            smoothing_factor = 1 / (1 + np.exp(-(cat_stats['count'] - min_samples_leaf) / min_samples_leaf))
            cat_stats['smoothed_mean'] = (
                cat_stats['mean'] * smoothing_factor + 
                global_mean * (1 - smoothing_factor)
            )
            
            encoding_stats[col] = dict(zip(cat_stats[col], cat_stats['smoothed_mean']))
            encoding_stats[f'{col}_global_mean'] = global_mean
    
    # Apply encoding
    for col in categorical_cols:
        if col in encoding_stats:
            df_out[f'{col}_target_encoded'] = df_out[col].map(encoding_stats[col]).fillna(
                encoding_stats[f'{col}_global_mean']
            )
    
    return df_out, encoding_stats

def create_uncertainty_features(df, models_dict=None, is_train=True):
    """
    Creates model uncertainty features for meta-learning.
    Based on advanced guide: prediction confidence as features.
    """
    df_out = df.copy()
    
    if not USE_UNCERTAINTY_FEATURES or models_dict is None or 'num_orders' not in df_out.columns:
        return df_out
    
    # Create base features for uncertainty estimation
    base_features = ['checkout_price', 'base_price', 'discount_pct', 'weekofyear']
    base_features = [f for f in base_features if f in df_out.columns]
    
    if len(base_features) < 2:
        return df_out
    
    predictions = []
    for model_name, model in models_dict.items():
        try:
            pred = model.predict(df_out[base_features].fillna(0))
            predictions.append(pred)
            df_out[f'meta_pred_{model_name}'] = pred
        except:
            continue
    
    if len(predictions) >= 2:
        predictions = np.array(predictions)
        # Ensemble disagreement (variance between models)
        df_out['prediction_uncertainty'] = np.var(predictions, axis=0)
        df_out['prediction_mean'] = np.mean(predictions, axis=0)
        df_out['prediction_std'] = np.std(predictions, axis=0)
    
    return df_out

def create_residual_features(df, model=None, is_train=True, residual_stats=None):
    """
    Creates features based on residual analysis to capture systematic patterns.
    Based on advanced guide: learn from prediction errors.
    """
    df_out = df.copy()
    
    if model is None or 'num_orders' not in df_out.columns:
        return df_out, residual_stats
    
    # Create simple features for residual model
    simple_features = ['checkout_price', 'base_price', 'weekofyear']
    simple_features = [f for f in simple_features if f in df_out.columns]
    if len(simple_features) < 2:
        return df_out, residual_stats
    
    try:
        # Get predictions and residuals
        predictions = model.predict(df_out[simple_features].fillna(0))
        if is_train:
            residuals = df_out['num_orders'] - predictions
            
            # Calculate residual patterns by different groups
            residual_stats = {}
            
            # Residual patterns by price quartiles
            try:
                # Create quartiles and store the boundaries for later use
                df_out['price_quartile'] = pd.qcut(df_out['checkout_price'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                # Store quartile boundaries for test data
                quartile_boundaries = pd.qcut(df_out['checkout_price'], q=4, duplicates='drop').cat.categories
                residual_stats['quartile_boundaries'] = quartile_boundaries
                
                if 'price_quartile' in df_out.columns:
                    temp_residuals = pd.Series(residuals, index=df_out.index)
                    price_residual_means = df_out.groupby('price_quartile').apply(
                        lambda x: temp_residuals.iloc[x.index].mean()
                    ).to_dict()
                    residual_stats['price_quartile'] = price_residual_means
            except Exception as e:
                logging.warning(f"Error creating price quartiles: {e}")
            
            # Residual patterns by week of year
            week_residual_means = df_out.groupby('weekofyear').apply(
                lambda x: pd.Series(residuals, index=df_out.index).iloc[x.index].mean()
            ).to_dict()
            residual_stats['weekofyear'] = week_residual_means
        else:
            # For test data, recreate price quartiles using stored boundaries
            if residual_stats and 'quartile_boundaries' in residual_stats:
                try:
                    df_out['price_quartile'] = pd.cut(df_out['checkout_price'], 
                                                    bins=residual_stats['quartile_boundaries'], 
                                                    labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                                    include_lowest=True, duplicates='drop')
                except Exception as e:
                    logging.warning(f"Error applying price quartiles to test data: {e}")
          # Apply residual corrections
        if residual_stats:
            if 'price_quartile' in residual_stats and 'price_quartile' in df_out.columns:
                # Convert categorical to string first to avoid category assignment issues
                price_quartile_str = df_out['price_quartile'].astype(str)
                df_out['residual_correction_price'] = price_quartile_str.map(
                    residual_stats['price_quartile']
                ).fillna(0)
            
            if 'weekofyear' in residual_stats:
                df_out['residual_correction_week'] = df_out['weekofyear'].map(
                    residual_stats['weekofyear']
                ).fillna(0)
                
    except Exception as e:
        logging.warning(f"Error in residual feature creation: {e}")
        if is_train:
            residual_stats = {}
    
    return df_out, residual_stats

def create_volatility_features(df):
    """
    Add features that capture prediction uncertainty patterns.
    These features help the model learn volatility and trend patterns that would
    be better than post-hoc adaptive adjustments.
    """
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS)
    
    # Coefficient of variation in recent orders
    for window in [3, 5, 7]:
        if f'num_orders_rolling_mean_{window}' in df_out.columns and f'num_orders_rolling_std_{window}' in df_out.columns:
            rolling_mean = df_out[f'num_orders_rolling_mean_{window}']
            rolling_std = df_out[f'num_orders_rolling_std_{window}']
            df_out[f'cv_{window}'] = (rolling_std / (rolling_mean + 1e-10)).fillna(0)
    
    # Trend strength using linear regression slope
    for window in [3, 5, 7]:
        if f'num_orders_rolling_trend_{window}' in df_out.columns:
            # Use existing trend features and create strength indicators
            trend_values = df_out[f'num_orders_rolling_trend_{window}'].fillna(0)
            df_out[f'trend_strength_{window}'] = np.abs(trend_values)
            df_out[f'trend_direction_{window}'] = np.sign(trend_values)
    
    # Volatility patterns - standard deviation relative to mean
    for window in [5, 10, 14]:
        if f'num_orders_rolling_std_{window}' in df_out.columns and f'num_orders_rolling_mean_{window}' in df_out.columns:
            std_col = df_out[f'num_orders_rolling_std_{window}']
            mean_col = df_out[f'num_orders_rolling_mean_{window}']
            df_out[f'volatility_ratio_{window}'] = (std_col / (mean_col + 1e-10)).fillna(0)
    
    # Order consistency - how consistent are recent orders
    for window in [3, 5]:
        if f'num_orders_lag_1' in df_out.columns:
            shifted = group['num_orders'].shift(1)
            # Calculate range (max - min) over window
            rolling_max = shifted.rolling(window, min_periods=1).max()
            rolling_min = shifted.rolling(window, min_periods=1).min()
            rolling_mean = shifted.rolling(window, min_periods=1).mean()
            df_out[f'order_range_{window}'] = (rolling_max - rolling_min) / (rolling_mean + 1e-10)
    
    # Recent vs historical comparison
    if 'num_orders_rolling_mean_3' in df_out.columns and 'num_orders_rolling_mean_14' in df_out.columns:
        recent_mean = df_out['num_orders_rolling_mean_3']
        historical_mean = df_out['num_orders_rolling_mean_14']
        df_out['recent_vs_historical'] = (recent_mean / (historical_mean + 1e-10)).fillna(1.0)
    
    return df_out

def apply_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None, 
                            encoding_stats=None, uncertainty_models=None, residual_stats=None, residual_model=None):
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
    
    # Advanced meta-learning features
    df_out, new_encoding_stats = create_target_encoding_features(df_out, is_train=is_train, encoding_stats=encoding_stats)
    if is_train:
        encoding_stats = new_encoding_stats
    
    # Uncertainty features (only if we have trained models)
    if uncertainty_models:
        df_out = create_uncertainty_features(df_out, uncertainty_models, is_train=is_train)
    
    # Residual correction features
    df_out, new_residual_stats = create_residual_features(df_out, residual_model, is_train=is_train, residual_stats=residual_stats)
    if is_train:
        residual_stats = new_residual_stats
        
    # Volatility features
    df_out = create_volatility_features(df_out)

    if is_train:
        return df_out, encoding_stats, residual_stats
    else:
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

# Create simple models for uncertainty estimation
simple_features = ['checkout_price', 'base_price', 'weekofyear']
simple_features = [f for f in simple_features if f in train_df.columns]

uncertainty_models = {}
if USE_UNCERTAINTY_FEATURES and len(simple_features) >= 2:
    logging.info("Training uncertainty estimation models...")
    # Train simple models for uncertainty estimation
    X_simple = train_df[simple_features].fillna(0)
    y_simple = train_df['num_orders']
    
    # Random Forest for uncertainty
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1)
    rf_model.fit(X_simple, y_simple)
    uncertainty_models['rf'] = rf_model
    
    # Extra Trees for uncertainty
    et_model = ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1)
    et_model.fit(X_simple, y_simple)
    uncertainty_models['et'] = et_model

# Train a simple model for residual analysis
residual_model = None
if len(simple_features) >= 2:
    residual_model = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=SEED, n_jobs=-1)
    residual_model.fit(train_df[simple_features].fillna(0), train_df['num_orders'])

# First apply feature engineering to train to get seasonality means
train_df, encoding_stats, residual_stats = apply_feature_engineering(
    train_df, is_train=True, uncertainty_models=uncertainty_models, residual_model=residual_model
)

# Extract seasonality means for use in test data
weekofyear_means = train_df.groupby('weekofyear')['num_orders'].mean()
month_means = train_df.groupby('month')['num_orders'].mean()

# Now apply feature engineering to test with the seasonality means
test_df = apply_feature_engineering(
    test_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means,
    encoding_stats=encoding_stats, uncertainty_models=uncertainty_models, 
    residual_stats=residual_stats, residual_model=residual_model
)

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

# Add exponential decay lag features
FEATURES += [f"{TARGET}_exp_lag_{lag}" for lag in EXP_DECAY_LAGS if f"{TARGET}_exp_lag_{lag}" in train_df.columns]

# Add exponential moving averages  
FEATURES += [f"{TARGET}_exp_ma_{w}" for w in [3, 5, 7, 10] if f"{TARGET}_exp_ma_{w}" in train_df.columns]

# Add advanced rolling statistics
FEATURES += [f"{TARGET}_rolling_median_{w}" for w in ROLLING_WINDOWS if f"{TARGET}_rolling_median_{w}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_skew_{w}" for w in ROLLING_WINDOWS if w >= 3 and f"{TARGET}_rolling_skew_{w}" in train_df.columns]
FEATURES += [f"{TARGET}_rolling_trend_{w}" for w in ROLLING_WINDOWS if w >= 2 and f"{TARGET}_rolling_trend_{w}" in train_df.columns]

# Add target encoding features
FEATURES += [col for col in train_df.columns if col.endswith('_target_encoded')]

# Add uncertainty and meta-learning features
FEATURES += [col for col in train_df.columns if col.startswith('prediction_') or col.startswith('meta_pred_')]

# Add residual correction features
FEATURES += [col for col in train_df.columns if col.startswith('residual_correction_')]

# Add volatility and trend pattern features
FEATURES += [col for col in train_df.columns if any(col.startswith(prefix) for prefix in ['cv_', 'trend_strength_', 'trend_direction_', 'volatility_ratio_', 'order_range_', 'recent_vs_historical'])]

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
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 2000),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 1000.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 1000.0, log=True),
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

# --- Create Ensemble from Top Optuna Trials ---
logging.info("Creating ensemble from top Optuna trials...")

def create_ensemble_from_optuna_trials(study, X_train, y_train, X_valid, y_valid, top_n=5):
    """
    Create ensemble from the best trials found during hyperparameter optimization.
    This ensures all models in the ensemble are actually optimized and performant.
    """
    # Get top trials sorted by objective value (RMSLE)
    trials_df = study.trials_dataframe()
    trials_df = trials_df.dropna(subset=['value'])  # Remove failed trials
    trials_df = trials_df.sort_values('value').head(top_n * 2)  # Get more than needed in case some fail
    
    # Set performance threshold (within 5% of best)
    performance_threshold = study.best_value * 1.05
    logging.info(f"Performance threshold set to: {performance_threshold:.5f}")
    
    ensemble_models = {}
    ensemble_weights = {}
    validation_scores = {}
    
    models_added = 0
    
    for idx, trial_row in trials_df.iterrows():
        if models_added >= top_n:
            break
            
        # Skip if performance is worse than threshold
        if trial_row['value'] > performance_threshold:
            logging.info(f"Skipping trial {trial_row['number']} - RMSLE {trial_row['value']:.5f} > threshold {performance_threshold:.5f}")
            continue
        
        try:
            # Reconstruct parameters from trial
            trial_params = {}
            for param_name in ['learning_rate', 'num_leaves', 'max_depth', 'feature_fraction', 
                             'bagging_fraction', 'bagging_freq', 'min_child_samples', 'lambda_l1', 'lambda_l2']:
                if f'params_{param_name}' in trial_row.index:
                    trial_params[param_name] = trial_row[f'params_{param_name}']
            
            # Add fixed parameters
            trial_params.update({
                'objective': 'regression_l1',
                'boosting_type': 'gbdt',
                'n_estimators': 2000,
                'seed': SEED,
                'n_jobs': -1,
                'verbose': -1,
                'metric': 'None'
            })
            
            model_name = f"optuna_trial_{int(trial_row['number'])}"
            logging.info(f"Training ensemble model: {model_name} (Optuna RMSLE: {trial_row['value']:.5f})")
            
            # Create and train model
            model = LGBMRegressor(**trial_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=lgb_rmsle,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Validate the model performance
            y_pred = model.predict(X_valid)
            validation_rmsle = rmsle(y_valid, y_pred)
            
            # Double-check performance is still good
            if validation_rmsle <= performance_threshold * 1.02:  # Allow small variance
                ensemble_models[model_name] = model
                validation_scores[model_name] = validation_rmsle
                models_added += 1
                logging.info(f"Model {model_name} validation RMSLE: {validation_rmsle:.5f} - INCLUDED")
            else:
                logging.info(f"Model {model_name} validation RMSLE: {validation_rmsle:.5f} - EXCLUDED (worse than expected)")
                
        except Exception as e:
            logging.warning(f"Error training model from trial {trial_row['number']}: {e}")
            continue
    
    # Fallback to single best model if no ensemble can be created
    if len(ensemble_models) == 0:
        logging.warning("No models met performance criteria. Using single best model.")
        best_trial_params = best_params.copy()
        best_trial_params.update({
            'objective': 'regression_l1',
            'boosting_type': 'gbdt',
            'n_estimators': 2000,
            'seed': SEED,
            'n_jobs': -1,
            'verbose': -1,
            'metric': 'None'
        })
        
        model = LGBMRegressor(**best_trial_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=lgb_rmsle,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        ensemble_models = {'best_single': model}
        validation_scores = {'best_single': study.best_value}
    
    # Calculate ensemble weights using performance-based weighting
    if len(ensemble_models) > 1:
        # Use inverse error weighting with smoothing and capping
        max_weight = 0.5  # Prevent any single model from dominating
        
        total_inv_error = 0
        for score in validation_scores.values():
            # Add small epsilon to prevent division by zero
            inv_error = 1 / (score + 0.001)
            total_inv_error += inv_error
        
        for model_name in ensemble_models:
            weight = (1 / (validation_scores[model_name] + 0.001)) / total_inv_error
            # Cap individual model weight
            ensemble_weights[model_name] = min(weight, max_weight)
        
        # Renormalize after capping
        total_weight = sum(ensemble_weights.values())
        ensemble_weights = {k: v/total_weight for k, v in ensemble_weights.items()}
        
    else:
        ensemble_weights = {list(ensemble_models.keys())[0]: 1.0}
    
    logging.info(f"Final ensemble has {len(ensemble_models)} models")
    logging.info(f"Ensemble weights: {ensemble_weights}")
    logging.info(f"Individual model performance: {validation_scores}")
    
    return ensemble_models, ensemble_weights

# Create ensemble from top Optuna trials
ensemble_models, ensemble_weights = create_ensemble_from_optuna_trials(
    study, 
    train_split_df[FEATURES], train_split_df[TARGET],
    valid_df[FEATURES], valid_df[TARGET],
    top_n=ENSEMBLE_MODELS  # Use top ENSEMBLE_MODELS models
)

# Validate ensemble performance
def ensemble_predict(models, weights, X):
    """Make predictions using ensemble of models with weighted averaging."""
    predictions = {}
    
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)
    
    # Weighted average of predictions
    ensemble_pred = np.zeros(len(X))
    for model_name, pred in predictions.items():
        ensemble_pred += weights[model_name] * pred
    
    return ensemble_pred, predictions

# Test ensemble performance on validation set
if len(ensemble_models) > 1:
    ensemble_valid_preds, individual_preds = ensemble_predict(ensemble_models, ensemble_weights, valid_df[FEATURES])
    ensemble_rmsle = rmsle(valid_df[TARGET], ensemble_valid_preds)
    
    logging.info(f"Ensemble validation RMSLE: {ensemble_rmsle:.5f}")
    logging.info(f"Best single model RMSLE: {study.best_value:.5f}")
    
    if ensemble_rmsle < study.best_value:
        logging.info(f"Ensemble improves upon best single model by {study.best_value - ensemble_rmsle:.5f}")
    else:
        logging.warning(f"Ensemble performs worse than best single model by {ensemble_rmsle - study.best_value:.5f}")

# Train final ensemble on full training data
logging.info("Training final ensemble on full training data...")
final_ensemble_models = {}
final_ensemble_weights = {}

for model_name, model in ensemble_models.items():
    # Get the same parameters as the ensemble model
    model_params = model.get_params()
    
    final_model = LGBMRegressor(**model_params)
    
    # Train on the entire training dataset with eval set for detecting overfitting
    train_size = int(0.9 * len(train_df))
    train_indices = np.random.choice(len(train_df), train_size, replace=False)
    eval_indices = np.array([i for i in range(len(train_df)) if i not in train_indices])
    
    final_model.fit(
        train_df[FEATURES], train_df[TARGET], 
        eval_set=[
            (train_df.iloc[train_indices][FEATURES], train_df.iloc[train_indices][TARGET]),
            (train_df.iloc[eval_indices][FEATURES], train_df.iloc[eval_indices][TARGET])
        ],
        eval_metric=lgb_rmsle,
        callbacks=[early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=20, verbose=False)]
    )
    
    final_ensemble_models[model_name] = final_model
    final_ensemble_weights[model_name] = ensemble_weights[model_name]

# Use the ensemble for main predictions
final_model = final_ensemble_models
final_weights = final_ensemble_weights

logging.info(f"Final ensemble trained with {len(final_model)} models")

# --- Recursive Prediction ---
logging.info("Starting recursive prediction on the test set...")
# Prepare the combined data history (training data + test structure)
# We need the structure of test_df but will fill num_orders recursively
history_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

# Extract seasonality means from train_df for use in recursive prediction
weekofyear_means = train_df.groupby('weekofyear')['num_orders'].mean()
month_means = train_df.groupby('month')['num_orders'].mean()

test_weeks = sorted(test_df['week'].unique())

# Initialize adaptive learning tracking
historical_performance = {}  # Track performance for adaptive adjustment


for week_num in test_weeks:
    logging.info(f"Predicting for week {week_num}...")
    # Identify rows for the current week to predict
    current_week_mask = history_df['week'] == week_num

    # Re-apply feature engineering for the current state with seasonality means
    history_df_updated = apply_feature_engineering(
        history_df, is_train=False, 
        weekofyear_means=weekofyear_means, 
        month_means=month_means,
        encoding_stats=encoding_stats,
        uncertainty_models=uncertainty_models,
        residual_stats=residual_stats,
        residual_model=residual_model
    )
    
    # Update history_df with new features
    history_df = history_df_updated

    current_features = history_df.loc[current_week_mask, FEATURES]

    # Handle potential missing columns in test data after alignment
    missing_cols = [col for col in FEATURES if col not in current_features.columns]
    if missing_cols:
        logging.warning(f"Missing columns during prediction for week {week_num}: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            current_features[col] = 0
    current_features = current_features[FEATURES] # Ensure correct order      # Predict for the current week using ensemble
    # Make ensemble predictions
    if isinstance(final_model, dict) and len(final_model) > 1:  # Ensemble model
        current_preds, individual_preds = ensemble_predict(final_model, final_weights, current_features)
    elif isinstance(final_model, dict) and len(final_model) == 1:  # Single model in dict
        model_name = list(final_model.keys())[0]
        current_preds = final_model[model_name].predict(current_features)
    else:  # Single model fallback
        current_preds = final_model.predict(current_features)
    
    # Simple clipping and rounding - no adaptive adjustment
    current_preds = np.clip(current_preds, 0, None).round().astype(float)

    # Update the 'num_orders' in history_df for the current week with predictions
    # This ensures the next iteration uses the predicted values to calculate lags/rolling features
    history_df.loc[current_week_mask, 'num_orders'] = current_preds
    
    # Store prediction statistics for adaptive learning (simplified tracking)
    pred_mean = np.mean(current_preds)
    pred_std = np.std(current_preds)
    
    # Log historical performance for the week
    if week_num not in historical_performance:
        historical_performance[week_num] = {}
    historical_performance[week_num]['predictions'] = current_preds
    historical_performance[week_num]['pred_mean'] = pred_mean
    historical_performance[week_num]['pred_std'] = pred_std
    historical_performance[week_num]['cv'] = pred_std / (pred_mean + 1e-10)  # Coefficient of variation as performance proxy

logging.info("Recursive prediction finished.")

# Extract final predictions for the original test set IDs
final_predictions_df = history_df.loc[history_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int) # Final conversion to int
final_predictions_df['id'] = final_predictions_df['id'].astype(int)

# --- Create Submission File ---
submission_path = f"{SUBMISSION_FILE_PREFIX}_enhanced_ensemble.csv"
final_predictions_df.to_csv(submission_path, index=False)
logging.info(f"Enhanced ensemble submission file saved to {submission_path}")

# Save model performance summary
summary_data = {
    'validation_rmsle': [study.best_value],
    'best_optuna_params': [str(best_params)],
    'ensemble_weights': [str(final_ensemble_weights)],
    'total_features': [len(FEATURES)],
    'advanced_features_enabled': [f"Target encoding: {USE_TARGET_ENCODING}, Uncertainty: {USE_UNCERTAINTY_FEATURES}"],
    'exp_decay_lags': [len(EXP_DECAY_LAGS)],
    'ensemble_models': [len(final_model) if isinstance(final_model, dict) else 1]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{SUBMISSION_FILE_PREFIX}_model_summary.csv", index=False)
logging.info("Model summary saved.")

# --- SHAP Analysis ---
logging.info("Calculating SHAP values...")
try:
    # Sample data for SHAP to keep computation reasonable
    if len(train_df) > N_SHAP_SAMPLES:
        shap_sample = train_df.sample(n=N_SHAP_SAMPLES, random_state=SEED)
    else:
        shap_sample = train_df.copy()

    # Handle ensemble model for SHAP analysis - use the best performing individual model
    if isinstance(final_model, dict):
        # Find the model with highest weight (best performing)
        best_model_name = max(final_ensemble_weights, key=final_ensemble_weights.get)
        shap_model = final_model[best_model_name]
        logging.info(f"Using {best_model_name} for SHAP analysis (weight: {final_ensemble_weights[best_model_name]:.3f})")
    else:
        shap_model = final_model

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(shap_sample[FEATURES])

    # Save SHAP values and importance
    shap_values_df = pd.DataFrame(shap_values, columns=FEATURES)
    shap_values_df.to_csv(f"{SHAP_FILE_PREFIX}_ensemble_values.csv", index=False)

    shap_importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(f"{SHAP_FILE_PREFIX}_ensemble_feature_importances.csv", index=False)

    # Generate SHAP plots
    logging.info("Generating SHAP plots...")
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_sample[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_ensemble_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Importance Bar Plot (Top 25 for more comprehensive view)
    plt.figure(figsize=(12, 10))
    top_features = shap_importance_df.head(25)
    plt.barh(range(len(top_features)), top_features['mean_abs_shap'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.gca().invert_yaxis() # Display most important at the top
    plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
    plt.title('Top 25 SHAP Feature Importances (Enhanced Ensemble Model)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_ensemble_top25_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance comparison plot
    plt.figure(figsize=(14, 8))
    
    # Separate different feature types for visualization
    lag_features = [f for f in FEATURES if 'lag' in f and f in shap_importance_df['feature'].values]
    rolling_features = [f for f in FEATURES if 'rolling' in f and f in shap_importance_df['feature'].values]
    interaction_features = [f for f in FEATURES if '_x_' in f and f in shap_importance_df['feature'].values]
    advanced_features = [f for f in FEATURES if any(keyword in f for keyword in ['target_encoded', 'prediction_', 'residual_', 'exp_']) and f in shap_importance_df['feature'].values]
    
    feature_categories = {
        'Lag Features': lag_features[:10],
        'Rolling Features': rolling_features[:10], 
        'Interaction Features': interaction_features[:10],
        'Advanced Features': advanced_features[:10]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (category, features) in enumerate(feature_categories.items()):
        if features and idx < 4:
            cat_importance = shap_importance_df[shap_importance_df['feature'].isin(features)].head(10)
            if not cat_importance.empty:
                axes[idx].barh(range(len(cat_importance)), cat_importance['mean_abs_shap'])
                axes[idx].set_yticks(range(len(cat_importance)))
                axes[idx].set_yticklabels([f.replace('num_orders_', '') for f in cat_importance['feature']], fontsize=8)
                axes[idx].set_title(f'{category}', fontsize=10)
                axes[idx].grid(axis='x', alpha=0.3)
    
    plt.suptitle('SHAP Feature Importance by Category', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{SHAP_FILE_PREFIX}_ensemble_category_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Enhanced SHAP analysis saved.")

except Exception as e:
    logging.error(f"Error during SHAP analysis: {e}")


# --- Plotting Example: Actual vs Predicted for Validation Set ---
logging.info("Generating enhanced validation plots...")
try:
    # Get ensemble predictions for validation
    if isinstance(final_model, dict):
        valid_preds, _ = ensemble_predict(final_model, final_weights, valid_df[FEATURES])
    else:
        valid_preds = final_model.predict(valid_df[FEATURES])
    
    valid_preds = np.clip(valid_preds, 0, None)
    
    # Create comprehensive validation plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Actual vs Predicted scatter
    axes[0,0].scatter(valid_df[TARGET], valid_preds, alpha=0.5, s=10, color='blue')
    axes[0,0].plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
                   [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0,0].set_xlabel("Actual Orders")
    axes[0,0].set_ylabel("Predicted Orders")
    axes[0,0].set_title(f"Actual vs Predicted (RMSLE: {rmsle(valid_df[TARGET], valid_preds):.4f})")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Predicted
    residuals = valid_df[TARGET] - valid_preds
    axes[0,1].scatter(valid_preds, residuals, alpha=0.5, s=10, color='green')
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel("Predicted Orders")
    axes[0,1].set_ylabel("Residuals")
    axes[0,1].set_title("Residual Plot")
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    relative_errors = np.abs(residuals) / (valid_df[TARGET] + 1)  # +1 to avoid division by zero
    axes[1,0].hist(relative_errors, bins=50, alpha=0.7, color='orange')
    axes[1,0].set_xlabel("Absolute Relative Error")
    axes[1,0].set_ylabel("Frequency")
    axes[1,0].set_title(f"Error Distribution (Mean: {relative_errors.mean():.3f})")
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Performance by prediction magnitude
    pred_ranges = pd.qcut(valid_preds, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    range_performance = []
    range_labels = []
    
    for range_label in pred_ranges.cat.categories:
        mask = pred_ranges == range_label
        if mask.sum() > 0:
            range_rmsle = rmsle(valid_df[TARGET][mask], valid_preds[mask])
            range_performance.append(range_rmsle)
            range_labels.append(range_label)
    
    axes[1,1].bar(range_labels, range_performance, color='purple', alpha=0.7)
    axes[1,1].set_xlabel("Prediction Range")
    axes[1,1].set_ylabel("RMSLE")
    axes[1,1].set_title("Performance by Prediction Magnitude")
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced Validation Analysis - Ensemble Model', fontsize=16)
    plt.tight_layout()
    plt.savefig("enhanced_validation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional ensemble-specific analysis
    if isinstance(final_model, dict):
        fig, axes = plt.subplots(1, len(final_model), figsize=(5*len(final_model), 6))
        if len(final_model) == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(final_model.items()):
            individual_pred = model.predict(valid_df[FEATURES])
            individual_rmsle = rmsle(valid_df[TARGET], individual_pred)
            
            axes[idx].scatter(valid_df[TARGET], individual_pred, alpha=0.5, s=10)
            axes[idx].plot([valid_df[TARGET].min(), valid_df[TARGET].max()], 
                          [valid_df[TARGET].min(), valid_df[TARGET].max()], 'r--', lw=2)
            axes[idx].set_xlabel("Actual Orders")
            axes[idx].set_ylabel("Predicted Orders")
            axes[idx].set_title(f"{model_name}\nRMSLE: {individual_rmsle:.4f}\nWeight: {final_ensemble_weights[model_name]:.3f}")
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Individual Model Performance in Ensemble', fontsize=14)
        plt.tight_layout()
        plt.savefig("ensemble_individual_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info("Enhanced validation plots saved.")

except Exception as e:
    logging.error(f"Error during plotting: {e}")

# --- Performance Summary ---
logging.info("=== ENHANCED MODEL PERFORMANCE SUMMARY ===")
try:
    if isinstance(final_model, dict):
        ensemble_valid_preds, _ = ensemble_predict(final_model, final_weights, valid_df[FEATURES])
        final_rmsle = rmsle(valid_df[TARGET], ensemble_valid_preds)
        logging.info(f"Final Ensemble Validation RMSLE: {final_rmsle:.5f}")
        logging.info(f"Number of ensemble models: {len(final_model)}")
        logging.info(f"Ensemble weights: {final_weights}")
    else:
        single_valid_preds = final_model.predict(valid_df[FEATURES])
        final_rmsle = rmsle(valid_df[TARGET], single_valid_preds)
        logging.info(f"Final Single Model Validation RMSLE: {final_rmsle:.5f}")
    
    logging.info(f"Total features used: {len(FEATURES)}")
    logging.info(f"Advanced features enabled:")
    logging.info(f"  - Target encoding: {USE_TARGET_ENCODING}")
    logging.info(f"  - Uncertainty features: {USE_UNCERTAINTY_FEATURES}")
    logging.info(f"  - Exponential decay lags: {len(EXP_DECAY_LAGS)} lags")
    logging.info(f"  - Advanced rolling stats: median, skew, trend")
    logging.info(f"  - Ensemble models: {ENSEMBLE_MODELS}")
    
except Exception as e:
    logging.error(f"Error in performance summary: {e}")

logging.info("Enhanced script finished successfully!")

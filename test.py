import time
import plotext as pltx  # For live ASCII plotting in the console

import warnings
warnings.filterwarnings("ignore", message="The reported value is ignored because this `step` .* is already reported.")

import pandas as pd
import numpy as np
from math import pi
import matplotlib
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
import lightgbm as lgb  # Added for early stopping callback
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
from optuna.integration import LightGBMPruningCallback
from optuna.samplers.nsgaii import UniformCrossover, SBXCrossover
import shap

import re
import os
from functools import partial
from itertools import combinations
import logging
from tqdm import tqdm

OUTPUT_DIRECTORY = "output"
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
# SEED = 42
import random
SEED = random.randint(0, 1000000) # Random seed distributed for each run
MAX_INTERACTION_ORDER = 4 # Max order of interactions to consider (2nd order = pairwise, 3rd order = triplet, etc.)
MAX_INTERACTIONS_PER_ORDER = {2: 18, 3: 4, 4: 1, 5: 0}
LAG_WINDOWS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]
ROLLING_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 35, 42, 49, 52]
OPTUNA_MULTI_OBJECTIVE = True  # Set to True for multi-objective (mean_valid, gap_penalty, etc.)
OBJECTIVE_WEIGHT_MEAN_VALID = 1.0
OBJECTIVE_WEIGHT_GAP_PENALTY = 0.5
OBJECTIVE_WEIGHT_COMPLEXITY_PENALTY = 0.0001
OBJECTIVE_WEIGHT_REG_REWARD = 0.001
N_ENSEMBLE_MODELS = 5
OVERFIT_ROUNDS = 17 # Overfitting detection rounds
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
N_WARMUP_STEPS = 200 # Warmup steps for Optuna pruning
POPULATION_SIZE = 32 # Population size for Genetic algorithm
OPTUNA_SAMPLER = "Default"
# OPTUNA_SAMPLER = "NSGAIISampler"
# OPTUNA_SAMPLER = "NSGAIIISampler"
PRUNING_ENABLED = False # Enable pruning for Optuna trials
OPTUNA_TRIALS = 1000000 # Number of Optuna trials (increased for better search)
OPTUNA_TIMEOUT = 60 * 60 * 24 # Timeout for Optuna trials (in seconds)
RERUN_TOP_N = 0 # Number of top trials to rerun for final model training
RERUN_OPTUNA_STUDY_NAME = "recursive_lgbm_tuning" # Study name for rerun

OPTUNA_STUDY_NAME = "multi_objective_lgbm_tuning"
# OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
# OPTUNA_DB = "postgresql://neondb_owner:npg_b9Jo7RhUgpSd@ep-proud-dust-a4fztafy-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
OPTUNA_DB = "postgresql://postgres:optuna@34.55.13.135:5432/optuna"

SUBMISSION_FILE_PREFIX = "submission_recursive"
SHAP_FILE_PREFIX = "shap_recursive"
N_SHAP_SAMPLES = 2000
matplotlib.use('Agg')

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

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

def create_temporal_features(df):
    df_out = df.copy()
    df_out["weekofyear"] = df_out["week"] % 52
    df_out["weekofyear_sin"] = np.sin(2 * pi * df_out["weekofyear"] / 52)
    df_out["weekofyear_cos"] = np.cos(2 * pi * df_out["weekofyear"] / 52)
    if "month" not in df_out.columns:
        df_out["month"] = ((df_out["week"] - 1) // 4) % 12 + 1
    df_out["month_sin"] = np.sin(2 * pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * pi * df_out["month"] / 12)
    return df_out

def create_lag_rolling_features(df, target_col='num_orders', rolling_windows=ROLLING_WINDOWS):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS, observed=False)
    shifted = group[target_col].shift(1)
    for window in rolling_windows:
        df_out[f"{target_col}_lag_{window}"] = group[target_col].shift(window)
        df_out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)
    return df_out

def create_other_features(df):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS, observed=False)
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = group["checkout_price"].diff()
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
    return df, weekofyear_means, month_means

def add_binary_rolling_means(df, binary_cols=["emailer_for_promotion", "homepage_featured"], windows=ROLLING_WINDOWS):
    df_out = df.copy()
    group = df_out.groupby(GROUP_COLS, observed=False)
    for col in binary_cols:
        if col in df_out.columns:
            shifted = group[col].shift(1)
            for window in windows:
                df_out[f"{col}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
    return df_out

def apply_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None):
    df_out = df.copy()
    df_out = create_temporal_features(df_out)
    if is_train or 'num_orders' in df_out.columns:
        df_out = create_lag_rolling_features(df_out)
    df_out = create_other_features(df_out)
    df_out = add_binary_rolling_means(df_out, ["emailer_for_promotion", "homepage_featured"], ROLLING_WINDOWS)
    df_out = create_group_aggregates(df_out)
    df_out, weekofyear_means, month_means = add_seasonality_features(df_out, weekofyear_means, month_means, is_train=is_train)
    # Fill NaNs for all engineered features
    # lag_roll_diff_cols = [col for col in df_out.columns if any(sub in col for sub in [
    #     "lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_mean", "_std"
    # ])]
    # cols_to_fill = [col for col in lag_roll_diff_cols if col in df_out.columns and len(df_out[col]) == len(df_out)]
    # if cols_to_fill:
        # df_out.loc[:, cols_to_fill] = df_out[cols_to_fill].fillna(0)
    # if "discount_pct" in df_out.columns:
        # df_out["discount_pct"] = df_out["discount_pct"].fillna(0)
    # Defragment and deduplicate DataFrame ONCE at the end
    df_out = df_out.copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    return df_out, weekofyear_means, month_means

logging.info("Applying feature engineering...")
df_full = pd.concat([df, test], ignore_index=True)
df_full = create_other_features(df_full)
for prefix in ["category_", "cuisine_", "center_type_"]:
    df_full = df_full.loc[:, ~df_full.columns.str.startswith(prefix)]

train_df = df_full[df_full['week'].isin(df['week'].unique())].copy()
test_df = df_full[df_full['week'].isin(test['week'].unique())].copy()

train_df, weekofyear_means, month_means = apply_feature_engineering(train_df, is_train=True)
test_df, _, _ = apply_feature_engineering(test_df, is_train=False, weekofyear_means=weekofyear_means, month_means=month_means)

CATEGORICAL_FEATURES = [col for col in ["category", "cuisine", "center_type", "center_id", "meal_id"] if col in train_df.columns]
for df_ in [train_df, test_df]:
    for col in CATEGORICAL_FEATURES:
        df_[col] = df_[col].astype("category")

# --- Define Features and Target ---
TARGET = "num_orders"
features_set = set()
FEATURES = []

# Add base features
base_features = [
    "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff",
    "category", "cuisine", "center_type", "center_id", "meal_id"
]
for f in base_features:
    if f in train_df.columns and f not in features_set:
        FEATURES.append(f)
        features_set.add(f)

# Add lag features
for lag in LAG_WINDOWS:
    lag_col = f"{TARGET}_lag_{lag}"
    if lag_col in train_df.columns and lag_col not in features_set:
        FEATURES.append(lag_col)
        features_set.add(lag_col)

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

# Add rolling means for binary features
for col in ["emailer_for_promotion", "homepage_featured"]:
    for w in ROLLING_WINDOWS:
        mean_col = f"{col}_rolling_mean_{w}"
        if mean_col in train_df.columns and mean_col not in features_set:
            FEATURES.append(mean_col)
            features_set.add(mean_col)

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

# --- Custom LightGBM RMSLE metric for sklearn API (LGBMRegressor) ---
def rmsle_lgbm(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False # lower is better

# --- Custom Early Stopping Callback with Overfitting Detection ---
def early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=OVERFIT_ROUNDS, verbose=False):
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
            # LightGBM results are in format (data_name, score, is_higher_better) or sometimes (data_name, (metric_name, score, is_higher_better))
            # Correctly unpack the values based on their structure
            try:
                if len(item) == 2:
                    eval_name, eval_result = item
                elif len(item) == 3:
                    eval_name, eval_result, _ = item
                else:
                    continue  # Skip if format is unexpected
                
                # Extract the score value and ensure it's a float
                if isinstance(eval_result, tuple):
                    # Handle (metric_name, score, is_higher_better) format
                    if len(eval_result) >= 2:
                        result_value = eval_result[1]  # score is at index 1
                    else:
                        result_value = eval_result[0]
                else:
                    # Handle case where eval_result is directly the score
                    result_value = eval_result
                
                # Assign to appropriate variable based on dataset name
                if 'train' in eval_name:
                    train_loss = float(result_value)
                elif 'valid' in eval_name or 'validation' in eval_name:
                    valid_loss = float(result_value)
            except (ValueError, TypeError, IndexError) as e:
                # Skip this metric if it can't be processed
                continue
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

final_params = {
    'objective': 'regression_l1',
    'boosting_type': 'gbdt',
    'n_estimators': 3000, # Increase slightly for final training
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'metric': 'None',
    # Stronger regularization defaults (will be overwritten by Optuna if found)
    'lambda_l1': 10.0,
    'lambda_l2': 10.0,
    'min_child_samples': 150,
    'min_data_in_leaf': 250,
    'num_leaves': 16,
    'max_depth': 4,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3
}

class CustomPruningCallback:
    """
    Custom Optuna pruning callback for multi-objective studies.
    Prunes based on the weighted sum of objectives, matching the main Optuna objective calculation.
    """
    def __init__(self, trial, metric, valid_name, objective_weights=(1.0, 1.0, 1.0, 1.0)):
        self._trial = trial
        self._metric = metric
        self._valid_name = valid_name
        self._objective_weights = objective_weights

    def __call__(self, env):
        # Check if the study is multi-objective
        if hasattr(self._trial.study, "directions") and len(self._trial.study.directions) > 1:
            # Find the validation score for the specified metric and valid_name
            for eval_name, score, is_higher_better, _ in env.evaluation_result_list:
                if self._valid_name in eval_name and self._metric in eval_name:
                    # Retrieve all objectives from the trial's user attributes
                    mean_valid = self._trial.user_attrs.get('mean_valid')
                    gap_penalty = self._trial.user_attrs.get('gap_penalty')
                    complexity_penalty = self._trial.user_attrs.get('complexity_penalty')
                    reg_reward = self._trial.user_attrs.get('reg_reward')
                    # If any are missing, fallback to score for pruning
                    if None in (mean_valid, gap_penalty, complexity_penalty, reg_reward):
                        value = score
                    else:
                        w = self._objective_weights
                        value = (
                            w[0] * mean_valid +
                            w[1] * gap_penalty
                            # + w[2] * complexity_penalty +
                            # w[3] * reg_reward
                        )
                    self._trial.report(value, step=env.iteration)
                    if self._trial.should_prune():
                        raise optuna.TrialPruned()
        else:
            # Single-objective fallback: prune on metric
            for eval_name, score, is_higher_better, _ in env.evaluation_result_list:
                if self._valid_name in eval_name and self._metric in eval_name:
                    self._trial.report(score, step=env.iteration)
                    if self._trial.should_prune():
                        raise optuna.TrialPruned()


class ExpandingGroupTimeSeriesSplit:
    """
    Expanding window cross-validator for time series data with group awareness.
    For each split, the training set starts at the beginning and expands,
    the validation set is the next val_window unique weeks.
    """
    def __init__(self, n_splits=5, min_train_window=20, val_window=10, week_col='week'):
        self.n_splits = n_splits
        self.min_train_window = min_train_window
        self.val_window = val_window
        self.week_col = week_col

    def split(self, X, y=None, groups=None):
        weeks = np.sort(X[self.week_col].unique())
        total_weeks = len(weeks)
        max_start = total_weeks - self.min_train_window - self.val_window + 1
        if self.n_splits > max_start:
            raise ValueError(f"Not enough weeks for {self.n_splits} splits with min_train_window={self.min_train_window} and val_window={self.val_window}.")
        for i in range(self.n_splits):
            train_end = self.min_train_window + i * (max_start // self.n_splits)
            val_start = train_end
            val_end = val_start + self.val_window
            train_weeks = weeks[:train_end]
            val_weeks = weeks[val_start:val_end]
            train_mask = X[self.week_col].isin(train_weeks)
            val_mask = X[self.week_col].isin(val_weeks)
            train_indices = np.where(train_mask & pd.notnull(groups))[0]
            val_indices = np.where(val_mask & pd.notnull(groups))[0]
            yield train_indices, val_indices

class RollingGroupTimeSeriesSplit:
    """
    Rolling window cross-validator for time series data with group awareness.
    For each split, the training set is a rolling window of train_window unique weeks,
    and the validation set is the next val_window unique weeks.
    Groups are respected (e.g., center_id, meal_id).
    No gap is used between train and validation.
    """
    def __init__(self, n_splits=3, train_window=80, val_window=10, week_col='week'):
        self.n_splits = n_splits
        self.train_window = train_window
        self.val_window = val_window
        self.week_col = week_col

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Group labels must be provided for RollingGroupTimeSeriesSplit.")
        weeks = np.sort(X[self.week_col].unique())
        total_weeks = len(weeks)
        max_start = total_weeks - self.train_window - self.val_window + 1
        if self.n_splits > max_start:
            raise ValueError(f"Not enough weeks for {self.n_splits} splits with train_window={self.train_window} and val_window={self.val_window}.")
        for i in range(self.n_splits):
            train_start = i * (max_start // self.n_splits)
            train_end = train_start + self.train_window
            val_start = train_end
            val_end = val_start + self.val_window
            train_weeks = weeks[train_start:train_end]
            val_weeks = weeks[val_start:val_end]
            train_mask = X[self.week_col].isin(train_weeks)
            val_mask = X[self.week_col].isin(val_weeks)
            # Respect groups: only include indices where group is not missing
            train_indices = np.where(train_mask & pd.notnull(groups))[0]
            val_indices = np.where(val_mask & pd.notnull(groups))[0]
            yield train_indices, val_indices

# --- Feature Selection and Hyperparameter Tuning with Optuna ---

# --- Precompute eligible features and all possible combos for Optuna feature interactions (module-level, before study) ---

# --- Freeze eligible features for Optuna interaction search ---
FROZEN_FEATURES_FOR_INTERACTIONS = [
    'checkout_price', 'base_price', 'homepage_featured', 'emailer_for_promotion',
    'discount', 'discount_pct', 'price_diff',
    'center_orders_mean', 'meal_orders_mean',
    'mean_orders_by_weekofyear', 'mean_orders_by_month'
    # Add more features as needed, but do not change this list between runs of the same study!
]
ALL_COMBOS_STR = {}

for order in range(2, MAX_INTERACTION_ORDER + 1):
    combos = list(combinations(FROZEN_FEATURES_FOR_INTERACTIONS, order))
    ALL_COMBOS_STR[order] = ["|".join(str(f) for f in combo) for combo in combos]

# Defensive: ensure all elements are strings (Optuna requires this for categorical choices)
for order, combos in ALL_COMBOS_STR.items():
    for c in combos:
        assert isinstance(c, str), f"ALL_COMBOS_STR[{order}] contains non-string: {c} ({type(c)})"

# --- Noise injection for robust training ---
def add_training_noise(df, features, target,
                       noise_target_level=0.0,
                       noise_feature_level=0.0,
                       bootstrap_frac=0.0,
                       seed=None,
                       group_cols=None):
    """
    Add noise to training data for robustness.
    noise_target_level: std multiplier for Gaussian noise on target.
    noise_feature_level: std multiplier for Gaussian noise on numeric features.
    bootstrap: whether to apply group-wise bootstrap sampling.
    """
    df_noise = df.copy()
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    X = df_noise[features].copy()
    y = df_noise[target].copy()
    # target noise
    if noise_target_level > 0:
        y_std = y.std()
        y += rng.normal(0, noise_target_level * y_std, size=len(y)).clip(0) # Ensure non-negative
    # feature noise
    if noise_feature_level > 0:
        for col in features:
            # Numeric columns only, skip categorical dtype
            if pd.api.types.is_numeric_dtype(X[col]) and not isinstance(X[col].dtype, pd.CategoricalDtype):
                f_std = X[col].std()
                if f_std > 0:
                    X[col] += rng.normal(0, noise_feature_level * f_std, size=len(X))
    # bootstrap noise (fractional group-wise resampling)
    if bootstrap_frac > 0 and group_cols:
        idx = []
        # explicit observed=False to suppress future warning
        for _, grp in df_noise.groupby(group_cols, observed=False):
            inds = grp.index.values
            n = max(1, int(len(inds) * bootstrap_frac))
            idx.extend(rng.choice(inds, size=n, replace=True))
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)
    return X, y

def optuna_feature_selection_and_hyperparam_objective(trial, train_split_df=train_split_df):
    # Hyperparameter search space
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    if boosting_type != 'goss':
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0)  # Lower min for more regularization (helps generalization)
        bagging_freq = trial.suggest_int('bagging_freq', 0, 10)
    else:
        bagging_fraction = 1.0
        bagging_freq = 0
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True), # Higher max allows faster learning, but can overfit if too high
        'num_leaves': trial.suggest_int('num_leaves', 4, 512), # Higher max allows more complex trees, but can overfit
        'max_depth': trial.suggest_int('max_depth', 2, 30),    # Higher max allows deeper trees, but can overfit
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),  # Lower min increases regularization, helps generalization
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 2000), # Higher max prevents overfit with large trees
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1000.0, log=True), # Higher max increases regularization, helps prevent overfit
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1000.0, log=True), # Higher max increases regularization, helps prevent overfit
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0), 
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 2000), # Higher max prevents overfit with large trees
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 500000),
        'boosting_type': boosting_type,
        'max_bin': trial.suggest_int('max_bin', 32, 1024), # Higher max allows finer binning, can help with continuous features
        'objective': 'regression_l1',
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000), # Higher max allows more trees, but can overfit
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric': 'rmsle',
    }
    
    # Noise injection hyperparameters as continuous tunables
    noise_target_level = trial.suggest_float('noise_target_level', 0.0, 0.1)
    noise_feature_level = trial.suggest_float('noise_feature_level', 0.0, 0.1)
    bootstrap_frac = trial.suggest_float('bootstrap_frac', 0.0, 0.3)

    # Find all features with sin/cos in their name (excluding those already in a pair)
    sincos_features = [f for f in FEATURES if re.search(r'(_sin|_cos)', f)]
    pair_map = {}
    for f in sincos_features:
        m = re.match(r'(.*)_sin$', f)
        if m and f.replace('_sin', '_cos') in sincos_features:
            prefix = m.group(1)
            pair_map[prefix] = (f, f.replace('_sin', '_cos'))
    # For each pair, add a trial param
    selected_features = []
    for prefix, (sin, cos) in pair_map.items():
        pair_name = f"{sin}_{cos}_pair"
        if trial.suggest_categorical(pair_name, [True, False]):
            selected_features.extend([sin, cos])
    # Only tune non-sin/cos features individually
    selected_features += [f for f in FEATURES if (f not in sincos_features) and trial.suggest_categorical(f, [True, False])]
    
    # --- Robust dynamic feature interaction logic (DRY, all types) ---
    interaction_features = []
    new_interaction_cols = {}
    used_interactions = set()
    eligible = FROZEN_FEATURES_FOR_INTERACTIONS.copy()
    for order in range(2, MAX_INTERACTION_ORDER + 1):
        combos = list(combinations(eligible, order))
        all_combos_str = ["|".join(str(f) for f in combo) for combo in combos]
        max_this_order = min(MAX_INTERACTIONS_PER_ORDER.get(order, 1), len(all_combos_str))
        n_this_order = trial.suggest_int(f"n_{order}th_order", 0, max_this_order) if max_this_order > 0 else 0
        for i in range(n_this_order):
            combo_str = trial.suggest_categorical(f"inter_{order}th_{i}", all_combos_str)
            if combo_str in used_interactions:
                continue
            used_interactions.add(combo_str)
            combo = combo_str.split("|")
            # Product
            new_col_prod = '_prod_'.join(combo)
            if new_col_prod not in train_split_df.columns and new_col_prod not in new_interaction_cols:
                col_val = train_split_df[combo[0]]
                for f in combo[1:]:
                    col_val = col_val * train_split_df[f]
                new_interaction_cols[new_col_prod] = col_val
            interaction_features.append(new_col_prod)
            # Additive
            new_col_add = '_add_'.join(combo)
            if new_col_add not in train_split_df.columns and new_col_add not in new_interaction_cols:
                col_val = train_split_df[combo[0]]
                for f in combo[1:]:
                    col_val = col_val + train_split_df[f]
                new_interaction_cols[new_col_add] = col_val
            interaction_features.append(new_col_add)
            # Ratio (only for pairs)
            if len(combo) == 2:
                new_col_div = '_div_'.join(combo)
                if new_col_div not in train_split_df.columns and new_col_div not in new_interaction_cols:
                    denominator = train_split_df[combo[1]].replace(0, np.nan)
                    new_interaction_cols[new_col_div] = train_split_df[combo[0]] / denominator + 1e-15 # Avoid division by zero
                    new_interaction_cols[new_col_div] = new_interaction_cols[new_col_div]
                interaction_features.append(new_col_div)
                # Polynomial: feature1 * feature2 ** 2
                new_col_poly2 = f'{combo[0]}_poly2_{combo[1]}'
                if new_col_poly2 not in train_split_df.columns and new_col_poly2 not in new_interaction_cols:
                    new_interaction_cols[new_col_poly2] = train_split_df[combo[0]] * (train_split_df[combo[1]] ** 2)
                interaction_features.append(new_col_poly2)
    if new_interaction_cols:
        train_split_df = pd.concat([train_split_df, pd.DataFrame(new_interaction_cols, index=train_split_df.index)], axis=1)
    selected_features += interaction_features

    # Ensure selected features are unique and not empty
    selected_features = list(dict.fromkeys(selected_features))
    if len(selected_features) < 10:
        logging.warning(f"Optuna selected {len(selected_features)} features, which is less than 10. This may lead to overfitting.")
        raise optuna.TrialPruned()
    rgs = ExpandingGroupTimeSeriesSplit(n_splits=5, min_train_window=30, val_window=10, week_col='week')
    groups = train_split_df["center_id"]
    train_scores, valid_scores = [], []
    is_multi_objective = isinstance(trial.study.directions, list) and len(trial.study.directions) > 1
    
    if not PRUNING_ENABLED:
        callbacks = [
            early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False),
        ]
    elif is_multi_objective or OPTUNA_SAMPLER in ["NSGAIISampler", "NSGAIIISampler"]:
        callbacks = [
            CustomPruningCallback(trial, metric='rmsle', valid_name='valid_1'),
            early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False),
        ]
    else:
        callbacks = [
            LightGBMPruningCallback(trial, metric='rmsle', valid_name='valid_1'),
            early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False),
        ]

    categorical_feature=[col for col in CATEGORICAL_FEATURES if col in selected_features]

    for train_idx, valid_idx in rgs.split(train_split_df, groups=groups):
        # Inject noise into training split
        train_sub = train_split_df.iloc[train_idx].reset_index(drop=True)
        # train_sub = train_split_df.iloc[train_idx].copy()

        X_train, y_train = add_training_noise(
            train_sub, selected_features, TARGET,
            noise_target_level=noise_target_level,
            noise_feature_level=noise_feature_level,
            bootstrap_frac=bootstrap_frac,
            seed=SEED + trial.number,
            group_cols=GROUP_COLS
        )
        model = LGBMRegressor(**params)
        # Train with only callbacks for early stopping/pruning
        model.fit(
            X_train, y_train,
            eval_set=[
                (X_train, y_train),
                (train_split_df.iloc[valid_idx][selected_features], train_split_df.iloc[valid_idx][TARGET])
            ],
            eval_metric=rmsle_lgbm,
            callbacks=callbacks,
            categorical_feature=categorical_feature
        )
        y_train_pred = model.predict(train_split_df.iloc[train_idx][selected_features])
        y_valid_pred = model.predict(train_split_df.iloc[valid_idx][selected_features])
        train_score = rmsle(train_split_df.iloc[train_idx][TARGET], y_train_pred)
        valid_score = rmsle(train_split_df.iloc[valid_idx][TARGET], y_valid_pred)
        train_scores.append(train_score)
        valid_scores.append(valid_score)

    mean_train = np.mean(train_scores)
    mean_valid = np.mean(valid_scores)
    generalization_gap = mean_valid - mean_train

    # --- Penalty terms as separate objectives ---
    valid_reward = OBJECTIVE_WEIGHT_MEAN_VALID * mean_valid
    gap_penalty = OBJECTIVE_WEIGHT_GAP_PENALTY * max(0, generalization_gap)  # Weight can be tuned. Higher weight = more penalty for overfitting
    complexity_penalty = OBJECTIVE_WEIGHT_COMPLEXITY_PENALTY * len(selected_features)
    complexity_penalty += OBJECTIVE_WEIGHT_COMPLEXITY_PENALTY * params['num_leaves']
    complexity_penalty += OBJECTIVE_WEIGHT_COMPLEXITY_PENALTY * (params['max_depth'] if params['max_depth'] > 0 else 0)
    # Use log1p to avoid favoring very high lambda values (log-scale regularization)
    reg_reward = OBJECTIVE_WEIGHT_REG_REWARD * (np.log1p(params['lambda_l1']) + np.log1p(params['lambda_l2']))

    # Store metrics for logging in callback
    trial.set_user_attr('mean_train', float(mean_train))
    trial.set_user_attr('mean_valid', float(mean_valid))
    trial.set_user_attr('generalization_gap', float(generalization_gap))
    trial.set_user_attr('valid_reward', float(valid_reward))
    trial.set_user_attr('gap_penalty', float(gap_penalty))
    trial.set_user_attr('complexity_penalty', float(complexity_penalty))
    trial.set_user_attr('reg_reward', float(reg_reward))
    # Store selected features and objective value for callback/plotting
    trial.set_user_attr('n_features', float(len(selected_features)))
    if (is_multi_objective):
        objective_val = mean_valid + gap_penalty
    else:
        objective_val = mean_valid
    trial.set_user_attr('objective', objective_val)

    # Multi-objective: minimize mean_valid, gap_penalty, complexity_penalty, maximize reg_reward (so minimize -reg_reward)
    # If any objective is nan/inf, prune
    objectives = [mean_valid, gap_penalty]
    if any(np.isnan(obj) or np.isinf(obj) for obj in objectives):
        logging.warning(f"Optuna trial {trial.number} produced invalid objectives: {objectives}.")
        raise optuna.TrialPruned()

    if (is_multi_objective):
        return objectives
    else:
        return mean_valid

logging.info("Starting Optuna feature+hyperparam selection...")

# Reduce Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

class TqdmOptunaCallback:
    def __init__(self, n_trials, study=None, print_every=1):
        self.n_trials = n_trials
        self.print_every = print_every
        self.study = study
        self.terminal_height = None
        self.pbar = tqdm(total=n_trials, desc="Optuna Trials", position=0, leave=False)
        # Track best trial number and value for display
        self.best_trial_number = None
        self.best_trial_value = float('inf')
        # Try to initialize from study if available
        if study is not None:
            try:
                # Find the best trial by minimum objective value (not just study.best_trial, which may be incomplete)
                trials = [t for t in getattr(study, 'trials', []) if hasattr(t, 'user_attrs') and 'objective' in t.user_attrs]
                if trials:
                    best = min(trials, key=lambda t: t.user_attrs['objective'] if t.user_attrs['objective'] is not None else float('inf'))
                    self.best_trial_number = best.number
                    self.best_trial_value = best.user_attrs['objective']
                elif hasattr(study, 'best_trials') and study.best_trials:
                    self.best_trial_number = study.best_trials[0].number
                    self.best_trial_value = study.best_trials[0].values[0] if hasattr(study.best_trials[0], 'values') else study.best_trials[0].value
                elif hasattr(study, 'best_trial') and study.best_trial is not None:
                    self.best_trial_number = study.best_trial.number
                    self.best_trial_value = study.best_trial.value
            except Exception:
                pass


    def __call__(self, study, trial):
        self.pbar.update(1)
        num_leaves = trial.params.get('num_leaves', None)
        max_depth = trial.params.get('max_depth', None)
        lambda_l1 = trial.params.get('lambda_l1', None)
        lambda_l2 = trial.params.get('lambda_l2', None)
        mean_valid = trial.user_attrs.get('mean_valid', None)
        generalization_gap = trial.user_attrs.get('generalization_gap', None)
        n_features = trial.user_attrs.get('n_features', None)
        objective_val = trial.user_attrs.get('objective', None)

        # Update best trial number/value if this trial is better (minimize objective)
        # Use objective_val if available, else mean_valid
        try:
            val = float(objective_val) if objective_val is not None else (float(mean_valid) if mean_valid is not None else None)
            if val is not None and val < self.best_trial_value:
                self.best_trial_value = val
                self.best_trial_number = trial.number
        except Exception:
            pass

        def fmt(x):
            if x is None:
                return 'None'
            if isinstance(x, float):
                return f"{x:.5f}"
            return str(x)

        sep = "!" if trial.number == self.best_trial_number else ":"

        msg = (
            f"Trial {trial.number} | Best Trial{sep} {self.best_trial_number} ({fmt(self.best_trial_value)}) | Objective: {fmt(objective_val)} | Mean Valid: {fmt(mean_valid)} | Gap: {fmt(generalization_gap)} | "
            f"Features: {int(n_features) if n_features is not None else 'None'} | "
            f"Num Leaves: {fmt(num_leaves)} | Max Depth: {fmt(max_depth)} | "
            f"L1: {fmt(lambda_l1)} | L2: {fmt(lambda_l2)}"
        )
        # Color green if this is the best trial so far
        if trial.number == self.best_trial_number:
            msg = f"\033[92m{msg}\033[0m"

        # Live ASCII plot using plotext, with progress bar handling
        if trial.number % self.print_every == 0:
            import plotext as pltx

            start_t = getattr(self.pbar, 'start_t', None)

            self.pbar.close()  # Close progress bar before plotting

            # Clear previous plot lines
            if self.terminal_height is not None:
                pltx.clear_terminal(lines=self.terminal_height)

            # Set terminal size for plotext clear_terminal
            self.terminal_height = pltx.terminal_height()

            self.live_plot_objectives(study.trials, top_string=msg)

            self.pbar = tqdm(total=self.n_trials, desc="Optuna Trials", position=0, leave=False)
            self.pbar.n = trial.number + 1  # Restore progress

            # Restore the original start time for correct elapsed/ETA display
            if start_t is not None:
                self.pbar.start_t = start_t
                self.pbar.last_print_t = time.time()  # Force immediate refresh

            self.pbar.refresh()


    def live_plot_objectives(self, trials, top_string):
        pltx.clf()
        trial_nums = []
        mean_valids = []
        gap_penalties = []
        complexity_penalties = []
        reg_rewards = []
        objectives = []
        n_features_list = []

        # Determine number of objectives in the study
        directions = getattr(self.study, 'directions', None)
        n_obj = len(directions) if directions is not None else 1
        for t in trials:
            if t.values is not None and len(t.values) >= n_obj:
                trial_nums.append(t.number)
                # Primary objectives
                if n_obj >= 1:
                    mean_valids.append(t.values[0])
                if n_obj >= 2:
                    gap_penalties.append(t.values[1])
                if n_obj >= 3:
                    complexity_penalties.append(t.values[2])
                if n_obj >= 4:
                    # Negate reg_reward objective as it's minimized on -reg_reward
                    reg_rewards.append(-t.values[3])
                # Store combined objective and features
                obj = t.user_attrs.get('objective', None)
                objectives.append(obj)
                n_features = t.user_attrs.get('n_features', None)
                n_features_list.append(n_features)

        # Min-max scaling helper
        def minmax(lst):
            if not lst or min(lst) == max(lst):
                return [0.5 for _ in lst]
            mn, mx = min(lst), max(lst)
            return [(x - mn) / (mx - mn) for x in lst]

        # Compute best value so far (across all trials, including parallel/remote)
        best_value_str = "N/A"
        best_objective_str = "N/A"
        if self.study is not None:
            try:
                # For multi-objective, show best value for the first objective (mean_valid) and best objective (weighted sum)
                is_multi_objective = hasattr(self.study, 'directions') and len(getattr(self.study, 'directions', [])) > 1
                all_trials = self.study.get_trials(deepcopy=False)
                if is_multi_objective:
                    # Only consider trials with valid values
                    valid_trials = [t for t in all_trials if t.values is not None and not any(pd.isnull(v) or np.isinf(v) for v in t.values)]
                    if valid_trials:
                        best_value = min(t.values[0] for t in valid_trials)
                        best_value_str = f"{best_value:.5f}"
                    # Best objective (from user_attrs['objective'])
                    valid_obj_trials = [t for t in all_trials if hasattr(t, 'user_attrs') and 'objective' in t.user_attrs and t.user_attrs['objective'] is not None and not (pd.isnull(t.user_attrs['objective']) or np.isinf(t.user_attrs['objective']))]
                    if valid_obj_trials:
                        best_objective = min(t.user_attrs['objective'] for t in valid_obj_trials)
                        best_objective_str = f"{best_objective:.5f}"
                else:
                    valid_trials = [t for t in all_trials if t.value is not None and not (pd.isnull(t.value) or np.isinf(t.value))]
                    if valid_trials:
                        best_value = min(t.value for t in valid_trials)
                        best_value_str = f"{best_value:.5f}"
                    # Best objective (from user_attrs['objective'])
                    valid_obj_trials = [t for t in all_trials if hasattr(t, 'user_attrs') and 'objective' in t.user_attrs and t.user_attrs['objective'] is not None and not (pd.isnull(t.user_attrs['objective']) or np.isinf(t.user_attrs['objective']))]
                    if valid_obj_trials:
                        best_objective = min(t.user_attrs['objective'] for t in valid_obj_trials)
                        best_objective_str = f"{best_objective:.5f}"
            except Exception:
                pass

        # Limit to most recent trials based on console width (at most 1 trial per character)
        if trial_nums:
            try:
                console_width = pltx.terminal_width()
            except Exception:
                console_width = 80  # Fallback if plotext fails
            n = len(trial_nums)
            if n > console_width:
                # Keep only the most recent trials, at most 1 per character
                trial_nums = trial_nums[-console_width:]
                mean_valids = mean_valids[-console_width:]
                gap_penalties = gap_penalties[-console_width:]
                complexity_penalties = complexity_penalties[-console_width:]
                reg_rewards = reg_rewards[-console_width:]
                objectives = objectives[-console_width:]
                n_features_list = n_features_list[-console_width:]

            # Min-max scale all objectives
            gap_penalties_scaled = gap_penalties
            complexity_penalties_scaled = complexity_penalties
            reg_rewards_scaled = reg_rewards
            mean_valids_scaled = mean_valids
            objectives_scaled = objectives
            # gap_penalties_scaled = minmax(gap_penalties)
            # complexity_penalties_scaled = minmax(complexity_penalties)
            # reg_rewards_scaled = minmax(reg_rewards)
            # mean_valids_scaled = minmax(mean_valids)
            # objectives_scaled = minmax(objectives)

            # Label for mean_valid's value (last trial)
            gap_penalty_label = f"gap_penalty (last: {gap_penalties[-1]:.4f})" if gap_penalties else "gap_penalty"
            complexity_penalty_label = f"complexity_penalty (last: {complexity_penalties[-1]:.4f})" if complexity_penalties else "complexity_penalty"
            reg_reward_label = f"reg_reward (last: {reg_rewards[-1]:.4f})" if reg_rewards else "reg_reward"
            mean_valid_label = f"mean_valid (last: {mean_valids[-1]:.4f})" if mean_valids else "mean_valid"
            objective_label = f"objective (last: {objectives[-1]:.4f})" if objectives else "objective"

            # Plot each objective series if available
            if mean_valids:
                pltx.plot(trial_nums, mean_valids_scaled, label=mean_valid_label, color=tuple([0,255,0]), marker='braille')
            if gap_penalties:
                pltx.plot(trial_nums, gap_penalties_scaled, label=gap_penalty_label, color=tuple([0,255,255]), marker='braille')
            if complexity_penalties:
                pltx.plot(trial_nums, complexity_penalties_scaled, label=complexity_penalty_label, color=tuple([255,0,0]), marker='braille')
            if reg_rewards:
                pltx.plot(trial_nums, reg_rewards_scaled, label=reg_reward_label, color=tuple([128,0,255]), marker='braille')
            if objectives:
                pltx.plot(trial_nums, objectives_scaled, label=objective_label, color=tuple([255,255,0]), marker='braille')

            pltx.title(f'Optuna Objectives (Live) | Best validation: {best_value_str} | Best objective: {best_objective_str}')
            pltx.xlabel('Trial')
            pltx.ylabel('Value')

            pltx.canvas_color(tuple([0, 0, 0]))
            pltx.axes_color(tuple([0, 0, 0]))
            pltx.ticks_color('grey')
            
            pltx.grid(True)

            if (top_string):
                print(top_string)

            pltx.show()

    def close(self):
        self.pbar.close()

# Create the study 
optuna_storage = OPTUNA_DB
if OPTUNA_SAMPLER == "NSGAIISampler":
    sampler = optuna.samplers.NSGAIISampler(
        seed=SEED,
        population_size=POPULATION_SIZE,
        crossover=UniformCrossover(),
        crossover_prob=0.9,
        swapping_prob=0.5,
    )
    pruner = optuna.pruners.NopPruner()  # Pruning not supported for multi-objective
elif OPTUNA_SAMPLER == "NSGAIIISampler":
    sampler = optuna.samplers.NSGAIIISampler(
        seed=SEED,
        population_size=POPULATION_SIZE,
        crossover=UniformCrossover(),
        crossover_prob=0.9,
        swapping_prob=0.5,
    )
    if PRUNING_ENABLED:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,           # Minimum number of iterations/training steps before pruning
            max_resource=300,         # Maximum number of iterations (matches early stopping rounds)
            reduction_factor=3,       # How aggressively to halve trials (3 is a good default)
            bootstrap_count=0         # No bootstrapping, start pruning immediately
        )
    else:
        pruner = optuna.pruners.NopPruner() # No pruning
else:
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
    )
    # For single-objective, you can use any of the objectives, e.g. mean_valid
    if PRUNING_ENABLED:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=N_WARMUP_STEPS,
            n_min_trials=5,
            interval_steps=1,
        )
    else:
        pruner = optuna.pruners.NopPruner() # No pruning

if (OPTUNA_MULTI_OBJECTIVE):
    directions = ["minimize", "minimize"]  # mean_valid, gap_penalty
    # directions = ["minimize", "minimize", "minimize", "minimize"]  # mean_valid, gap_penalty, complexity_penalty, -reg_reward
    feature_hyperparam_study = optuna.create_study(
        directions=directions,
        pruner=pruner,
        study_name=OPTUNA_STUDY_NAME,
        storage=optuna_storage,
        load_if_exists=True,
        sampler=sampler
    )
else:
    directions = "minimize"
    feature_hyperparam_study = optuna.create_study(
        direction=directions,
        pruner=pruner,
        study_name=OPTUNA_STUDY_NAME,
        storage=optuna_storage,
        load_if_exists=True,
        sampler=sampler
    )

# Pass the study to the callback so it can initialize best_value/best_trial
print (f"Optuna study name: {OPTUNA_STUDY_NAME}")
optuna_callback = TqdmOptunaCallback(OPTUNA_TRIALS, study=feature_hyperparam_study, print_every=1)



def rerun_old_trials_in_new_study(old_study, new_study, train_split_df=train_split_df, n_trials=None):
    """
    Rerun top N trials from old_study in new_study, skipping any with params not valid in the new search space.
    """

    # Get completed trials sorted by value (lowest first)
    completed = [t for t in old_study.trials if t.state == 'COMPLETE']
    completed = sorted(completed, key=lambda t: t.value if hasattr(t, "value") and t.value is not None else float("inf"))
    if n_trials is not None:
        completed = completed[:n_trials]

    # Diagnostic: print ALL_COMBOS_STR for each order
    print("\n[DIAGNOSTIC] ALL_COMBOS_STR (new search space):")
    for order, combos in ALL_COMBOS_STR.items():
        print(f"  Order {order}: {len(combos)} combos")
        if len(combos) <= 10:
            print(f"    {combos}")
        else:
            print(f"    {combos[:5]} ... {combos[-5:]}")

    for t in completed:
        params = t.params
        # Diagnostic: print params for this trial
        print(f"\n[DIAGNOSTIC] Checking trial {t.number} params:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        # Validate all categorical params are still valid in the new search space
        valid = True
        for k, v in params.items():
            # If this is an interaction param, check if the value is in the new ALL_COMBOS_STR
            if k.startswith("inter_"):
                order = int(k.split("_")[1][0])
                if v not in ALL_COMBOS_STR.get(order, []):
                    print(f"[DIAGNOSTIC]   -> INVALID: {k} value '{v}' not in ALL_COMBOS_STR[{order}]")
                    valid = False
                    break
        if not valid:
            print(f"Skipping trial {t.number}: param {k} value {v} not in new search space.")
            continue
        # Rerun the trial in the new study
        def fixed_objective(trial, train_split_df):
            # Set all params as fixed, but for known categoricals use full search space
            for k, v in params.items():
                if k == 'boosting_type':
                    trial.suggest_categorical(k, ['gbdt', 'dart', 'goss'])
                # Add other known categoricals here as needed
                elif isinstance(v, bool):
                    trial.suggest_categorical(k, [True, False])
                elif isinstance(v, int):
                    trial.suggest_int(k, v, v)
                elif isinstance(v, float):
                    trial.suggest_float(k, v, v)
                else:
                    # For interaction features, use the full set from ALL_COMBOS_STR
                    if k.startswith("inter_"):
                        order = int(k.split("_")[1][0])
                        trial.suggest_categorical(k, ALL_COMBOS_STR.get(order, [v]))
                    else:
                        trial.suggest_categorical(k, [v])
            # Call the main objective (with fixed params)
            return optuna_feature_selection_and_hyperparam_objective(trial, train_split_df=train_split_df)
        print(f"Rerunning trial {t.number} in new study...")
        # new_study.optimize(fixed_objective, n_trials=1, catch=(Exception,))
        new_study.optimize(
            partial(fixed_objective, train_split_df=train_split_df),
            n_trials=1,
            timeout=OPTUNA_TIMEOUT,
            callbacks=[optuna_callback],
            n_jobs=1
        )

# Load the old study and create a new study with the same name
if RERUN_TOP_N > 0 and RERUN_OPTUNA_STUDY_NAME:
    print(f"Rerunning top {RERUN_TOP_N} trials from old study '{RERUN_OPTUNA_STUDY_NAME}'...")
    old_study = optuna.load_study(study_name=RERUN_OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    rerun_old_trials_in_new_study(old_study, feature_hyperparam_study, train_split_df=train_split_df, n_trials=RERUN_TOP_N)



try:
    feature_hyperparam_study.optimize(
        partial(optuna_feature_selection_and_hyperparam_objective, train_split_df=train_split_df),
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        callbacks=[optuna_callback],
        n_jobs=1
    )
except KeyboardInterrupt:
    logging.warning("Optuna optimization interrupted by user. All completed trials are saved.")
except Exception as e:
    logging.warning(f"Optuna optimization failed: {e}")

if hasattr(feature_hyperparam_study, 'best_trials') and feature_hyperparam_study.best_trials:
    print("Optuna study completed. Best trials:")
    for i, t in enumerate(feature_hyperparam_study.best_trials):
        print(f"  Pareto {i}: trial #{t.number}, values={t.values}")
else:
    print("Optuna study completed. No best trials found.")

# Close the progress bar
optuna_callback.close()

# Reload the study from storage to ensure best_trial is up to date
feature_hyperparam_study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)

# Diagnostic: Show all trial values and states after reload

# Handle both single- and multi-objective studies
df_trials = feature_hyperparam_study.trials_dataframe()
value_cols = [col for col in df_trials.columns if col.startswith('values_')]
if value_cols:
    
    print(df_trials[['number'] + value_cols + ['state']].sort_values(value_cols[0]).head(20))
    print("Best value_0 among COMPLETE trials:", df_trials[df_trials['state'] == 'COMPLETE'][value_cols[0]].min())
    print("Best value_0 among ALL trials:", df_trials[value_cols[0]].min())
    if hasattr(feature_hyperparam_study, 'best_trials') and feature_hyperparam_study.best_trials:
        print(f"Final best value_0: {feature_hyperparam_study.best_trials[0].values[0]:.5f}")
else:
    print(df_trials[['number', 'value', 'state']].sort_values('value').head(20))
    complete_trials = df_trials[df_trials['state'] == 'COMPLETE']
    if not complete_trials.empty and complete_trials['value'].notnull().any():
        print("Best value among COMPLETE trials:", complete_trials['value'].min())
        print(f"Final best value: {complete_trials['value'].min():.5f}")
    else:
        print("No COMPLETE trials found. Cannot report best value.")
    print("Best value among ALL trials:", df_trials['value'].min())
    print(f"Final best value: {feature_hyperparam_study.best_value:.5f}")

# Extract best features and params, but handle missing best_trial gracefully

# --- Patch: Always use the best trial by value, regardless of state (COMPLETE or PRUNED) ---
is_multi_objective = isinstance(feature_hyperparam_study.directions, list) and len(feature_hyperparam_study.directions) > 1

# Use a weighted sum of all objectives for best trial selection (adjust weights as needed)
# These weights should match the intent of your objective function.
# For example, if you want to penalize overfitting more, increase the weight for gap_penalty, etc.
# Already set and weighted in the objective function
objective_weights = [1.0, 1.0, 1.0, 1.0]  # [mean_valid, gap_penalty, complexity_penalty, -reg_reward]
def get_weighted_objective(trial):
    if is_multi_objective and trial.values is not None:
        # Defensive: Only use as many weights as there are objectives
        n_obj = min(len(objective_weights), len(trial.values))
        return sum(objective_weights[i] * trial.values[i] for i in range(n_obj))
    elif not is_multi_objective and trial.value is not None:
        return trial.value
    else:
        return float('inf')  # Exclude invalid trials

trials_with_value = [
    t for t in feature_hyperparam_study.get_trials(deepcopy=False)
    if (
        (t.values is not None and not any(pd.isnull(v) or np.isinf(v) for v in t.values)) if is_multi_objective
        else (t.value is not None and not (pd.isnull(t.value) or np.isinf(t.value)))
    )
]

if not trials_with_value:
    logging.warning("No Optuna trial with a value found. Skipping feature/param extraction.")
    SELECTED_FEATURES = FEATURES.copy()
    # No tuned noise parameters
    tuned_noise_target_level = 0.0
    tuned_noise_feature_level = 0.0
    tuned_bootstrap_frac = 0.0
    best_params = final_params.copy()
else:
    best_trial = min(trials_with_value, key=get_weighted_objective)
    best_value = get_weighted_objective(best_trial)
    print(f"Best trial number: {best_trial.number}, weighted value: {best_value}, state: {best_trial.state}")
    best_mask = [best_trial.params.get(f, False) for f in FEATURES]
    SELECTED_FEATURES = [f for f, keep in zip(FEATURES, best_mask) if keep]
    # --- Add selected interaction features from best_trial ---
    interaction_features = []
    for k, v in best_trial.params.items():
        if k.startswith('inter_') and v is not None:
            combo = v.split('|')
            # Product
            new_col_prod = '_prod_'.join(combo)
            interaction_features.append(new_col_prod)
            # Additive
            new_col_add = '_add_'.join(combo)
            interaction_features.append(new_col_add)
            # Ratio (only for pairs)
            if len(combo) == 2:
                new_col_div = '_div_'.join(combo)
                interaction_features.append(new_col_div)
                # Polynomial: feature1 * feature2 ** 2
                new_col_poly2 = f'{combo[0]}_poly2_{combo[1]}'
                interaction_features.append(new_col_poly2)
    # Add only unique and not already present
    for f in interaction_features:
        if f not in SELECTED_FEATURES:
            SELECTED_FEATURES.append(f)
    # Extract LightGBM params and noise parameters
    raw_params = {k: v for k, v in best_trial.params.items() if k not in FEATURES and not k.endswith('_pair') and not k.startswith('inter_')}
    # Tuned noise parameters
    tuned_noise_target_level = raw_params.pop('noise_target_level', 0.0)
    tuned_noise_feature_level = raw_params.pop('noise_feature_level', 0.0)
    tuned_bootstrap_frac = raw_params.pop('bootstrap_frac', 0.0)
    best_params = raw_params
    logging.info(f"Optuna-selected params: {best_params}")
    logging.info(f"Optuna-selected features: ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    if best_params.get('boosting_type') == 'goss':
        best_params['bagging_fraction'] = 1.0
        best_params['bagging_freq'] = 0
    selected_pairs = {k: v for k, v in best_trial.params.items() if k.endswith('_pair')}
    # Add both features from each selected cyclical pair
    for pair_name, is_selected in selected_pairs.items():
        if is_selected:
            pair_feats = pair_name[:-5].split('_')
            for i in range(1, len(pair_feats)):
                if pair_feats[i].endswith('sin') or pair_feats[i].endswith('cos'):
                    feat1 = '_'.join(pair_feats[:i+1])
                    feat2 = '_'.join(pair_feats[i+1:])
                    for feat in [feat1, feat2]:
                        if feat and feat in train_df.columns and feat not in SELECTED_FEATURES:
                            SELECTED_FEATURES.append(feat)
                    break
    logging.info(f"Optuna-selected params: {best_params}")
    logging.info(f"Optuna-selected features: ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

# --- Ensure all dynamic interaction features exist in train/valid/test DataFrames ---
def ensure_interaction_features(df, feature_names):
    for f in feature_names:
        # Product interactions
        if '_prod_' in f and f not in df.columns:
            parts = f.split('_prod_')
            if all(p in df.columns for p in parts):
                col_val = df[parts[0]]
                for p in parts[1:]:
                    col_val = col_val * df[p]
                df[f] = col_val
            else:
                df[f] = 0
        # Additive interactions
        elif '_add_' in f and f not in df.columns:
            parts = f.split('_add_')
            if all(p in df.columns for p in parts):
                col_val = df[parts[0]]
                for p in parts[1:]:
                    col_val = col_val + df[p]
                df[f] = col_val
            else:
                df[f] = 0
        # Ratio interactions
        elif '_div_' in f and f not in df.columns:
            parts = f.split('_div_')
            if len(parts) == 2 and all(p in df.columns for p in parts):
                denominator = df[parts[1]].replace(0, np.nan)
                df[f] = df[parts[0]] / denominator + 1e-15  # Avoid division by zero
                df[f] = df[f].replace([np.inf, -np.inf], 0)
            else:
                df[f] = 0
        # Polynomial interactions: feature1 * feature2 ** 2
        elif '_poly2_' in f and f not in df.columns:
            parts = f.split('_poly2_')
            if len(parts) == 2 and all(p in df.columns for p in parts):
                df[f] = df[parts[0]] * (df[parts[1]] ** 2)
            else:
                df[f] = 0
    return df

# Apply to all relevant DataFrames
train_df = ensure_interaction_features(train_df, SELECTED_FEATURES)
valid_df = ensure_interaction_features(valid_df, SELECTED_FEATURES)
test_df = ensure_interaction_features(test_df, SELECTED_FEATURES)

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
        # Ensure dynamic interaction features are present
        history_df = ensure_interaction_features(history_df, FEATURES)
        current_features = history_df.loc[current_week_mask, FEATURES]
        missing_features = [col for col in FEATURES if col not in current_features.columns]
        if missing_features:
            logging.warning(f"Missing features in prediction (week {week_num}): {missing_features}")
        for col in missing_features:
            current_features[col] = 0
        current_preds = np.clip(model.predict(current_features[FEATURES]), 0, None).round().astype(float)
        history_df.loc[current_week_mask, 'num_orders'] = current_preds
    final_predictions = history_df.loc[history_df['id'].isin(predict_df['id']), ['id', 'num_orders']].copy()
    final_predictions['num_orders'] = final_predictions['num_orders'].round().astype(int)
    final_predictions['id'] = final_predictions['id'].astype(int)
    return final_predictions.set_index('id')['num_orders']

def recursive_ensemble(train_df, test_df, FEATURES, weekofyear_means=None, month_means=None, n_models=N_ENSEMBLE_MODELS, eval_metric=None, noise_target_level=0, noise_feature_level=0, bootstrap_frac=0.0):
    preds_list = []
    models = []
    for i in tqdm(range(n_models), desc="Ensemble Models", position=0):
        logging.info(f"Training ensemble model {i+1}/{n_models}...")
        params = final_params.copy(); params.pop('seed', None)

        # Inject noise into training data for robustness
        X_train, y_train = add_training_noise(
            train_df, FEATURES, TARGET,
            noise_target_level=noise_target_level,
            noise_feature_level=noise_feature_level,
            bootstrap_frac=bootstrap_frac,
            seed=SEED + i,
            group_cols=GROUP_COLS
        )
        
        model = LGBMRegressor(**params, seed=SEED+i)
        
        model.fit(
            X_train, y_train,
            eval_set=[(valid_df[FEATURES], valid_df[TARGET])],
            eval_metric=eval_metric,
            callbacks=[early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False)],
            categorical_feature=[col for col in CATEGORICAL_FEATURES if col in FEATURES]
        )
        
        # After model is trained, get predictions for the test set without noise
        preds_list.append(recursive_predict(model, train_df, test_df, FEATURES, weekofyear_means, month_means).values)
        models.append(model)
    
    return np.mean(preds_list, axis=0).round().astype(int), models

# --- Recursive Ensemble Prediction with Selected Features ---
logging.info("Running recursive ensemble prediction with selected features...")
ensemble_preds, ensemble_models = recursive_ensemble(
    train_df, test_df, FEATURES, weekofyear_means, month_means,
    n_models=N_ENSEMBLE_MODELS,
    eval_metric=rmsle_lgbm,
    noise_target_level=tuned_noise_target_level,
    noise_feature_level=tuned_noise_feature_level,
    bootstrap_frac=tuned_bootstrap_frac
)
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
    # Ensure categorical features in SHAP sample match training categories
    shap_sample_for_shap = shap_sample.copy()
    for col in CATEGORICAL_FEATURES:
        if col in shap_sample_for_shap.columns and col in train_df.columns:
            if pd.api.types.is_categorical_dtype(train_df[col]):
                shap_sample_for_shap[col] = shap_sample_for_shap[col].astype('category')
                shap_sample_for_shap[col] = shap_sample_for_shap[col].cat.set_categories(train_df[col].cat.categories)
    explainer = shap.TreeExplainer(ensemble_models[0])
    shap_values = explainer.shap_values(shap_sample_for_shap[FEATURES])
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
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_summary_all_features.png"))
    plt.close()
    plt.figure(figsize=(10, 8))
    shap_importance_df.head(20).plot(kind='barh', x='feature', y='mean_abs_shap', legend=False, figsize=(10, 8))
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
    plt.title('Top 20 SHAP Feature Importances (Ensemble)')
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_top20_importance.png"))
    plt.close()
    for feat in shap_importance_df['feature']:
        shap.dependence_plot(feat, shap_values, shap_sample[FEATURES], show=False)
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{SHAP_FILE_PREFIX}_final_optuna_ensemble_dependence_{feat}.png"))
        plt.close()
    try:
        shap_interaction_values = explainer.shap_interaction_values(shap_sample[FEATURES])
        plt.figure()
        shap.summary_plot(shap_interaction_values, shap_sample[FEATURES], show=False, max_display=20)
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
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, "validation_actual_vs_predicted_ensemble_recursive.png"))
    plt.close()
    logging.info("Validation plot (recursive) saved.")
except Exception as e:
    logging.error(f"Error during plotting (recursive): {e}")

logging.info("All tasks completed.")

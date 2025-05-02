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
from optuna.samplers import NSGAIISampler, TPESampler
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
SEED = 42
LAG_WEEKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ROLLING_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28]
N_ENSEMBLE_MODELS = 5
OVERFIT_ROUNDS = 16 # Overfitting detection rounds
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
N_WARMUP_STEPS = 150 # Warmup steps for Optuna pruning
POPULATION_SIZE = 32 # Population size for Genetic algorithm
#OPTUNA_SAMPLER = "Default"
OPTUNA_SAMPLER = "NSGAIISampler"
PRUNING_ENABLED = False # Enable Optuna pruning
OPTUNA_TRIALS = 1000000 # Number of Optuna trials (increased for better search)
OPTUNA_TIMEOUT = 60 * 60 * 24 # Timeout for Optuna trials (in seconds)
OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
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

def add_binary_rolling_means(df, binary_cols=["emailer_for_promotion", "homepage_featured"], windows=LAG_WEEKS):
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
    df_out = add_binary_rolling_means(df_out, ["emailer_for_promotion", "homepage_featured"], LAG_WEEKS)
    df_out = create_group_aggregates(df_out)
    df_out, weekofyear_means, month_means = add_seasonality_features(df_out, weekofyear_means, month_means, is_train=is_train)
    # Fill NaNs for all engineered features
    lag_roll_diff_cols = [col for col in df_out.columns if any(sub in col for sub in [
        "lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_mean", "_std"
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
for lag in [1, 2, 3, 5, 10]:
    lag_col = f"{TARGET}_lag_{lag}"
    if lag_col in train_df.columns and lag_col not in features_set:
        FEATURES.append(lag_col)
        features_set.add(lag_col)

# Add rolling means/stds
for w in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28]:
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
    for w in LAG_WEEKS:
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
            # return optuna.TrialPruned()
            raise lgb.callback.EarlyStopException(env.iteration, best_score[0])
        # Standard early stopping
        if env.iteration - best_iter[0] >= stopping_rounds:
            if verbose:
                print(f"Stopping early due to no improvement at iteration {env.iteration}")
            # return optuna.TrialPruned()
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

class RollingGroupTimeSeriesSplit:
    """
    Rolling window cross-validator for time series data with group awareness.
    For each split, the training set is a rolling window of train_window unique weeks,
    and the validation set is the next val_window unique weeks.
    Groups are respected (e.g., center_id, meal_id).
    No gap is used between train and validation.
    """
    def __init__(self, n_splits=3, train_window=20, val_window=4, week_col='week'):
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
MAX_INTERACTION_ORDER = 2 # Max order of interactions to consider (2nd order = pairwise, 3rd order = triplet, etc.)
MAX_INTERACTIONS_PER_ORDER = {2: 3, 3: 2, 4: 0, 5: 0}
ALL_COMBOS_STR = {}

for order in range(2, MAX_INTERACTION_ORDER + 1):
    combos = list(combinations(FROZEN_FEATURES_FOR_INTERACTIONS, order))
    ALL_COMBOS_STR[order] = ["|".join(str(f) for f in combo) for combo in combos]

# Defensive: ensure all elements are strings (Optuna requires this for categorical choices)
for order, combos in ALL_COMBOS_STR.items():
    for c in combos:
        assert isinstance(c, str), f"ALL_COMBOS_STR[{order}] contains non-string: {c} ({type(c)})"

def optuna_feature_selection_and_hyperparam_objective(trial, train_split_df=train_split_df):
    # Hyperparameter search space
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    if boosting_type != 'goss':
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.8, 1.0)  # ↑ Min value for more regularization
        bagging_freq = trial.suggest_int('bagging_freq', 0, 10)
    else:
        bagging_fraction = 1.0
        bagging_freq = 0
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True), # Lower for less overfit
        'num_leaves': trial.suggest_int('num_leaves', 4, 128), # Lower for less complexity
        'max_depth': trial.suggest_int('max_depth', 2, 12),    # Lower for less complexity
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),  # ↑ Min value
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 1000), # ↑ Min value
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 100.0, log=True), # ↑ Min value, must be >0 for log
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 100.0, log=True), # ↑ Min value, must be >0 for log
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0), 
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000), # ↑ Min value
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 500000),
        'boosting_type': boosting_type,
        'max_bin': trial.suggest_int('max_bin', 32, 512), # Lower for regularization
        'objective': 'regression_l1',
        'n_estimators': 500,
        'seed': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'metric': 'rmsle',
    }
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

    # --- Robust dynamic feature interaction logic ---
    interaction_features = []
    new_interaction_cols = {}
    used_interactions = set()
    # Dynamically generate all pairwise and higher-order products from eligible features
    # (no single features, only interactions)
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
            new_col = "_prod_".join(combo)
            if new_col not in train_split_df.columns and new_col not in new_interaction_cols:
                col_val = train_split_df[combo[0]]
                for f in combo[1:]:
                    col_val = col_val * train_split_df[f]
                new_interaction_cols[new_col] = col_val
            interaction_features.append(new_col)
    if new_interaction_cols:
        train_split_df = pd.concat([train_split_df, pd.DataFrame(new_interaction_cols, index=train_split_df.index)], axis=1)
    selected_features += interaction_features

    # Ensure selected features are unique and not empty
    selected_features = list(dict.fromkeys(selected_features))
    if len(selected_features) < 10:
        logging.warning(f"Optuna selected {len(selected_features)} features, which is less than 10. This may lead to overfitting.")
        return optuna.TrialPruned()

    # Use rolling window group time series split
    rgs = RollingGroupTimeSeriesSplit(n_splits=3, train_window=20, val_window=4, week_col='week')
    groups = train_split_df["center_id"]
    train_scores, valid_scores = [], []
    if not PRUNING_ENABLED or OPTUNA_SAMPLER == "NSGAIISampler":
        callbacks = [
            early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=True)
        ]  # No pruning callback
    else:
        callbacks = [
            LightGBMPruningCallback(trial, metric='rmsle', valid_name='valid_1'),
            early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=True)
        ]

    for train_idx, valid_idx in rgs.split(train_split_df, groups=groups):
        model = LGBMRegressor(**params)
        model.fit(
            train_split_df.iloc[train_idx][selected_features],
            train_split_df.iloc[train_idx][TARGET],
            eval_set=[
                (train_split_df.iloc[train_idx][selected_features], train_split_df.iloc[train_idx][TARGET]),
                (train_split_df.iloc[valid_idx][selected_features], train_split_df.iloc[valid_idx][TARGET])
            ],
            eval_metric=rmsle_lgbm,
            callbacks=callbacks
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
    gap_penalty = max(0, generalization_gap) * 2.0  # Weight can be tuned. Higher weight = more penalty for overfitting
    complexity_penalty = 0.01 * len(selected_features)
    complexity_penalty += 0.01 * params['num_leaves']
    complexity_penalty += 0.01 * (params['max_depth'] if params['max_depth'] > 0 else 0)
    reg_reward = 0.01 * (params['lambda_l1'] + params['lambda_l2'])

    # Store metrics for logging in callback
    trial.set_user_attr('mean_train', float(mean_train))
    trial.set_user_attr('mean_valid', float(mean_valid))
    trial.set_user_attr('generalization_gap', float(generalization_gap))
    trial.set_user_attr('gap_penalty', float(gap_penalty))
    trial.set_user_attr('complexity_penalty', float(complexity_penalty))
    trial.set_user_attr('reg_reward', float(reg_reward))

    # Multi-objective: minimize mean_valid, gap_penalty, complexity_penalty, maximize reg_reward (so minimize -reg_reward)
    # If any objective is nan/inf, prune
    objectives = [mean_valid, gap_penalty, complexity_penalty, -reg_reward]
    if any(np.isnan(obj) or np.isinf(obj) for obj in objectives):
        logging.warning(f"Optuna trial {trial.number} produced invalid objectives: {objectives}.")
        if isinstance(directions, list) and len(directions) > 1:
            return tuple([float('inf')] * len(directions))
        else:
            return float('inf')

    # Return correct type for single- or multi-objective
    if isinstance(directions, list) and len(directions) > 1:
        return tuple(objectives)
    else:
        return objectives[0]

logging.info("Starting Optuna feature+hyperparam selection...")

# Reduce Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

class TqdmOptunaCallback:
    def __init__(self, n_trials, study=None, print_every=1):
        self.n_trials = n_trials
        self.print_every = print_every
        self.study = study
        self.pbar = tqdm(total=n_trials, desc="Optuna Trials", position=0, leave=True)

    def __call__(self, study, trial):
        self.pbar.update(1)
        n_features = sum([v for k, v in trial.params.items() if isinstance(v, bool) and v])
        num_leaves = trial.params.get('num_leaves', None)
        max_depth = trial.params.get('max_depth', None)
        lambda_l1 = trial.params.get('lambda_l1', None)
        lambda_l2 = trial.params.get('lambda_l2', None)
        mean_train = trial.user_attrs.get('mean_train', None)
        mean_valid = trial.user_attrs.get('mean_valid', None)
        generalization_gap = trial.user_attrs.get('generalization_gap', None)
        msg = (
            f"Trial {trial.number} | mean_valid: {mean_valid} | gap: {generalization_gap} | "
            f"Features: {n_features} | num_leaves: {num_leaves} | max_depth: {max_depth} | "
            f"lambda_l1: {lambda_l1} | lambda_l2: {lambda_l2}"
        )
        tqdm.write(msg)

        # Live ASCII plot using plotext, with progress bar handling
        if trial.number % self.print_every == 0:
            self.pbar.close()  # Close progress bar before plotting
            self.live_plot_objectives(study.trials)
            self.pbar = tqdm(total=self.n_trials, desc="Optuna Trials", position=0, leave=True)
            self.pbar.n = trial.number + 1  # Restore progress
            self.pbar.refresh()

    def live_plot_objectives(self, trials):
        pltx.clf()
        trial_nums = []
        mean_valids = []
        gap_penalties = []
        complexity_penalties = []
        reg_rewards = []
        for t in trials:
            if t.values is not None and len(t.values) >= 4:
                trial_nums.append(t.number)
                mean_valids.append(t.values[0])
                gap_penalties.append(t.values[1])
                complexity_penalties.append(t.values[2])
                reg_rewards.append(-t.values[3])  # Negated because you minimize -reg_reward
        if trial_nums:
            # Use darker, less bright colors for plotext
            pltx.plot(trial_nums, mean_valids, label='mean_valid', color='cyan')
            pltx.plot(trial_nums, gap_penalties, label='gap_penalty', color='magenta')
            pltx.plot(trial_nums, complexity_penalties, label='complexity_penalty', color='blue')
            pltx.plot(trial_nums, reg_rewards, label='reg_reward', color='green')
            pltx.title('Optuna Objectives (Live)')
            pltx.xlabel('Trial')
            pltx.ylabel('Value')
            pltx.canvas_color('black')
            pltx.axes_color('black')
            pltx.ticks_color('grey')
            # pltx.grid_color('grey')
            pltx.grid(True)
            pltx.show()

    def close(self):
        self.pbar.close()

# class TqdmOptunaCallback:
#     def __init__(self, n_trials, study=None, print_every=1):
#         self.pbar = tqdm(total=n_trials, desc="Optuna Trials", position=0, leave=True)
#         self.print_every = print_every
#         # Initialize best_value and best_trial from study if available
#         if study is not None:
#             try:
#                 if study.best_trial is not None and study.best_trial.value is not None:
#                     self.best_value = study.best_trial.value
#                     self.best_trial = study.best_trial.number
#                 else:
#                     self.best_value = float('inf')
#                     self.best_trial = None
#             except Exception:
#                 self.best_value = float('inf')
#                 self.best_trial = None
#         else:
#             self.best_value = float('inf')
#             self.best_trial = None
#     def __call__(self, study, trial):
#         self.pbar.update(1)
#         # Extract best trial info
#         best_trial = study.best_trial if study.best_trial is not None else None
#         best_trial_num = best_trial.number if best_trial is not None else None
#         best_trial_val = best_trial.value if best_trial is not None else None

#         # Extract this trial's info
#         trial_num = trial.number
#         trial_val = trial.value
#         params = trial.params

#         # Extract scores and gap from trial's user_attrs (set in objective)
#         mean_train = trial.user_attrs.get('mean_train', None)
#         mean_valid = trial.user_attrs.get('mean_valid', None)
#         generalization_gap = trial.user_attrs.get('generalization_gap', None)

#         # Number of selected features
#         n_features = sum([v for k, v in params.items() if isinstance(v, bool) and v])

#         # Main hyperparameters
#         num_leaves = params.get('num_leaves', None)
#         max_depth = params.get('max_depth', None)
#         lambda_l1 = params.get('lambda_l1', None)
#         lambda_l2 = params.get('lambda_l2', None)

#         # Format values for pretty printing
#         def fmt(val, prec=5):
#             if val is None:
#                 return 'None'
#             if isinstance(val, float):
#                 return f"{val:.{prec}f}"
#             return str(val)

#         msg = (
#             f"Trial {trial_num} | Obj: {fmt(trial_val)} | "
#             f"Train: {fmt(mean_train)} | Valid: {fmt(mean_valid)} | Gap: {fmt(generalization_gap)} | "
#             f"Features: {n_features} | num_leaves: {num_leaves} | max_depth: {max_depth} | "
#             f"lambda_l1: {fmt(lambda_l1,3)} | lambda_l2: {fmt(lambda_l2,3)} | "
#             f"Best so far: #{best_trial_num} ({fmt(best_trial_val)})"
#         )
#         tqdm.write(msg)
#     def close(self):
#         self.pbar.close()

# Create the study 

optuna_storage = OPTUNA_DB

if OPTUNA_SAMPLER == "NSGAIISampler":
    sampler = NSGAIISampler(
        seed=SEED,
        population_size=POPULATION_SIZE,
        crossover=UniformCrossover(),
        crossover_prob=0.9,
        swapping_prob=0.5,
    )
    pruner = optuna.pruners.NopPruner()  # Pruning not supported for multi-objective
    directions = ["minimize", "minimize", "minimize", "minimize"]  # mean_valid, gap_penalty, complexity_penalty, -reg_reward
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
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=N_WARMUP_STEPS)
    directions = "minimize"  # Single-objective

feature_hyperparam_study = optuna.create_study(
    directions=directions,
    pruner=pruner,
    study_name=OPTUNA_STUDY_NAME,
    storage=optuna_storage,
    load_if_exists=True,
    sampler=sampler
)

# Pass the study to the callback so it can initialize best_value/best_trial

optuna_callback = TqdmOptunaCallback(OPTUNA_TRIALS, study=feature_hyperparam_study, print_every=1)
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
    print("Best value among COMPLETE trials:", df_trials[df_trials['state'] == 'COMPLETE']['value'].min())
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
            # v is the combo string, e.g. 'discount_pct|price_diff|meal_orders_mean'
            combo = v.split('|')
            new_col = '_prod_'.join(combo)
            interaction_features.append(new_col)
    # Add only unique and not already present
    for f in interaction_features:
        if f not in SELECTED_FEATURES:
            SELECTED_FEATURES.append(f)
    best_params = {k: v for k, v in best_trial.params.items() if k not in FEATURES and not k.endswith('_pair') and not k.startswith('inter_')}
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
        if '_prod_' in f and f not in df.columns:
            parts = f.split('_prod_')
            # Defensive: only create if all base features exist
            if all(p in df.columns for p in parts):
                col_val = df[parts[0]]
                for p in parts[1:]:
                    col_val = col_val * df[p]
                df[f] = col_val
            else:
                # If any part is missing, fill with zeros (fail-safe)
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

def recursive_ensemble(train_df, test_df, FEATURES, weekofyear_means=None, month_means=None, n_models=N_ENSEMBLE_MODELS, eval_metric=None):
    preds_list = []
    models = []
    for i in tqdm(range(n_models), desc="Ensemble Models", position=0):
        logging.info(f"Training ensemble model {i+1}/{n_models}...")
        params = final_params.copy(); params.pop('seed', None)
        # Optionally, add a small amount of noise to the training targets for robustness
        # train_y = train_df[TARGET] + np.random.normal(0, 0.01 * train_df[TARGET].std(), size=len(train_df))
        train_y = train_df[TARGET]
        model = LGBMRegressor(**params, seed=SEED+i)
        if eval_metric:
            model.fit(
                train_df[FEATURES], train_y,
                eval_set=[(train_df[FEATURES], train_y), (valid_df[FEATURES], valid_df[TARGET])],
                eval_metric=eval_metric,
                callbacks=[early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False)],
                categorical_feature=CATEGORICAL_FEATURES
            )
        else:
            model.fit(train_df[FEATURES], train_y, categorical_feature=CATEGORICAL_FEATURES)
        preds_list.append(recursive_predict(model, train_df, test_df, FEATURES, weekofyear_means, month_means).values)
        models.append(model)
    return np.mean(preds_list, axis=0).round().astype(int), models

# --- Recursive Ensemble Prediction with Selected Features ---
logging.info("Running recursive ensemble prediction with selected features...")
ensemble_preds, ensemble_models = recursive_ensemble(train_df, test_df, FEATURES, weekofyear_means, month_means, n_models=N_ENSEMBLE_MODELS, eval_metric=rmsle_lgbm)
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

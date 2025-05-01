import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from functools import partial

import os
OUTPUT_DIRECTORY = "output"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

import re
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb  # Added for early stopping callback
import optuna
import shap
from tqdm import tqdm
import logging
import csv
import random
from optuna.integration import LightGBMPruningCallback

import warnings
warnings.filterwarnings("ignore", message="The reported value is ignored because this `step` .* is already reported.")

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28]
N_ENSEMBLE_MODELS = 5
OVERFIT_ROUNDS = 16 # Overfitting detection rounds
VALIDATION_WEEKS = 8 # Use last 8 weeks for validation
N_WARMUP_STEPS = 30 # Warmup steps for Optuna pruning
POPULATION_SIZE = 32 # Population size for Genetic algorithm
OPTUNA_TRIALS = 1000000 # Number of Optuna trials (increased for better search)
OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
SUBMISSION_FILE_PREFIX = "submission_recursive"
SHAP_FILE_PREFIX = "shap_recursive"
N_SHAP_SAMPLES = 2000

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

def create_selected_interaction_features(df):
    df_out = df.copy()
    if 'price_diff' in df_out.columns and 'emailer_for_promotion' in df_out.columns:
        df_out['price_diff_x_emailer'] = df_out['price_diff'] * df_out['emailer_for_promotion']
    if 'price_diff' in df_out.columns and 'homepage_featured' in df_out.columns:
        df_out['price_diff_x_homepage'] = df_out['price_diff'] * df_out['homepage_featured']
    if 'discount_pct' in df_out.columns and 'emailer_for_promotion' in df_out.columns:
        df_out['discount_pct_x_emailer'] = df_out['discount_pct'] * df_out['emailer_for_promotion']
    if 'discount_pct' in df_out.columns and 'homepage_featured' in df_out.columns:
        df_out['discount_pct_x_homepage'] = df_out['discount_pct'] * df_out['homepage_featured']
    if 'meal_orders_mean' in df_out.columns and 'discount_pct' in df_out.columns:
        df_out['meal_orders_mean_x_discount_pct'] = df_out['meal_orders_mean'] * df_out['discount_pct']
    if 'meal_orders_mean' in df_out.columns and 'emailer_for_promotion' in df_out.columns:
        df_out['meal_orders_mean_x_emailer_for_promotion'] = df_out['meal_orders_mean'] * df_out['emailer_for_promotion']
    return df_out

# def add_promo_recency_features(df):
#     df_out = df.copy()
#     group_cols = [col for col in ["center_id", "meal_id"] if col in df_out.columns]
#     for promo_col, name in [("emailer_for_promotion", "weeks_since_last_emailer"), ("homepage_featured", "weeks_since_last_homepage")]:
#         if promo_col in df_out.columns:
#             def weeks_since_last(x):
#                 last = -1
#                 out = []
#                 for i, val in enumerate(x):
#                     if val:
#                         last = i
#                     out.append(i - last if last != -1 else np.nan)
#                 return out
#             df_out[name] = df_out.groupby(group_cols, observed=False)[promo_col].transform(weeks_since_last)
#     return df_out

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

def add_binary_rolling_means(df, binary_cols=["emailer_for_promotion", "homepage_featured"], windows=[3, 5, 10]):
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
    df_out = add_binary_rolling_means(df_out, ["emailer_for_promotion", "homepage_featured"], [3, 5, 10])
    df_out = create_group_aggregates(df_out)
    df_out = create_selected_interaction_features(df_out)
    # df_out = add_promo_recency_features(df_out)
    df_out, weekofyear_means, month_means = add_seasonality_features(df_out, weekofyear_means, month_means, is_train=is_train)
    # Fill NaNs for all engineered features
    lag_roll_diff_cols = [col for col in df_out.columns if any(sub in col for sub in [
        "lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home",
        "_x_discount_pct", "_x_price_diff", "_x_weekofyear", "_mean", "_std"
        # "_x_discount_pct", "_x_price_diff", "_x_weekofyear", "_mean", "_std", "weeks_since_last_emailer", "weeks_since_last_homepage"
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
    for w in [3, 5, 10]:
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


# Add selected interaction features
for col in [
    "price_diff_x_emailer", "price_diff_x_homepage", "discount_pct_x_emailer", "discount_pct_x_homepage",
    "meal_orders_mean_x_discount_pct", "meal_orders_mean_x_emailer_for_promotion"
]:
    if col in train_df.columns and col not in features_set:
        FEATURES.append(col)
        features_set.add(col)

# Add recency features
# for col in ["weeks_since_last_emailer", "weeks_since_last_homepage"]:
#     if col in train_df.columns and col not in features_set:
#         FEATURES.append(col)
#         features_set.add(col)

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

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False # lower is better

# # --- Custom LightGBM RMSLE metric for Optuna integration ---
# def rmsle_lgbm(y_pred, dataset):
#     """Custom RMSLE metric for LightGBM (Optuna callback compatible)"""
#     y_true = dataset.get_label()
#     y_pred = np.clip(y_pred, 0, None)
#     rmsle_score = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
#     return 'rmsle', rmsle_score, False

# --- Custom LightGBM RMSLE metric for sklearn API (LGBMRegressor) ---
def rmsle_lgbm(y_true, y_pred):
    """Custom RMSLE metric for LightGBM sklearn API (Optuna callback compatible)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)
    rmsle_score = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    return 'rmsle', rmsle_score, False

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
    'min_data_in_leaf': 150,
    'num_leaves': 16,
    'max_depth': 4,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3
}

# --- Empirical Overfitting Patience Analysis (run early) ---
def analyze_overfitting_patience(train_df, valid_df, FEATURES, TARGET, params, max_rounds=300, plot_path=None):
    """
    Empirically analyze the longest consecutive streak where validation loss increases and training loss decreases.
    Suggest a patience value for overfitting detection. Optionally plot loss curves.
    Returns: (recommended_patience, train_loss, valid_loss)
    """
    params = params.copy()
    n_estimators = params.pop('n_estimators', max_rounds)
    model = LGBMRegressor(**params, n_estimators=n_estimators) 
    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(train_df[FEATURES], train_df[TARGET]), (valid_df[FEATURES], valid_df[TARGET])],
        eval_metric='l1'
    )
    evals_result = model.evals_result_
    keys = list(evals_result.keys())
    if 'validation_0' in keys and 'validation_1' in keys:
        train_loss = evals_result['validation_0']['l1']
        valid_loss = evals_result['validation_1']['l1']
    elif 'train' in keys and 'valid' in keys:
        train_loss = evals_result['train']['l1']
        valid_loss = evals_result['valid']['l1']
    elif 'valid_0' in keys and 'valid_1' in keys:
        train_loss = evals_result['valid_0']['l1']
        valid_loss = evals_result['valid_1']['l1']
    else:
        print(f"Unexpected evals_result_ keys: {keys}")
        raise ValueError(f"Could not find expected keys in evals_result_: {keys}")
    streak = 0
    max_streak = 0
    for i in range(1, len(train_loss)):
        if valid_loss[i] > valid_loss[i-1] and train_loss[i] < train_loss[i-1]:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"Longest overfitting streak: {max_streak} rounds.")
    recommended_patience = max(5, max_streak + 2)
    print(f"Recommended overfitting patience: {recommended_patience} rounds.")
    # Plot loss curves if requested
    if plot_path:
        plt.figure(figsize=(8,5))
        plt.plot(train_loss, label='Train L1 Loss')
        plt.plot(valid_loss, label='Valid L1 Loss')
        plt.xlabel('Boosting Round')
        plt.ylabel('L1 Loss')
        plt.title('Train/Validation Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    return recommended_patience, train_loss, valid_loss


def multi_seed_overfitting_patience_analysis(train_df, valid_df, FEATURES, TARGET, params, seeds=[208, 209, 210, 211, 212, 213, 214, 215, 216, 217], max_rounds=300):
    """
    Run the overfitting patience analysis for multiple seeds and save results to CSV and plot.
    Also plot all loss curves on the same chart.
    """
    patience_results = []
    all_train_losses = []
    all_valid_losses = []
    for seed in seeds:
        params_seed = params.copy()
        params_seed['seed'] = seed
        np.random.seed(seed)
        random.seed(seed)
        plot_path = os.path.join(OUTPUT_DIRECTORY, f"loss_curve_seed_{seed}.png")
        try:
            patience, train_loss, valid_loss = analyze_overfitting_patience(train_df, valid_df, FEATURES, TARGET, params_seed, max_rounds=max_rounds, plot_path=plot_path)
            patience_results.append((seed, patience))
            all_train_losses.append((seed, train_loss))
            all_valid_losses.append((seed, valid_loss))
        except Exception as e:
            print(f"Patience analysis failed for seed {seed}: {e}")
            patience_results.append((seed, None))
    # Save to CSV
    output_csv = os.path.join(OUTPUT_DIRECTORY, "overfitting_patience_seeds.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Recommended_Patience"])
        for row in patience_results:
            writer.writerow(row)
    print(f"Multi-seed overfitting patience results saved to {output_csv}")
    # Plot patience values as boxplot
    valid_patiences = [p for s, p in patience_results if p is not None]
    if valid_patiences:
        plt.figure(figsize=(6,4))
        plt.boxplot(valid_patiences, vert=False)
        plt.title('Distribution of Recommended Patience (Multi-Seed)')
        plt.xlabel('Patience Rounds')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, "overfitting_patience_boxplot.png"))
        plt.close()
        print(f"Patience values: {valid_patiences}")
        print(f"Mean: {np.mean(valid_patiences):.2f}, Median: {np.median(valid_patiences):.2f}, Max: {np.max(valid_patiences)}, Min: {np.min(valid_patiences)}")
        print(f"Recommended patience (max): {np.max(valid_patiences)}")
    else:
        print("No valid patience values computed.")
    # Plot all loss curves on the same chart
    if all_train_losses and all_valid_losses:
        import matplotlib.cm as cm
        import matplotlib
        colors = matplotlib.colormaps.get_cmap('tab10')
        plt.figure(figsize=(10,6))
        for idx, (seed, train_loss) in enumerate(all_train_losses):
            plt.plot(train_loss, label=f'Train (seed={seed})', linestyle='--', alpha=0.7, color=colors(idx))
        for idx, (seed, valid_loss) in enumerate(all_valid_losses):
            plt.plot(valid_loss, label=f'Valid (seed={seed})', linewidth=2, color=colors(idx))
        plt.xlabel('Boosting Round')
        plt.ylabel('L1 Loss')
        plt.title('Train/Validation Loss Curves (All Seeds)')
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, "all_loss_curves_multi_seed.png"))
        plt.close()
    return patience_results

# --- Call multi-seed overfitting patience analysis after all dependencies are defined ---
# try:
#     multi_seed_overfitting_patience_analysis(train_split_df, valid_df, FEATURES, TARGET, final_params)
# except Exception as e:
#     logging.warning(f"Could not run multi-seed overfitting patience analysis: {e}")







# --- Empirical Rolling Window Check ---
# def empirical_rolling_window_check(df, features, window_sizes=[2, 3, 5, 7, 10], output_path="empirical_rolling_window_check.csv"):
#     """
#     For each feature and window size, compute rolling mean and sum, and report the percentage of non-zero (or non-constant) values.
#     Saves a CSV summary for review.
#     """
#     import pandas as pd
#     import numpy as np
#     results = []
#     group_cols = [col for col in ["center_id", "meal_id"] if col in df.columns]
#     for feat in features:
#         if feat not in df.columns:
#             continue
#         for window in window_sizes:
#             # Compute rolling mean and sum by group (if group columns exist)
#             if group_cols:
#                 rolled = df.groupby(group_cols, observed=False)[feat].rolling(window, min_periods=1).mean().reset_index(level=group_cols, drop=True)
#                 rolled_sum = df.groupby(group_cols, observed=False)[feat].rolling(window, min_periods=1).sum().reset_index(level=group_cols, drop=True)
#             else:
#                 rolled = df[feat].rolling(window, min_periods=1).mean()
#                 rolled_sum = df[feat].rolling(window, min_periods=1).sum()
#             # % non-zero for mean and sum
#             pct_nonzero_mean = (rolled != 0).mean() * 100
#             pct_nonzero_sum = (rolled_sum != 0).mean() * 100
#             # % unique for mean and sum
#             pct_unique_mean = (rolled.nunique() / len(rolled)) * 100
#             pct_unique_sum = (rolled_sum.nunique() / len(rolled_sum)) * 100
#             results.append({
#                 "feature": feat,
#                 "window": window,
#                 "pct_nonzero_mean": pct_nonzero_mean,
#                 "pct_nonzero_sum": pct_nonzero_sum,
#                 "pct_unique_mean": pct_unique_mean,
#                 "pct_unique_sum": pct_unique_sum,
#             })
#     df_results = pd.DataFrame(results)
#     df_results.to_csv(output_path, index=False)
#     print(f"Empirical rolling window check saved to {output_path}")

# --- Call empirical rolling window check after feature engineering ---
# empirical_rolling_window_check(train_df, ["emailer_for_promotion", "homepage_featured", "num_orders", "checkout_price", "discount_pct"], output_path="empirical_rolling_window_check.csv")











# --- Custom TimeSeriesSplit with Overlapping Validation Sets ---
# def custom_timeseries_split_with_overlap(df, n_splits=3, overlap_weeks=4):
#     """
#     Custom TimeSeriesSplit that creates overlapping validation sets.
#     For each split, the validation set starts before the end of the train set,
#     allowing for evaluation of the model's performance on the most recent data.
#     """
#     df = df.sort_values('week')
#     unique_weeks = df['week'].unique()
#     n_weeks = len(unique_weeks)
#     fold_size = n_weeks // n_splits
#     for i in range(n_splits):
#         val_start = max(0, (i * fold_size) - overlap_weeks)
#         val_end = (i + 1) * fold_size
#         if val_start >= val_end:
#             continue
#         train_idx = df[df['week'] < unique_weeks[val_start]].index
#         val_idx = df[(df['week'] >= unique_weeks[val_start]) & (df['week'] <= unique_weeks[val_end - 1])].index
#         yield train_idx, val_idx

# --- Custom GroupTimeSeriesSplit ---
class GroupTimeSeriesSplit:
    """
    Optimized cross-validator for time series data with non-overlapping groups.
    Precomputes group-to-indices mapping for faster splits.
    Each group appears in only one validation fold, and time order is respected.
    """
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Group labels must be provided for GroupTimeSeriesSplit.")
        groups = pd.Series(groups).reset_index(drop=True)
        # Precompute group to row indices mapping
        group_to_indices = {}
        for idx, group in enumerate(groups):
            group_to_indices.setdefault(group, []).append(idx)
        unique_groups = list(group_to_indices.keys())
        group_folds = np.array_split(unique_groups, self.n_splits)
        for fold_groups in group_folds:
            val_indices = []
            for g in fold_groups:
                val_indices.extend(group_to_indices[g])
            val_indices = np.array(val_indices)
            train_groups = set(unique_groups) - set(fold_groups)
            train_indices = []
            for g in train_groups:
                train_indices.extend(group_to_indices[g])
            train_indices = np.array(train_indices)
            # Sort indices by time if possible
            if hasattr(X, 'iloc') and 'week' in X.columns:
                train_indices = train_indices[np.argsort(X.iloc[train_indices]['week'].values)]
                val_indices = val_indices[np.argsort(X.iloc[val_indices]['week'].values)]
            yield train_indices, val_indices


# --- Rolling Window GroupTimeSeriesSplit ---
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

def compare_time_series_cv(train_df, FEATURES, TARGET, output_dir="output", seeds=[13, 123, 1999, 2025, 9001]):
    """
    Compare TimeSeriesSplit and GroupTimeSeriesSplit in detail, including per-fold metrics and plots, for multiple seeds.
    """
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    results = []
    all_folds = []
    for seed in seeds:
        np.random.seed(seed)
        cv_methods = {
            'TimeSeriesSplit': TimeSeriesSplit(n_splits=5),
            'GroupTimeSeriesSplit': GroupTimeSeriesSplit(n_splits=5),
            'RollingGroupTimeSeriesSplit': RollingGroupTimeSeriesSplit(n_splits=5, train_window=120, val_window=10, week_col='week')
        }
        X = train_df[FEATURES]
        # For RollingGroupTimeSeriesSplit, we need to include 'week' column
        if 'week' not in FEATURES:
            X_with_week = train_df[FEATURES + ['week']]
        else:
            X_with_week = X
        y = train_df[TARGET]
        groups = train_df['center_id'] if 'center_id' in train_df.columns else None
        for name, cv in cv_methods.items():
            print(f"\nTesting {name} (seed={seed})...")
            fold_metrics = []
            # Handle group requirements
            if name in ['GroupTimeSeriesSplit', 'RollingGroupTimeSeriesSplit'] and groups is None:
                print(f"No group labels available, skipping {name}.")
                continue
            if name == 'RollingGroupTimeSeriesSplit':
                splits = cv.split(X_with_week, y, groups=groups)
                X_used = X_with_week
            elif name == 'GroupTimeSeriesSplit':
                splits = cv.split(X, y, groups=groups)
                X_used = X
            else:
                splits = cv.split(X, y)
                X_used = X
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X_used.iloc[train_idx], X_used.iloc[val_idx]
                # Drop 'week' column for RollingGroupTimeSeriesSplit before fitting
                if name == 'RollingGroupTimeSeriesSplit' and 'week' in X_train.columns:
                    X_train = X_train.drop(columns=['week'])
                    X_val = X_val.drop(columns=['week'])
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = LGBMRegressor(n_estimators=2000, random_state=seed)
                model.fit(X_train, y_train, callbacks=[early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False)])
                y_pred = model.predict(X_val)
                rmsle = np.sqrt(mean_squared_log_error(np.maximum(0, y_val), np.maximum(0, y_pred)))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                fold_metrics.append({'seed': seed, 'cv': name, 'fold': fold+1, 'rmsle': rmsle, 'mae': mae, 'r2': r2})
                print(f"  Fold {fold+1}: RMSLE={rmsle:.4f}, MAE={mae:.2f}, R2={r2:.4f}")
            # Save results
            all_folds.extend(fold_metrics)
            results.append({
                'seed': seed,
                'cv': name,
                'mean_rmsle': np.mean([m['rmsle'] for m in fold_metrics]),
                'std_rmsle': np.std([m['rmsle'] for m in fold_metrics]),
                'mean_mae': np.mean([m['mae'] for m in fold_metrics]),
                'std_mae': np.std([m['mae'] for m in fold_metrics]),
                'mean_r2': np.mean([m['r2'] for m in fold_metrics]),
                'std_r2': np.std([m['r2'] for m in fold_metrics]),
                'folds': fold_metrics
            })
            # Plot per-fold RMSLE for this seed and CV type
            plt.plot([m['rmsle'] for m in fold_metrics], marker='o', label=f"{name} (seed={seed})")
    plt.xlabel('Fold')
    plt.ylabel('RMSLE')
    plt.title('Time Series CV Comparison: Fold RMSLEs (Multiple Seeds)')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'timeseries_cv_comparison_multiseed.png')
    plt.savefig(plot_path)
    plt.close()
    # Save results to CSV
    import pandas as pd
    pd.DataFrame(all_folds).to_csv(os.path.join(output_dir, 'timeseries_cv_comparison_folds_multiseed.csv'), index=False)
    pd.DataFrame(results).drop('folds', axis=1).to_csv(os.path.join(output_dir, 'timeseries_cv_comparison_summary_multiseed.csv'), index=False)
    print(f"\nDetailed results saved to {plot_path}, timeseries_cv_comparison_folds_multiseed.csv, and timeseries_cv_comparison_summary_multiseed.csv")

# --- Early exit for debugging purposes ---
# compare_time_series_cv(train_df, FEATURES, TARGET, output_dir=OUTPUT_DIRECTORY, seeds=[13, 123, 1999, 2025, 9001])

# --- Cross-Validation Strategy Experiment ---
# def run_cv_strategy_experiment(train_df, FEATURES, TARGET, output_dir="output"):
#     """
#     Run and plot a comparison of different CV strategies on the current data and feature set.
#     """
#     import matplotlib.pyplot as plt
#     from lightgbm import LGBMRegressor
#     from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
#     from sklearn.metrics import mean_squared_log_error
#     import pandas as pd
#     import numpy as np
#     import os
#     # Optional: iterative-stratification
#     try:
#         from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, IterativeStratification, RepeatedMultilabelStratifiedKFold
#         HAS_ITERSTRAT = True
#     except ImportError:
#         HAS_ITERSTRAT = False
#         print("iterative-stratification not installed. Skipping those CV types.")
#     X = train_df[FEATURES]
#     y = train_df[TARGET]
#     cv_strategies = {
#         'KFold': KFold(n_splits=5, shuffle=True, random_state=42),
#         'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
#         'TimeSeriesSplit': TimeSeriesSplit(n_splits=5),
#         'GroupKFold(center_id)': GroupKFold(n_splits=5),
#         'GroupTimeSeriesSplit': GroupTimeSeriesSplit(n_splits=5),
#     }
#     # Add custom_timeseries_split_with_overlap to CV strategies
#     def custom_timeseries_split_with_overlap_wrapper(X, y=None, n_splits=3, overlap_weeks=4):
#         # Use the global train_df for week info
#         for train_idx, val_idx in custom_timeseries_split_with_overlap(train_df, n_splits=n_splits, overlap_weeks=overlap_weeks):
#             if len(train_idx) > 0 and len(val_idx) > 0:
#                 yield train_idx, val_idx
#     cv_strategies['CustomTimeSeriesOverlap'] = custom_timeseries_split_with_overlap_wrapper
#     if HAS_ITERSTRAT:
#         cv_strategies['RepeatedMultilabelStratifiedKFold'] = RepeatedMultilabelStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
#         cv_strategies['MultilabelStratifiedKFold'] = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     # IterativeStratification handled as a special case below
#     def get_stratify_labels():
#         if 'category' in train_df.columns:
#             return train_df['category']
#         elif 'cuisine' in train_df.columns:
#             return train_df['cuisine']
#         else:
#             return pd.qcut(y, q=5, labels=False, duplicates='drop')
#     def get_groups():
#         if 'center_id' in train_df.columns:
#             return train_df['center_id']
#         else:
#             return None
#     def get_multilabel():
#         cols = []
#         for c in ['category', 'cuisine']:
#             if c in train_df.columns:
#                 cols.append(pd.get_dummies(train_df[c], prefix=c))
#         if cols:
#             return pd.concat(cols, axis=1).values
#         else:
#             return None
#     results = []    # Precompute labels
#     stratify_labels = get_stratify_labels()
#     groups = get_groups()
#     multilabel = get_multilabel()
#     for name, cv in cv_strategies.items():
#         print(f"\nRunning CV: {name}")
#         rmsle_scores = []
#         splits = None
#         if name == 'StratifiedKFold':
#             if stratify_labels is not None:
#                 splits = cv.split(X, stratify_labels)
#             else:
#                 print(f"No stratify labels available, skipping {name}.")
#                 continue
#         elif name == 'GroupKFold(center_id)':
#             if groups is not None:
#                 splits = cv.split(X, y, groups)
#             else:
#                 print(f"No group labels available, skipping {name}.")
#                 continue
#         elif name == 'CustomTimeSeriesOverlap':
#             splits = list(cv(X, y))
#         elif name == 'GroupTimeSeriesSplit':
#             if groups is not None:
#                 splits = cv.split(X, y, groups)
#             else:
#                 print(f"No group labels available, skipping {name}.")
#                 continue
#         elif name in ['MultilabelStratifiedKFold', 'RepeatedMultilabelStratifiedKFold'] and HAS_ITERSTRAT:
#             if multilabel is not None:
#                 splits = cv.split(X, multilabel)
#             else:
#                 print(f"No multilabels available, skipping {name}.")
#                 continue
#         elif name == 'IterativeStratification' and HAS_ITERSTRAT:
#             if multilabel is not None:
#                 from iterstrat.ml_stratifiers import IterativeStratification
#                 istrat = IterativeStratification(labels=multilabel, r=5, random_state=42)
#                 splits = istrat
#             else:
#                 print("No multilabels available, skipping IterativeStratification.")
#                 continue
#         else:
#             splits = cv.split(X, y)
#         for fold, (train_idx, val_idx) in enumerate(splits):
#             X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#             y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#             model = LGBMRegressor(n_estimators=300, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_val)
#             rmsle = np.sqrt(mean_squared_log_error(np.maximum(0, y_val), np.maximum(0, y_pred)))
#             rmsle_scores.append(rmsle)
#             print(f"  Fold {fold+1}: RMSLE={rmsle:.4f}")
#     # --- Plot Results ---
#     plt.figure(figsize=(10,6))
#     for r in results:
#         plt.plot(r['folds'], marker='o', label=f"{r['cv']} (mean={r['mean_rmsle']:.4f})")
#     plt.xlabel('Fold')
#     plt.ylabel('RMSLE')
#     plt.title('CV Strategy Comparison: Fold RMSLEs')
#     plt.legend()
#     plt.tight_layout()
#     plot_path = os.path.join(output_dir, 'cv_strategy_comparison.png')
#     plt.savefig(plot_path)
#     plt.close()
#     # Save results to CSV
#     pd.DataFrame(results).to_csv(os.path.join(output_dir, 'cv_strategy_comparison_results.csv'), index=False)
#     print(f"\nAll results saved to {plot_path} and cv_strategy_comparison_results.csv")

# --- Call CV strategy experiment after feature engineering ---
# run_cv_strategy_experiment(train_df, FEATURES, TARGET, OUTPUT_DIRECTORY)



# --- Empirical GroupTimeSeriesSplit Grouping Strategy Test ---
def empirical_group_split_test(train_df, FEATURES, TARGET, params=None, n_splits=3, max_folds=3):
    """
    Empirically test GroupTimeSeriesSplit with different groupings:
    - meal_id only
    - center_id only
    - (center_id, meal_id) composite
    Prints mean/std RMSLE for each and recommends the best.
    """
    # Ensure positional indices for iloc
    train_df = train_df.reset_index(drop=True)
    from collections import OrderedDict
    groupings = OrderedDict({
        'meal_id': train_df['meal_id'],
        'center_id': train_df['center_id'],
        'center_meal': train_df['center_id'].astype(str) + '_' + train_df['meal_id'].astype(str),
    })
    results = {}
    for name, groups in groupings.items():
        rmsles = []
        gtscv = GroupTimeSeriesSplit(n_splits=n_splits)
        print(f"\nTesting GroupTimeSeriesSplit with groups = {name}...")
        for fold, (train_idx, val_idx) in enumerate(gtscv.split(train_df, groups=groups)):
            if fold >= max_folds:
                break
            model_params = dict(params or final_params)
            # model_params.pop('n_estimators', None)
            train_idx = np.array(train_idx).flatten()
            val_idx = np.array(val_idx).flatten()
            X_tr, X_val = train_df.iloc[train_idx][FEATURES], train_df.iloc[val_idx][FEATURES]
            y_tr, y_val = train_df.iloc[train_idx][TARGET], train_df.iloc[val_idx][TARGET]
            model = LGBMRegressor(**model_params, random_state=SEED)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = rmsle(y_val, y_pred)
            rmsles.append(score)
            print(f"  Fold {fold+1}: RMSLE={score:.4f}")
        mean_rmsle = np.mean(rmsles)
        std_rmsle = np.std(rmsles)
        results[name] = (mean_rmsle, std_rmsle)
        print(f"Mean RMSLE: {mean_rmsle:.5f}, Std: {std_rmsle:.5f}")
    # Recommend best
    best_group = min(results, key=lambda k: results[k][0])
    print("\nSummary of Grouping Strategies:")
    for k, (mean_r, std_r) in results.items():
        print(f"  {k:15s}: Mean RMSLE={mean_r:.5f}, Std={std_r:.5f}")
    print(f"\nRecommended grouping for GroupTimeSeriesSplit: {best_group} (lowest mean RMSLE)")
    return results

# --- Run empirical group split test before Optuna tuning ---
# try:
#     empirical_group_split_test(train_split_df, FEATURES, TARGET, params=final_params, n_splits=3, max_folds=3)
# except Exception as e:
#     logging.warning(f"Could not run empirical group split test: {e}")


# --- Feature Selection and Hyperparameter Tuning with Optuna ---

# --- Precompute eligible features and all possible combos for Optuna feature interactions (module-level, before study) ---

# --- Freeze eligible features for Optuna interaction search ---
FROZEN_FEATURES_FOR_INTERACTIONS = [
    'checkout_price', 'base_price', 'discount', 'discount_pct', 'price_diff',
    'center_orders_mean', 'meal_orders_mean', 'meal_orders_mean_x_discount_pct',
    'meal_orders_mean_x_emailer_for_promotion', 'price_diff_x_emailer', 'price_diff_x_homepage',
    'discount_pct_x_emailer', 'discount_pct_x_homepage'
    # Add more features as needed, but do not change this list between runs of the same study!
]
from itertools import combinations
MAX_INTERACTION_ORDER = min(4, len(FROZEN_FEATURES_FOR_INTERACTIONS))
MAX_INTERACTIONS_PER_ORDER = {2: 5, 3: 3, 4: 2, 5: 2}
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
        bagging_freq = trial.suggest_int('bagging_freq', 1, 7)
    else:
        bagging_fraction = 1.0
        bagging_freq = 0
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),  # Lower for less overfit
        'num_leaves': trial.suggest_int('num_leaves', 4, 32),  # Lower for less complexity
        'max_depth': trial.suggest_int('max_depth', 2, 6),     # Lower for less complexity
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),  # ↑ Min value
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 400),  # ↑ Min value
        'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 30.0, log=True),    # ↑ Min value
        'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 30.0, log=True),    # ↑ Min value
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 400),    # ↑ Min value
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50000, 300000),
        'boosting_type': boosting_type,
        'max_bin': trial.suggest_int('max_bin', 68, 256),  # Lower for regularization
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



    interaction_features = []
    new_interaction_cols = {}
    used_interactions = set()
    for order in range(2, MAX_INTERACTION_ORDER + 1):
        # Use only the precomputed, static list of strings
        all_combos_str = ALL_COMBOS_STR[order]
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
                # Multiply all features in combo
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
    if len(selected_features) < 5:
        raise lgb.callback.EarlyStopException(best_iteration=0, best_score=float('inf'))

    # Use rolling window group time series split
    rgs = RollingGroupTimeSeriesSplit(n_splits=3, train_window=20, val_window=4, week_col='week')
    groups = train_split_df["center_id"]
    scores = []
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
            callbacks=[
                LightGBMPruningCallback(trial, metric='rmsle', valid_name='valid_1', ),
                early_stopping_with_overfit(300, OVERFIT_ROUNDS, verbose=False)
            ]
        )
        y_pred = model.predict(train_split_df.iloc[valid_idx][selected_features])
        score = rmsle(train_split_df.iloc[valid_idx][TARGET], y_pred)
        if (score is None or np.isnan(score) or np.isinf(score)):
            raise lgb.callback.EarlyStopException(best_iteration=0, best_score=float('inf'))
        scores.append(score)
    return np.mean(scores)

logging.info("Starting Optuna feature+hyperparam selection...")

# Reduce Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use NSGAIISampler for multi-objective/genetic search
from optuna.samplers import NSGAIISampler

class TqdmOptunaCallback:
    def __init__(self, n_trials, study=None, print_every=1):
        self.pbar = tqdm(total=n_trials, desc="Optuna Trials", position=0, leave=True)
        self.print_every = print_every
        # Initialize best_value and best_trial from study if available
        if study is not None:
            try:
                if study.best_trial is not None and study.best_trial.value is not None:
                    self.best_value = study.best_trial.value
                    self.best_trial = study.best_trial.number
                else:
                    self.best_value = float('inf')
                    self.best_trial = None
            except Exception:
                self.best_value = float('inf')
                self.best_trial = None
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
        # Handle None trial.value gracefully
        if trial.value is not None and trial.value < self.best_value:
            self.best_value = trial.value
            self.best_trial = trial.number
            # ANSI green for new best
            msg = f"\033[92mTrial {trial.number} finished with value: {trial.value:.5f} | BEST! {self.best_value:.5f}\033[0m | Features: {n_features} | Params: {params_str}"
        elif trial.number % self.print_every == 0:
            val_str = f"{trial.value:.5f}" if trial.value is not None else "None"
            best_str = f"{self.best_value:.5f}" if self.best_value is not None else "None"
            msg = f"Trial {trial.number} finished with value: {val_str} | Best: {best_str} | Features: {n_features} | Params: {params_str}"
        if msg:
            tqdm.write(msg)
    def close(self):
        self.pbar.close()

# Create the study with NSGAIISampler
# --- NSGAIISampler advanced configuration ---
from optuna.samplers.nsgaii import UniformCrossover, SBXCrossover

# Recommended: population_size = 16-32, crossover_prob=0.9, swapping_prob=0.5, mutation_prob=1/num_params

optuna_storage = OPTUNA_DB
feature_hyperparam_study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=N_WARMUP_STEPS),  
    study_name=OPTUNA_STUDY_NAME,
    storage=optuna_storage,
    load_if_exists=True,
    sampler=NSGAIISampler(
        seed=SEED,
        population_size=POPULATION_SIZE,
        crossover=UniformCrossover(), # Use SBXCrossover for continuous, UniformCrossover for mixed/categorical
        crossover_prob=0.9,
        swapping_prob=0.5,
        # mutation_prob=mutation_prob
    )
)
# Pass the study to the callback so it can initialize best_value/best_trial

optuna_callback = TqdmOptunaCallback(OPTUNA_TRIALS, study=feature_hyperparam_study, print_every=1)
try:
    feature_hyperparam_study.optimize(
        partial(optuna_feature_selection_and_hyperparam_objective, train_split_df=train_split_df),
        n_trials=OPTUNA_TRIALS,
        timeout=7200,
        callbacks=[optuna_callback],
        n_jobs=1
    )
except KeyboardInterrupt:
    print("\nInterrupted! Waiting for the current trial to finish and be saved...")
finally:
    print(f"Final best value: {feature_hyperparam_study.best_value:.5f}")
optuna_callback.close()

# Extract best features and params, but handle missing best_trial gracefully
if feature_hyperparam_study.best_trial is None:
    logging.warning("No completed Optuna trial found. Skipping feature/param extraction.")
    SELECTED_FEATURES = FEATURES.copy()
    best_params = final_params.copy()
else:
    best_mask = [feature_hyperparam_study.best_trial.params.get(f, False) for f in FEATURES]
    SELECTED_FEATURES = [f for f, keep in zip(FEATURES, best_mask) if keep]
    best_params = {k: v for k, v in feature_hyperparam_study.best_trial.params.items() if k not in FEATURES and not k.endswith('_pair')}
    if best_params.get('boosting_type') == 'goss':
        best_params['bagging_fraction'] = 1.0
        best_params['bagging_freq'] = 0
    selected_pairs = {k: v for k, v in feature_hyperparam_study.best_trial.params.items() if k.endswith('_pair')}
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
ensemble_preds, ensemble_models = recursive_ensemble(train_df, test_df, FEATURES, weekofyear_means, month_means, n_models=N_ENSEMBLE_MODELS, eval_metric=lgb_rmsle)
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

import os
import random
import time
import logging
import re
import numpy as np
import pandas as pd
from math import pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotext as pltx  # For live ASCII plotting
import shap
from tqdm import tqdm
from itertools import combinations
from functools import partial
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.samplers.nsgaii import UniformCrossover, SBXCrossover
from optuna.samplers import NSGAIISampler, NSGAIIISampler, RandomSampler, TPESampler
from optuna.pruners import MedianPruner, NopPruner

import warnings
warnings.filterwarnings("ignore", message="The reported value is ignored because this `step` .* is already reported.")

# --- Configuration & Constants ---
OUTPUT_DIRECTORY = "output"
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
# Randomized seed per-run for variability
SEED = random.randint(0, 1000000)
# Interaction search settings
GROUP_COLS = ["center_id", "meal_id"]
MAX_INTERACTION_ORDER = 4
MAX_INTERACTIONS_PER_ORDER = {2: 18, 3: 4, 4: 1, 5: 0}
# Rolling window sizes
LAGGING_WINDOWS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]
ROLLING_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 35, 42, 49, 52]
# Optuna study flags & weights
OPTUNA_MULTI_OBJECTIVE = True
OBJECTIVE_WEIGHT_MEAN_VALID = 1.0
OBJECTIVE_WEIGHT_GAP_PENALTY = 0.3
OBJECTIVE_WEIGHT_COMPLEXITY_PENALTY = 1e-4
OBJECTIVE_WEIGHT_REG_REWARD = 1e-3
# Ensemble & training settings
N_ENSEMBLE_MODELS = 5
OVERFIT_ROUNDS = 17
VALIDATION_WEEKS = 8
# Pruning & sampler settings
N_WARMUP_STEPS = 200
POPULATION_SIZE = 32
OPTUNA_SAMPLER = "Default"
PRUNING_ENABLED = False
# Optuna search limits
OPTUNA_TRIALS = 1000000
OPTUNA_TIMEOUT = 60 * 60 * 24
# Rerun top trials
RERUN_TOP_N = 0
RERUN_OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"
# Storage & naming
OPTUNA_STUDY_NAME = "multi_objective_lgbm_tuning"
OPTUNA_DB = f"sqlite:///optuna_study_{OPTUNA_STUDY_NAME}.db"
# OPTUNA_DB = "postgresql://postgres:optuna@34.55.13.135:5432/optuna"
SUBMISSION_FILE_PREFIX = "submission_recursive"
SHAP_FILE_PREFIX = "shap_recursive"
N_SHAP_SAMPLES = 2000

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def add_training_noise(df, features, target,
                       noise_target_level=0.0,
                       noise_feature_level=0.0,
                       bootstrap_frac=0.0,
                       seed=None,
                       group_cols=None):
    """
    Add noise and optional group-wise bootstrap sampling for robust training.
    """
    df_noise = df.copy()
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    X = df_noise[features].copy()
    y = df_noise[target].copy()
    # Target noise
    if noise_target_level > 0:
        y_std = y.std()
        y += rng.normal(0, noise_target_level * y_std, size=len(y)).clip(0)
    # Feature noise (numeric only)
    if noise_feature_level > 0:
        for col in features:
            if pd.api.types.is_numeric_dtype(X[col]) and not isinstance(X[col].dtype, pd.CategoricalDtype):
                f_std = X[col].std()
                if f_std > 0:
                    X[col] += rng.normal(0, noise_feature_level * f_std, size=len(X))
    # Bootstrap sampling by group
    if bootstrap_frac > 0 and group_cols:
        idx = []
        for _, grp in df_noise.groupby(group_cols, observed=False):
            inds = grp.index.values
            n = max(1, int(len(inds) * bootstrap_frac))
            idx.extend(rng.choice(inds, size=n, replace=True))
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)
    return X, y

class ExpandingGroupTimeSeriesSplit:
    """
    Expanding window cross-validator for time series data with group awareness.
    Training window expands over time; validation window is fixed length.
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
            raise ValueError(
                f"Not enough weeks for {self.n_splits} splits with min_train_window={self.min_train_window} and val_window={self.val_window}."
            )
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

# --- Data Loading ---
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        test = pd.read_csv(TEST_PATH)
        meal_info = pd.read_csv(MEAL_INFO_PATH)
        center_info = pd.read_csv(CENTER_INFO_PATH)
    except FileNotFoundError as e:
        logging.error(f"Error loading data file: {e}")
        raise
    return df, test, meal_info, center_info

# --- Preprocessing ---
def preprocess_data(df, meal_info, center_info):
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

# --- Feature Engineering ---
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

def create_lag_rolling_features(df, target_col='num_orders'):
    df_out = df.copy()
    grp = df_out.groupby(["center_id","meal_id"], observed=False)
    shifted = grp[target_col].shift(1)
    for l in LAGGING_WINDOWS:
        df_out[f"{target_col}_lag_{l}"] = grp[target_col].shift(l)
    for w in ROLLING_WINDOWS:
        df_out[f"{target_col}_rolling_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(drop=True)
        df_out[f"{target_col}_rolling_std_{w}"] = shifted.rolling(w, min_periods=1).std().reset_index(drop=True)
    return df_out

def create_other_features(df):
    df_out = df.copy()
    grp = df_out.groupby(["center_id","meal_id"], observed=False)
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = grp["checkout_price"].diff()
    return df_out

def add_binary_rolling_means(df, binary_cols=["emailer_for_promotion","homepage_featured"]):
    df_out = df.copy()
    grp = df_out.groupby(["center_id","meal_id"], observed=False)
    for col in binary_cols:
        if col in df_out.columns:
            shifted = grp[col].shift(1)
            for w in ROLLING_WINDOWS:
                df_out[f"{col}_rolling_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(drop=True)
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
    df_out = df.copy()
    if is_train:
        weekofyear_means = df_out.groupby('weekofyear')['num_orders'].mean()
        month_means = df_out.groupby('month')['num_orders'].mean()
    else:
        if weekofyear_means is None or month_means is None:
            raise ValueError("When is_train=False, weekofyear_means and month_means must be provided.")
    df_out['mean_orders_by_weekofyear'] = df_out['weekofyear'].map(weekofyear_means)
    df_out['mean_orders_by_month'] = df_out['month'].map(month_means)
    return df_out, weekofyear_means, month_means

def apply_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None):
    df_out = df.copy()
    df_out = create_temporal_features(df_out)
    if is_train or 'num_orders' in df_out.columns:
        df_out = create_lag_rolling_features(df_out)
    df_out = create_other_features(df_out)
    df_out = add_binary_rolling_means(df_out)
    df_out = create_group_aggregates(df_out)
    df_out, weekofyear_means, month_means = add_seasonality_features(df_out, weekofyear_means, month_means, is_train)
    # Fill NaNs
    cols = [c for c in df_out.columns if any(x in c for x in ['lag_','rolling_','price_diff'])]
    df_out[cols] = df_out[cols].fillna(0)
    if 'discount_pct' in df_out.columns:
        df_out['discount_pct'] = df_out['discount_pct'].fillna(0)
    # Remove duplicate cols
    df_out = df_out.loc[:,~df_out.columns.duplicated()]
    return df_out, weekofyear_means, month_means

# --- Model Training ---
#### Metrics & Callbacks ####
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def rmsle_lgbm(preds, data):
    """
    RMSLE metric for LightGBM. Supports both native Dataset and sklearn eval_set interfaces.
    """
    # Native LightGBM callback passes Dataset with get_label(), sklearn wrapper may pass y_true array
    if hasattr(data, 'get_label'):
        y_true = data.get_label()
    else:
        # data is numpy array of true labels
        y_true = data
    return 'rmsle', rmsle(y_true, preds), False

def early_stopping_with_overfit(stopping_rounds=300, overfit_rounds=OVERFIT_ROUNDS, verbose=False):
    best_score, best_iter = float('inf'), 0
    overfit_count, prev_train, prev_valid = 0, float('inf'), float('inf')
    def _callback(env):
        nonlocal best_score,best_iter,overfit_count,prev_train,prev_valid
        # extract
        evals = {name: score for name,name2,score,name3 in env.evaluation_result_list for name,score,name2,name3 in [ (env.evaluation_result_list[0][0],None,None,None) ] }
        # naive fallback to env.evaluation_result_list[0]
        train_loss = env.evaluation_result_list[0][2]
        valid_loss = env.evaluation_result_list[1][2]
        # standard
        if valid_loss < best_score:
            best_score, best_iter = valid_loss, env.iteration
            overfit_count=0
        else:
            if valid_loss>prev_valid and train_loss<prev_train:
                overfit_count+=1
            else:
                overfit_count=0
        prev_train, prev_valid = train_loss, valid_loss
        if overfit_count>=overfit_rounds or env.iteration-best_iter>=stopping_rounds:
            raise lgb.callback.EarlyStopException(best_iter, best_score)
    return _callback

# default final params for recursive training
final_params = {
    'objective':'regression_l1','boosting_type':'gbdt','n_estimators':3000,
    'seed':SEED,'n_jobs':-1,'verbose':-1,'metric':'None',
    'lambda_l1':10.0,'lambda_l2':10.0,'min_child_samples':150,'min_data_in_leaf':250,
    'num_leaves':16,'max_depth':4,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':3
}

### Optuna Feature & Hyperparameter Tuning Objective ###
def optuna_feature_selection_and_hyperparam_objective(trial, train_split_df):
    # Hyperparameter space
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt','dart','goss'])
    if boosting_type != 'goss':
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0)
        bagging_freq = trial.suggest_int('bagging_freq', 0, 10)
    else:
        bagging_fraction, bagging_freq = 1.0, 0
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 512),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 2000),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1e3, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1e3, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 2000),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 500000),
        'max_bin': trial.suggest_int('max_bin', 32, 1024),
        'objective': 'regression_l1',
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'n_jobs': -1, 'verbose': -1, 'metric': 'None'
    }
    # Noise injection params
    noise_t = trial.suggest_float('noise_target_level', 0.0, 0.1)
    noise_f = trial.suggest_float('noise_feature_level', 0.0, 0.1)
    bootstrap_frac = trial.suggest_float('bootstrap_frac', 0.0, 0.3)
    # Dynamic feature flags
    sincos = [f for f in FEATURES if re.search(r'(_sin|_cos)', f)]
    pair_map = {m.group(1): (m.group(0), m.group(1)+'_cos')
                for f in sincos if (m:=re.match(r'(.*)_sin', f))}
    selected = []
    for pref, (sin, cos) in pair_map.items():
        if trial.suggest_categorical(f"{sin}_{cos}_pair", [True, False]):
            selected += [sin, cos]
    selected += [f for f in FEATURES if f not in sincos and trial.suggest_categorical(f, [True, False])]
    
    # Feature interactions
    interaction_feats, new_cols, used = [], {}, set()
    for order, all_str in INTERACTION_POOL.items():

        # suggest number of interactions for this order using a predefined list
        max_this = min(MAX_INTERACTIONS_PER_ORDER.get(order, 1), len(all_str))
        n_this  = trial.suggest_int(f"n_{order}th_order", 0, max_this) if max_this > 0 else 0

        for i in range(n_this):
            # Suggest index for combo to avoid dynamic categorical distribution
            idx = trial.suggest_int(f"inter_{order}th_{i}_idx", 0, len(all_str) - 1)
            combo_str = all_str[idx]
            if combo_str in used:
                continue
            used.add(combo_str)
            combo = combo_str.split('|')
            # product
            prod = '_prod_'.join(combo)
            if prod not in train_split_df and prod not in new_cols:
                val = train_split_df[combo[0]].copy()
                for c in combo[1:]: val *= train_split_df[c]
                new_cols[prod] = val
            interaction_feats.append(prod)
            # additive
            add = '_add_'.join(combo)
            if add not in train_split_df and add not in new_cols:
                val = train_split_df[combo[0]].copy()
                for c in combo[1:]: val += train_split_df[c]
                new_cols[add] = val
            interaction_feats.append(add)
            # ratio/poly2
            if len(combo)==2:
                div = '_div_'.join(combo)
                if div not in train_split_df and div not in new_cols:
                    den = train_split_df[combo[1]].replace(0, np.nan)
                    new_cols[div] = train_split_df[combo[0]]/den + 1e-15
                interaction_feats.append(div)
                poly2 = f"{combo[0]}_poly2_{combo[1]}"
                if poly2 not in train_split_df and poly2 not in new_cols:
                    new_cols[poly2] = train_split_df[combo[0]]*(train_split_df[combo[1]]**2)
                interaction_feats.append(poly2)
    if new_cols:
        train_split_df = pd.concat([train_split_df, pd.DataFrame(new_cols,index=train_split_df.index)], axis=1)
    selected += interaction_feats
    selected = list(dict.fromkeys(selected))
    if len(selected)<10:
        raise optuna.TrialPruned()
    # CV training
    cv = ExpandingGroupTimeSeriesSplit(n_splits=5, min_train_window=30, val_window=10, week_col='week')
    train_scores, valid_scores = [], []
    callbacks = ([early_stopping_with_overfit()] if not PRUNING_ENABLED or OPTUNA_MULTI_OBJECTIVE else [LightGBMPruningCallback(trial, 'rmsle')])
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in selected]
    for tr,vd in cv.split(train_split_df, groups=train_split_df['center_id']):
        sub = train_split_df.iloc[tr].reset_index(drop=True)
        Xtr, Ytr = add_training_noise(sub, selected, TARGET, noise_t, noise_f, bootstrap_frac, seed=SEED+trial.number, group_cols=GROUP_COLS)
        m = LGBMRegressor(**params, seed=SEED+trial.number)
        m.fit(
            Xtr, Ytr,
            eval_set=[
                (Xtr, Ytr),
                (train_split_df.iloc[vd][selected], train_split_df.iloc[vd][TARGET])
            ],
            eval_metric=rmsle_lgbm,
            callbacks=callbacks,
            categorical_feature=cat_feats,
        )
        tr_pred = m.predict(train_split_df.iloc[tr][selected])
        vd_pred = m.predict(train_split_df.iloc[vd][selected])
        train_scores.append(rmsle(train_split_df.iloc[tr][TARGET],tr_pred))
        valid_scores.append(rmsle(train_split_df.iloc[vd][TARGET],vd_pred))
    mean_tr, mean_vd = np.mean(train_scores), np.mean(valid_scores)
    gap = mean_vd - mean_tr
    # Objectives
    val_obj = OBJECTIVE_WEIGHT_MEAN_VALID * mean_vd
    gap_pen = OBJECTIVE_WEIGHT_GAP_PENALTY * max(0, gap)
    comp_pen = OBJECTIVE_WEIGHT_COMPLEXITY_PENALTY * (len(selected) + params['num_leaves'] + max(params['max_depth'],0))
    reg_pen  = OBJECTIVE_WEIGHT_REG_REWARD * (np.log1p(params['lambda_l1']) + np.log1p(params['lambda_l2']))
    trial.set_user_attr('mean_valid', float(mean_vd))
    trial.set_user_attr('gap_penalty', float(gap_pen))
    trial.set_user_attr('complexity_penalty', float(comp_pen))
    trial.set_user_attr('reg_reward', float(reg_pen))
    trial.set_user_attr('objective', float(mean_vd + gap_pen))
    # store selected features and noise params
    trial.set_user_attr('selected_features', selected)
    trial.set_user_attr('noise_target_level', noise_t)
    trial.set_user_attr('noise_feature_level', noise_f)
    trial.set_user_attr('bootstrap_frac', bootstrap_frac)
    if OPTUNA_MULTI_OBJECTIVE:
        return [mean_vd, gap_pen]
    else:
        return mean_vd
    # end optuna_feature_selection_and_hyperparam_objective

#### Study Utilities ####
def rerun_old_trials_in_new_study(new_study):
    """
    Optionally enqueue top trials from a previous study into the new study.
    """
    if RERUN_TOP_N <= 0:
        return
    try:
        old = optuna.load_study(study_name=RERUN_OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
    except Exception as e:
        logging.warning(f"Could not load old study '{RERUN_OPTUNA_STUDY_NAME}': {e}")
        return
    # sort trials by stored objective
    sorted_trials = sorted(
        [t for t in old.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.user_attrs.get('objective', t.value)
    )
    for t in sorted_trials[:RERUN_TOP_N]:
        new_study.enqueue_trial(t.params)

def get_weighted_objective(trial):
    """
    Compute a single-objective value for a trial, using stored objective or value.
    """
    return trial.user_attrs.get('objective', trial.value if trial.value is not None else float('inf'))

#### Live Optuna Callback with plotext ####
class TqdmOptunaCallback:
    def __init__(self, n_trials, study=None, print_every=1):
        self.n_trials = n_trials
        self.study = study
        self.print_every = print_every
        self.pbar = tqdm(total=n_trials, desc='Optuna Trials', leave=False)
        self.best_trial = None
        self.best_value = float('inf')
    def __call__(self, study, trial):
        self.pbar.update(1)
        # val = trial.user_attrs.get('objective', trial.value if trial.value is not None else float('inf'))
        val = trial.user_attrs.get('objective')
        if val is None:
            val = get_weighted_objective(trial)
        if val is None:
            val = float('inf')
        if val < self.best_value:
            self.best_value = val
            self.best_trial = trial
        if trial.number % self.print_every == 0:
            # live plot objectives
            trials = study.trials
            xs, ys = [], []
            for t in trials:
                if 'mean_valid' in t.user_attrs:
                    xs.append(t.number)
                    ys.append(t.user_attrs['mean_valid'])
            pltx.clf()
            pltx.plot(xs, ys, label='mean_valid')
            pltx.title(f'Optuna mean_valid (best {self.best_value:.5f})')
            pltx.show()
        # no return
    def close(self):
        self.pbar.close()

#### Prediction Utilities ####
def ensure_interaction_features(df, feature_names):
    """
    Ensure all interaction feature columns exist in df (fill missing with zeros).
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df

def get_weighted_objective(trial):
    """
    Compute weighted sum of objectives for selecting best trial in multi-objective.
    """
    mv = trial.user_attrs.get('mean_valid', float('inf'))
    gp = trial.user_attrs.get('gap_penalty', 0)
    return OBJECTIVE_WEIGHT_MEAN_VALID * mv + OBJECTIVE_WEIGHT_GAP_PENALTY * gp

def recursive_predict(model, train_df, predict_df, features, weekofyear_means, month_means):
    """
    Sequentially predict num_orders for each row in predict_df, appending to df.
    """
    df_comb = pd.concat([train_df, predict_df], ignore_index=True)
    # ensure all feature columns exist for predict_df
    df_comb = ensure_interaction_features(df_comb, features)
    preds = []
    for i in range(len(train_df), len(df_comb)):
        row = df_comb.iloc[i:i+1]
        pred = model.predict(row[features])[0]
        df_comb.at[i, 'num_orders'] = pred
        preds.append(pred)
    return preds

def recursive_ensemble(train_df, test_df, features, weekofyear_means, month_means,
                       n_models=N_ENSEMBLE_MODELS, noise_target_level=0, noise_feature_level=0,
                       bootstrap_frac=0, eval_metric=None):
    """
    Train multiple models with noise injection and bootstrap, then average recursive predictions.
    """
    preds_sum = np.zeros(len(test_df))
    models = []
    for i in range(n_models):
        # clone train_df for noise
        X_tr, y_tr = add_training_noise(train_df, features, 'num_orders',
                                         noise_target_level, noise_feature_level,
                                         bootstrap_frac, seed=SEED+i, group_cols=GROUP_COLS)
        model = LGBMRegressor(**final_params, seed=SEED+i)
        # fit on noisy data
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr)],
            eval_metric=rmsle_lgbm,
            callbacks=[early_stopping_with_overfit()],
        )
        models.append(model)
        # recursive predict
        preds = recursive_predict(model, train_df, test_df.copy(), features,
                                  weekofyear_means, month_means)
        preds_sum += np.array(preds)
    return preds_sum / n_models, models


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Loading data...")
    df, test, meal_info, center_info = load_data()

    logging.info("Preprocessing data...")
    df = preprocess_data(df, meal_info, center_info)
    test = preprocess_data(test, meal_info, center_info)

    # --- Feature Engineering on full dataset ---
    logging.info("Applying feature engineering to full dataset...")
    df_full = pd.concat([df, test], ignore_index=True)
    # Drop any stale one-hot prefixes
    for prefix in ["category_", "cuisine_", "center_type_"]:
        df_full = df_full.loc[:, ~df_full.columns.str.startswith(prefix)]
    # Separate train vs test based on num_orders
    train_full = df_full[df_full['num_orders'].notna()].copy()
    test_full  = df_full[df_full['num_orders'].isna()].copy()
    # Apply feature engineering
    train_df, week_means, month_means = apply_feature_engineering(train_full, is_train=True)
    test_df, _, _           = apply_feature_engineering(test_full, is_train=False,
                                                       weekofyear_means=week_means,
                                                       month_means=month_means)
    # --- Define Features and Target ---
    TARGET = 'num_orders'
    FEATURES = []
    features_set = set()
    # base features
    base_feats = [
        'checkout_price','base_price','homepage_featured','emailer_for_promotion',
        'discount','discount_pct','price_diff',
        'category','cuisine','center_type','center_id','meal_id'
    ]
    for f in base_feats:
        if f in train_df.columns and f not in features_set:
            FEATURES.append(f); features_set.add(f)
    # lag and rolling
    for lag in ROLLING_WINDOWS:
        col = f"{TARGET}_lag_{lag}"
        if col in train_df.columns and col not in features_set:
            FEATURES.append(col); features_set.add(col)
    for w in ROLLING_WINDOWS:
        m = f"{TARGET}_rolling_mean_{w}"; s = f"{TARGET}_rolling_std_{w}"
        for c in (m,s):
            if c in train_df.columns and c not in features_set:
                FEATURES.append(c); features_set.add(c)
    # binary rolling
    for col in ['emailer_for_promotion','homepage_featured']:
        for w in ROLLING_WINDOWS:
            c = f"{col}_rolling_mean_{w}"
            if c in train_df.columns and c not in features_set:
                FEATURES.append(c); features_set.add(c)
    # group aggregates
    for prefix in ['center_orders_','meal_orders_','category_orders_']:
        for c in train_df.columns:
            if c.startswith(prefix) and c not in features_set and c!=TARGET and c!='id':
                FEATURES.append(c); features_set.add(c)
    # seasonality
    for c in ['weekofyear_sin','weekofyear_cos','month_sin','month_cos','mean_orders_by_weekofyear','mean_orders_by_month']:
        if c in train_df.columns and c not in features_set:
            FEATURES.append(c); features_set.add(c)
    # remove raw
    for c in ['weekofyear','month']:
        if c in FEATURES:
            FEATURES.remove(c); features_set.discard(c)

    logging.info(f"Using {len(FEATURES)} features: {FEATURES}")

    # restrict interactions to non-categorical features
    frozen_feats = [f for f in FEATURES if f not in ['category','cuisine','center_type','center_id','meal_id']]
    # precompute all the interaction strings once using only numeric/frozen features
    INTERACTION_POOL = {
        order: ["|".join(combo) for combo in combinations(frozen_feats, order)]
        for order in range(2, MAX_INTERACTION_ORDER+1)
    }

    # split train/validation
    max_week = train_df['week'].max()
    valid_df      = train_df[train_df['week'] >  max_week-VALIDATION_WEEKS].copy()
    train_split_df= train_df[train_df['week'] <= max_week-VALIDATION_WEEKS].copy()
    # set categorical dtype
    CATEGORICAL_FEATURES = [c for c in ['category','cuisine','center_type','center_id','meal_id'] if c in train_df]
    for d in [train_split_df, valid_df]:
        for c in CATEGORICAL_FEATURES:
            if c in d:
                d[c] = d[c].astype('category')

    # --- Configure Optuna sampler and pruner ---
    if OPTUNA_SAMPLER == "NSGAIISampler":
        sampler = NSGAIISampler(population_size=POPULATION_SIZE, crossover=UniformCrossover(), mutation=SBXCrossover())
    elif OPTUNA_SAMPLER == "NSGAIIISampler":
        sampler = NSGAIIISampler(population_size=POPULATION_SIZE)
    elif OPTUNA_SAMPLER == "RandomSampler":
        sampler = RandomSampler()
    else:
        sampler = TPESampler()
    pruner = MedianPruner(n_warmup_steps=N_WARMUP_STEPS) if PRUNING_ENABLED else NopPruner()
    # Create or load study
    if OPTUNA_MULTI_OBJECTIVE:
        study = optuna.create_study(directions=["minimize","minimize"], sampler=sampler,
                                    pruner=pruner, study_name=OPTUNA_STUDY_NAME,
                                    storage=OPTUNA_DB, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    pruner=pruner, study_name=OPTUNA_STUDY_NAME,
                                    storage=OPTUNA_DB, load_if_exists=True)
    logging.info(f"Study setup: {study.study_name}")
    # Optionally rerun top trials
    rerun_old_trials_in_new_study(study)
    # Optimize
    logging.info("Starting Optuna optimization...")
    cb = TqdmOptunaCallback(OPTUNA_TRIALS, study, print_every=10)
    study.optimize(
        lambda t: optuna_feature_selection_and_hyperparam_objective(t, train_split_df),
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        callbacks=[cb]
    )
    cb.close()
    # Select best trial
    if OPTUNA_MULTI_OBJECTIVE:
        best = min(study.trials, key=get_weighted_objective)
    else:
        best = study.best_trial
    logging.info(f"Best trial params: {best.params}")
    # Extract tuned settings
    tuned_feats = best.user_attrs['selected_features']
    noise_t = best.user_attrs.get('noise_target_level', 0)
    noise_f = best.user_attrs.get('noise_feature_level', 0)
    bf = best.user_attrs.get('bootstrap_frac', 0)
    # Recursive ensemble predictions
    logging.info("Generating recursive ensemble predictions...")
    preds, models = recursive_ensemble(train_df, test_df, tuned_feats,
                                       week_means, month_means,
                                       n_models=N_ENSEMBLE_MODELS,
                                       noise_target_level=noise_t,
                                       noise_feature_level=noise_f,
                                       bootstrap_frac=bf)
    # Save submission
    sub_file = f"{SUBMISSION_FILE_PREFIX}_final_optuna_ensemble.csv"
    test['num_orders'] = preds
    test[['id','num_orders']].to_csv(os.path.join(OUTPUT_DIRECTORY, sub_file), index=False)
    logging.info(f"Saved ensemble submission: {sub_file}")
    # SHAP analysis for first model
    logging.info("Running SHAP analysis...")
    explainer = shap.TreeExplainer(models[0])
    X_sample = train_df[tuned_feats].sample(min(N_SHAP_SAMPLES, len(train_df)), random_state=SEED)
    shap_vals = explainer.shap_values(X_sample)
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.tight_layout()
    shap_file = f"{SHAP_FILE_PREFIX}_summary.png"
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, shap_file))
    plt.clf()
    logging.info(f"Saved SHAP summary plot: {shap_file}")
    # Validation recursive plot
    logging.info("Plotting validation recursive predictions...")
    val_preds, _ = recursive_ensemble(train_split_df, valid_df, tuned_feats,
                                      week_means, month_means,
                                      n_models=1)
    plt.figure(figsize=(12,6))
    plt.plot(valid_df['num_orders'].values, label='actual')
    plt.plot(val_preds, label='pred')
    plt.legend()
    plt.title('Validation Recursive Predictions')
    val_plot = 'recursive_validation_plot.png'
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, val_plot))
    plt.clf()
    logging.info(f"Saved validation plot: {val_plot}")

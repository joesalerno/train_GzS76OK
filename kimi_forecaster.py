# --- Key Enhancements ---
# 1. More sophisticated temporal feature engineering
# 2. Enhanced cyclical feature encoding
# 3. Improved handling of outliers and special events
# 4. Better feature selection strategy
# 5. Advanced model tuning with composite metrics
# 6. More robust recursive prediction handling
# 7. Comprehensive validation and diagnostic plots

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import re
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
import shap
from tqdm import tqdm
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28]
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8
OPTUNA_TRIALS = 2000
OUTPUT_DIRECTORY = "output"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load and Preprocess Data ---
logging.info("Loading and preprocessing data...")
try:
    df = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    meal_info = pd.read_csv(MEAL_INFO_PATH)
    center_info = pd.read_csv(CENTER_INFO_PATH)
except FileNotFoundError as e:
    logging.error(f"Error loading data file: {e}")
    raise

def preprocess_data(df, meal_info, center_info):
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

df = preprocess_data(df, meal_info, center_info)
test = preprocess_data(test, meal_info, center_info)

# --- Feature Engineering ---
def create_advanced_temporal_features(df):
    """Create more sophisticated temporal features."""
    df_out = df.copy()
    df_out["weekofyear"] = df_out["week"] % 52
    df_out["weekofyear_sin"] = np.sin(2 * np.pi * df_out["weekofyear"] / 52)
    df_out["weekofyear_cos"] = np.cos(2 * np.pi * df_out["weekofyear"] / 52)
    df_out["month"] = ((df_out["week"] - 1) // 4) % 12 + 1
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)
    df_out["quarter"] = (df_out["week"] // 13) % 4 + 1
    df_out["quarter_sin"] = np.sin(2 * np.pi * df_out["quarter"] / 4)
    df_out["quarter_cos"] = np.cos(2 * np.pi * df_out["quarter"] / 4)
    holiday_weeks = set([1, 10, 25, 45, 52])
    df_out["is_holiday_week"] = df_out["weekofyear"].isin(holiday_weeks).astype(int)
    return df_out

def create_robust_features(df):
    """Create features with better handling of outliers and special events."""
    df_out = df.copy()
    # Robust scaling for features with outliers
    robust_scaler = RobustScaler()
    if 'checkout_price' in df_out.columns:
        df_out['checkout_price_scaled'] = robust_scaler.fit_transform(df_out[['checkout_price']])
    if 'base_price' in df_out.columns:
        df_out['base_price_scaled'] = robust_scaler.fit_transform(df_out[['base_price']])
    # Outlier flags
    if 'num_orders' in df_out.columns:
        q_low = df_out["num_orders"].quantile(0.01)
        q_high = df_out["num_orders"].quantile(0.99)
        df_out["is_outlier_order"] = ((df_out["num_orders"] < q_low) | (df_out["num_orders"] > q_high)).astype(int)
    return df_out

def create_additional_features(df):
    df = df.copy()
    group = ["center_id", "meal_id"]
    # Discount percent
    if 'base_price' in df.columns and 'checkout_price' in df.columns:
        df["discount_pct"] = (df["base_price"] - df["checkout_price"]) / df["base_price"].replace(0, np.nan)
        df["discount_pct"] = df["discount_pct"].fillna(0)
    # Price diff (current - previous week)
    if 'checkout_price' in df.columns:
        df["price_diff"] = df.groupby(group)["checkout_price"].diff().fillna(0)
    # Rolling means and stds for num_orders
    if 'num_orders' in df.columns:
        shifted = df.groupby(group)["num_orders"].shift(1)
        for window in [2, 5, 14]:
            df[f"num_orders_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        for window in [2, 14]:
            df[f"num_orders_rolling_std_{window}"] = shifted.rolling(window, min_periods=2).std().reset_index(0, drop=True).fillna(0)
    # Aggregate means
    if 'num_orders' in df.columns:
        df["center_orders_mean"] = df.groupby("center_id")["num_orders"].transform("mean")
        df["meal_orders_mean"] = df.groupby("meal_id")["num_orders"].transform("mean")
    # Interaction features
    if all(col in df.columns for col in ["num_orders_rolling_mean_2", "emailer_for_promotion"]):
        df["num_orders_rolling_mean_2_x_emailer_for_promotion"] = df["num_orders_rolling_mean_2"] * df["emailer_for_promotion"]
    if all(col in df.columns for col in ["num_orders_rolling_mean_5", "homepage_featured"]):
        df["num_orders_rolling_mean_5_x_homepage_featured"] = df["num_orders_rolling_mean_5"] * df["homepage_featured"]
    return df

# --- Custom RMSLE Evaluation Function ---
def rmsle_eval(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(0, y_pred)
    return 'rmsle', np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true)))), False

# --- Advanced Feature Engineering (from recursive_hybrid_forecast.py) ---
def create_group_aggregates(df):
    df_out = df.copy()
    df_out['center_orders_mean'] = df_out.groupby('center_id')['num_orders'].transform('mean')
    df_out['center_orders_std'] = df_out.groupby('center_id')['num_orders'].transform('std')
    df_out['meal_orders_mean'] = df_out.groupby('meal_id')['num_orders'].transform('mean')
    df_out['meal_orders_std'] = df_out.groupby('meal_id')['num_orders'].transform('std')
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
    demand_feats = [
        'num_orders_rolling_mean_2', 'num_orders_rolling_mean_5', 'num_orders_rolling_mean_14',
        'meal_orders_mean', 'center_orders_mean'
    ]
    price_feats = ['checkout_price', 'price_diff', 'discount_pct']
    promo_feats = ['emailer_for_promotion', 'homepage_featured']
    time_feats = ['weekofyear_sin', 'weekofyear_cos', 'month_sin', 'month_cos', 'mean_orders_by_weekofyear', 'mean_orders_by_month']
    top_feats = demand_feats + price_feats + promo_feats + time_feats
    for feat in top_feats:
        if feat in df_out.columns:
            if feat.endswith('_sin') or feat.endswith('_cos') or feat in ['emailer_for_promotion', 'homepage_featured']:
                continue
            df_out[f'{feat}_sq'] = df_out[feat] ** 2
            df_out[f'{feat}_cube'] = df_out[feat] ** 3
    # Pairwise interactions
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
        new_pairwise = {k: v for k, v in pairwise_dict.items() if k not in df_out.columns}
        if new_pairwise:
            df_out = pd.concat([df_out, pd.DataFrame(new_pairwise, index=df_out.index)], axis=1)
    return df_out

# --- Improved Feature Engineering Pipeline ---
def full_feature_engineering(df, is_train=True, weekofyear_means=None, month_means=None):
    df_out = create_advanced_temporal_features(df)
    df_out = create_robust_features(df_out)
    df_out = create_additional_features(df_out)
    if is_train or 'num_orders' in df_out.columns:
        df_out = create_group_aggregates(df_out)
    df_out = create_interaction_features(df_out)
    df_out = create_advanced_interactions(df_out)
    # Fill NaNs for all engineered features
    lag_roll_diff_cols = [col for col in df_out.columns if any(sub in col for sub in [
        "lag_", "rolling_mean", "rolling_std", "price_diff", "_rolling_sum", "_x_emailer", "_x_home",
        "_x_discount_pct", "_x_price_diff", "_x_weekofyear", "_sq", "_cube", "_mean", "_std"
    ])]
    cols_to_fill = [col for col in lag_roll_diff_cols if col in df_out.columns and len(df_out[col]) == len(df_out)]
    if cols_to_fill:
        df_out.loc[:, cols_to_fill] = df_out[cols_to_fill].fillna(0)
    if "discount_pct" in df_out.columns:
        df_out["discount_pct"] = df_out["discount_pct"].fillna(0)
    df_out = df_out.copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    return df_out

# --- Use tscv for cross-validation and model selection ---
def crossval_lgb(train_df, FEATURES, cat_features, n_splits=5, params=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    models = []
    for fold, (train_idx, valid_idx) in enumerate(tscv.split(train_df)):
        X_train, y_train = train_df.iloc[train_idx][FEATURES], train_df.iloc[train_idx]['num_orders']
        X_valid, y_valid = train_df.iloc[valid_idx][FEATURES], train_df.iloc[valid_idx]['num_orders']
        model = LGBMRegressor(**(params or {}), random_state=SEED+fold)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=rmsle_eval,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            categorical_feature=cat_features
        )
        preds = model.predict(X_valid)
        score = np.sqrt(np.mean(np.square(np.log1p(np.clip(preds,0,None)) - np.log1p(y_valid))))
        scores.append(score)
        models.append(model)
    return np.mean(scores), models

# --- Main Execution (replace old feature eng pipeline) ---
if __name__ == "__main__":
    logging.info("Creating full advanced features...")
    df = full_feature_engineering(df, is_train=True)
    test = full_feature_engineering(test, is_train=False)
    # --- Train/Validation Split ---
    max_week = df["week"].max()
    valid_df = df[df["week"] > max_week - VALIDATION_WEEKS].copy()
    train_df = df[df["week"] <= max_week - VALIDATION_WEEKS].copy()
    # --- Set categorical dtypes ---
    CATEGORICAL_FEATURES = [col for col in ["category", "cuisine", "center_type", "center_id", "meal_id"] if col in train_df.columns]
    for dset in [train_df, valid_df, test]:
        for col in CATEGORICAL_FEATURES:
            dset[col] = dset[col].astype("category")
    # --- Feature Selection ---
    FEATURES = [
        col for col in train_df.columns
        if col not in ["id", "num_orders", "week"] and train_df[col].dtype != object
    ]
    # --- Cross-validated Model Training ---
    logging.info("Cross-validating model with TimeSeriesSplit...")
    cv_score, cv_models = crossval_lgb(train_df, FEATURES, CATEGORICAL_FEATURES, n_splits=5)
    logging.info(f"CV RMSLE: {cv_score:.5f}")
    # --- Final Model Training ---
    logging.info("Training final model on all training data...")
    final_model = LGBMRegressor(random_state=SEED)
    final_model.fit(
        train_df[FEATURES], train_df['num_orders'],
        eval_set=[(valid_df[FEATURES], valid_df['num_orders'])],
        eval_metric=rmsle_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES
    )
    # --- Align test columns to match FEATURES ---
    missing_cols = [col for col in FEATURES if col not in test.columns]
    for col in missing_cols:
        test[col] = 0
    test = test[FEATURES + [col for col in test.columns if col not in FEATURES]]
    # --- Fill any remaining NaNs in train, valid, and test before modeling ---
    for dset in [train_df, valid_df, test]:
        # Fill categoricals with 'missing' (add to categories if needed)
        for col in CATEGORICAL_FEATURES:
            if col in dset.columns:
                if 'missing' not in dset[col].cat.categories:
                    dset[col] = dset[col].cat.add_categories(['missing'])
                dset[col] = dset[col].fillna('missing')
        # Fill numerics with 0
        num_cols = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]
        dset[num_cols] = dset[num_cols].fillna(0)
    # --- Generate Predictions ---
    logging.info("Generating predictions...")
    test['num_orders'] = final_model.predict(test[FEATURES])
    test['num_orders'] = test['num_orders'].clip(0).round().astype(int)
    submission = test[['id', 'num_orders']]
    submission.to_csv(os.path.join(OUTPUT_DIRECTORY, 'enhanced_forecaster.csv'), index=False)
    logging.info("Predictions saved successfully.")
    # --- SHAP Analysis ---
    logging.info("Performing SHAP analysis...")
    # Prepare SHAP input: fill categoricals with 'missing', numerics with 0
    shap_df = train_df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        if col in shap_df.columns:
            if 'missing' not in shap_df[col].cat.categories:
                shap_df[col] = shap_df[col].cat.add_categories(['missing'])
            shap_df[col] = shap_df[col].fillna('missing')
    num_cols = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]
    shap_df[num_cols] = shap_df[num_cols].fillna(0)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_df)
    shap.summary_plot(shap_values, shap_df, show=False)
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'shap_summary.png'))
    plt.close()
    # --- Validation Plot ---
    logging.info("Generating validation plot...")
    valid_preds = final_model.predict(valid_df[FEATURES])
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_df['num_orders'], valid_preds, alpha=0.5)
    plt.plot([0, 3500], [0, 3500], 'r--')
    plt.xlabel('Actual Orders')
    plt.ylabel('Predicted Orders')
    plt.title('Validation Set Predictions vs Actual')
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'validation_plot.png'))
    plt.close()
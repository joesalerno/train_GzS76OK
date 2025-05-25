import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import shap

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants from original file
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [2, 3, 5, 10, 14, 21]
OTHER_ROLLING_SUM_COLS = ["emailer_for_promotion", "homepage_featured"]
OTHER_ROLLING_SUM_WINDOW = 3
VALIDATION_WEEKS = 8
OPTUNA_TRIALS = 75
OPTUNA_STUDY_NAME = "feature_optimized"
SUBMISSION_FILE_PREFIX = "feature_optimized"

def load_data():
    """Load required data files."""
    try:
        df = pd.read_csv(DATA_PATH)
        test = pd.read_csv(TEST_PATH)
        meal_info = pd.read_csv(MEAL_INFO_PATH)
        center_info = pd.read_csv(CENTER_INFO_PATH)
        return df, test, meal_info, center_info
    except FileNotFoundError as e:
        logging.error(f"Error loading data file: {e}")
        raise

def preprocess_data(df, meal_info, center_info):
    """Merges dataframes and sorts."""
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

def create_simplified_features(df):
    """Create an optimized feature set based on dimensionality analysis."""
    df_out = df.copy()
    group = df_out.groupby(["center_id", "meal_id"])
    
    # 1. Create the most important lag features with minimal redundancy
    df_out["num_orders_lag_1"] = group["num_orders"].shift(1)
    df_out["num_orders_lag_3"] = group["num_orders"].shift(3)
    df_out["num_orders_lag_5"] = group["num_orders"].shift(5)
    df_out["num_orders_lag_10"] = group["num_orders"].shift(10)
    
    # 2. Create simplified rolling window features (only the most important ones)
    shifted = group["num_orders"].shift(1)
    for window in [3, 5, 14, 21]:
        df_out[f"num_orders_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
    
    # 3. Keep only the most important std features
    df_out["num_orders_rolling_std_14"] = shifted.rolling(14, min_periods=1).std().reset_index(drop=True)
    
    # 4. Price features (shown to be important in PCA)
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = group["checkout_price"].diff()
    
    # 5. Time features
    df_out["weekofyear"] = df_out["week"] % 52
    df_out["weekofyear_sin"] = np.sin(2 * np.pi * df_out["weekofyear"] / 52)
    df_out["weekofyear_cos"] = np.cos(2 * np.pi * df_out["weekofyear"] / 52)
    df_out["month"] = df_out["weekofyear"] // 4
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)
    
    # 6. Group aggregates (important according to SHAP values)
    # Center-level aggregates
    df_out['center_orders_mean'] = df_out.groupby('center_id')['num_orders'].transform('mean')
    df_out['center_orders_median'] = df_out.groupby('center_id')['num_orders'].transform('median')
    df_out['center_orders_std'] = df_out.groupby('center_id')['num_orders'].transform('std')
    
    # Meal-level aggregates
    df_out['meal_orders_mean'] = df_out.groupby('meal_id')['num_orders'].transform('mean')
    df_out['meal_orders_median'] = df_out.groupby('meal_id')['num_orders'].transform('median')
    df_out['meal_orders_std'] = df_out.groupby('meal_id')['num_orders'].transform('std')
    
    # Combined center-meal aggregates
    df_out['center_meal_orders_median_prod'] = df_out['center_orders_median'] * df_out['meal_orders_median']
    df_out['center_meal_orders_std_prod'] = df_out['center_orders_std'] * df_out['meal_orders_std']
    
    # 7. Category and center type encoding
    df_out = pd.get_dummies(df_out, columns=['category', 'cuisine', 'center_type'], drop_first=False)
    
    # 8. Keep only the most beneficial interaction features based on SHAP importance
    # High importance interactions with minimal redundancy
    df_out["lag1_x_rolling_mean_3"] = df_out["num_orders_lag_1"] * df_out["num_orders_rolling_mean_3"]
    df_out["rolling_mean_5_x_emailer"] = df_out["num_orders_rolling_mean_5"] * df_out["emailer_for_promotion"]
    df_out["price_diff_x_emailer"] = df_out["price_diff"] * df_out["emailer_for_promotion"]
    df_out["price_diff_x_home"] = df_out["price_diff"] * df_out["homepage_featured"]
    
    # 9. Cross-feature seasonality interactions (based on PCA)
    df_out["mean_orders_by_weekofyear"] = df_out.groupby("weekofyear")["num_orders"].transform("mean")
    df_out["mean_orders_by_month"] = df_out.groupby("month")["num_orders"].transform("mean")
    
    # 10. Add promotional features (important in PCA)
    for col in ["emailer_for_promotion", "homepage_featured"]:
        df_out[f"{col}_ewm_alpha_0.3"] = group[col].shift(1).ewm(alpha=0.3).mean().reset_index(drop=True)
        df_out[f"{col}_ewm_alpha_0.7"] = group[col].shift(1).ewm(alpha=0.7).mean().reset_index(drop=True)
    
    return df_out

def rmse(actual, predicted):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(actual, predicted))

def run_optimization():
    """Run the feature optimization experiment and evaluate results."""
    logging.info("Loading data...")
    df, test, meal_info, center_info = load_data()
    
    logging.info("Preprocessing data...")
    df = preprocess_data(df, meal_info, center_info)
    test = preprocess_data(test, meal_info, center_info)
    
    # Add placeholder for num_orders in test for alignment
    if 'num_orders' not in test.columns:
        test['num_orders'] = np.nan

    logging.info("Creating optimized features...")
    df = create_simplified_features(df)
    test = create_simplified_features(test)
    
    # Split train data for validation
    logging.info("Splitting data for validation...")
    max_week = df["week"].max()
    val_start_week = max_week - VALIDATION_WEEKS + 1
    train_df = df[df["week"] < val_start_week].copy()
    val_df = df[df["week"] >= val_start_week].copy()
    
    # Prepare features and target
    exclude_cols = ["id", "week", "center_id", "meal_id", "checkout_price", 
                    "base_price", "emailer_for_promotion", "homepage_featured", "num_orders"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Check for missing values and handle them
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        val_df[col] = val_df[col].fillna(train_df[col].median())
        test[col] = test[col].fillna(train_df[col].median())
    
    # Prepare datasets
    X_train = train_df[feature_cols]
    y_train = train_df["num_orders"]
    X_val = val_df[feature_cols]
    y_val = val_df["num_orders"]
    
    # Print feature info
    logging.info(f"Training with {len(feature_cols)} features after optimization")
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}")
    
    # Train a basic model without hyperparameter tuning for comparison
    logging.info("Training baseline model...")
    baseline_model = lgb.LGBMRegressor(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_val)
    baseline_rmse = rmse(y_val, baseline_preds)
    logging.info(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    # Feature importance from baseline model
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': baseline_model.feature_importances_
    }).sort_values('importance', ascending=False)
    logging.info("Top 20 important features from baseline model:")
    logging.info(feature_importance.head(20))
    
    # Save feature importance
    feature_importance.to_csv(f"{SUBMISSION_FILE_PREFIX}_feature_importance.csv", index=False)
    
    # Generate SHAP values
    logging.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(baseline_model)
    shap_values = explainer.shap_values(X_train.iloc[:2000])  # Sample for efficiency
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train.iloc[:2000], plot_type="bar", max_display=20)
    plt.savefig(f"{SUBMISSION_FILE_PREFIX}_shap_importance.png", bbox_inches='tight')

    # Get feature importances from SHAP
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'mean_abs_shap': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    feature_importances.to_csv(f"{SUBMISSION_FILE_PREFIX}_shap_importances.csv", index=False)
    
    logging.info("Top 20 SHAP feature importances:")
    logging.info(feature_importances.head(20))
    
    # Define final prediction function (simplified version)
    X_test = test[feature_cols]
    test_preds = baseline_model.predict(X_test)
    test["num_orders"] = test_preds
    
    # Create submission file
    submission = test[["id", "num_orders"]].copy()
    submission.to_csv(f"{SUBMISSION_FILE_PREFIX}_submission.csv", index=False)
    logging.info(f"Saved submission to {SUBMISSION_FILE_PREFIX}_submission.csv")
    
    return baseline_rmse

if __name__ == "__main__":
    logging.info("Starting feature optimization experiment...")
    rmse_score = run_optimization()
    logging.info(f"Feature optimization experiment completed with RMSE: {rmse_score:.4f}")

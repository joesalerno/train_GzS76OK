import pandas as pd
import numpy as np
import logging
import os
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import optuna
from optuna.samplers import TPESampler
from sklearn.cluster import KMeans

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 14, 21]
VALIDATION_WEEKS = 8
OPTUNA_TRIALS = 25  # Reduced for faster execution, increase for better results
OPTUNA_STUDY_NAME = "enhanced_optimized"
SUBMISSION_FILE_PREFIX = "enhanced_model"
SEED = 42

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

def create_enhanced_features(df):
    """Create an enhanced feature set based on dimensionality analysis and recommendations."""
    df_out = df.copy()
    group = df_out.groupby(["center_id", "meal_id"])
    
    # --- 1. CORE TEMPORAL FEATURES ---
    
    # Lag features - the most important core predictors
    for lag in LAG_WEEKS:
        df_out[f"num_orders_lag_{lag}"] = group["num_orders"].shift(lag)
    
    # Rolling window features - simplified to reduce redundancy
    shifted = group["num_orders"].shift(1)  # Shift by 1 to avoid data leakage
    for window in ROLLING_WINDOWS:
        df_out[f"num_orders_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(drop=True)
        # Keep only the most valuable std features
        if window in [14]:  # Only create std for key windows
            df_out[f"num_orders_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(drop=True)
    
    # Add momentum/trend features
    df_out["orders_momentum"] = df_out["num_orders_rolling_mean_5"] - df_out["num_orders_rolling_mean_14"]
    df_out["orders_trend_percent"] = df_out["orders_momentum"] / df_out["num_orders_rolling_mean_14"].replace(0, np.nan) * 100
    
    # --- 2. PRICE FEATURES ---
    
    # Basic price features
    df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
    df_out["discount_pct"] = df_out["discount"] / df_out["base_price"].replace(0, np.nan)
    df_out["price_diff"] = group["checkout_price"].diff()
    df_out["price_ratio"] = df_out["checkout_price"] / df_out["base_price"].replace(0, np.nan)
    
    # --- 3. TIME/SEASONALITY FEATURES ---
    
    # Week and month cyclical features
    df_out["weekofyear"] = df_out["week"] % 52
    df_out["weekofyear_sin"] = np.sin(2 * np.pi * df_out["weekofyear"] / 52)
    df_out["weekofyear_cos"] = np.cos(2 * np.pi * df_out["weekofyear"] / 52)
    df_out["month"] = df_out["weekofyear"] // 4
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)
      # --- 4. GROUP AGGREGATES ---
    
    # Center-level aggregates
    df_out['center_orders_mean'] = df_out.groupby('center_id')['num_orders'].transform('mean')
    df_out['center_orders_median'] = df_out.groupby('center_id')['num_orders'].transform('median')
    df_out['center_orders_std'] = df_out.groupby('center_id')['num_orders'].transform('std')
    
    # Meal-level aggregates
    df_out['meal_orders_mean'] = df_out.groupby('meal_id')['num_orders'].transform('mean')
    df_out['meal_orders_median'] = df_out.groupby('meal_id')['num_orders'].transform('median')
    df_out['meal_orders_std'] = df_out.groupby('meal_id')['num_orders'].transform('std')
    
    # Category-level aggregates (only if category column exists)
    if 'category' in df_out.columns:
        df_out['category_orders_mean'] = df_out.groupby('category')['num_orders'].transform('mean')
    
    # Center-meal combined aggregates (high SHAP importance)
    df_out['center_meal_orders_median_prod'] = df_out['center_orders_median'] * df_out['meal_orders_median']
    df_out['center_meal_orders_std_prod'] = df_out['center_orders_std'] * df_out['meal_orders_std']
    df_out['center_meal_ratio'] = df_out['center_orders_mean'] / df_out['meal_orders_mean'].replace(0, np.nan)
      # --- 5. SEASONAL AGGREGATES ---
    
    # Week and month aggregates
    df_out["mean_orders_by_weekofyear"] = df_out.groupby("weekofyear")["num_orders"].transform("mean")
    df_out["mean_orders_by_month"] = df_out.groupby("month")["num_orders"].transform("mean")
    
    # Category x Seasonality interactions (only if category column exists)
    if 'category' in df_out.columns:
        df_out["category_weekofyear_mean"] = df_out.groupby(["category", "weekofyear"])["num_orders"].transform("mean")
    
    # --- 6. PROMOTIONAL FEATURES ---
    
    # Exponentially weighted moving averages for promotions
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted_promo = group[col].shift(1)  # Avoid data leakage
        df_out[f"{col}_ewm_alpha_0.3"] = shifted_promo.ewm(alpha=0.3).mean().reset_index(drop=True)
        df_out[f"{col}_ewm_alpha_0.7"] = shifted_promo.ewm(alpha=0.7).mean().reset_index(drop=True)
    
    # Combined promotional indicator
    df_out["emailer_homepage_combined"] = df_out["emailer_for_promotion"] + df_out["homepage_featured"]
    
    # --- 7. HIGH-VALUE INTERACTION FEATURES ---
    
    # Keep only the most beneficial interactions based on SHAP values
    df_out["lag1_x_rolling_mean_3"] = df_out["num_orders_lag_1"] * df_out["num_orders_rolling_mean_3"]
    df_out["lag1_x_rolling_mean_2"] = df_out["num_orders_lag_1"] * df_out["num_orders_lag_2"]
    df_out["rolling_mean_2_x_rolling_mean_3"] = df_out["num_orders_lag_2"] * df_out["num_orders_rolling_mean_3"]
    df_out["price_diff_x_emailer"] = df_out["price_diff"] * df_out["emailer_for_promotion"]
    df_out["price_diff_x_home"] = df_out["price_diff"] * df_out["homepage_featured"]
    df_out["rolling_mean_5_x_emailer"] = df_out["num_orders_rolling_mean_5"] * df_out["emailer_for_promotion"]
      # --- 8. CATEGORY & CENTER ENCODING ---
    
    # One-hot encode categorical features if they exist
    categorical_cols = ['category', 'cuisine', 'center_type']
    cols_to_encode = [col for col in categorical_cols if col in df_out.columns]
    if cols_to_encode:
        df_out = pd.get_dummies(df_out, columns=cols_to_encode, drop_first=False)
    
    return df_out

def prepare_train_val_sets(df, val_weeks=VALIDATION_WEEKS):
    """Split data into training and validation sets."""
    max_week = df["week"].max()
    val_start_week = max_week - val_weeks + 1
    train_df = df[df["week"] < val_start_week].copy()
    val_df = df[df["week"] >= val_start_week].copy()
    return train_df, val_df

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0) # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False # lower is better

def objective(trial, X_train, y_train, X_val, y_val, feature_cols):    
    """Optuna objective function for hyperparameter tuning."""
    param = {
        'objective': 'regression_l1',  # MAE objective often works well for RMSLE
        'metric': 'None',  # Using custom metric
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 7, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
      # Simple feature selection with Optuna
    sample_rate = trial.suggest_float('feature_sample_rate', 0.7, 1.0)
    n_features = int(len(feature_cols) * sample_rate)
    selected_features = np.random.choice(feature_cols, n_features, replace=False)
    
    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_train[selected_features], y_train,
        eval_set=[(X_val[selected_features], y_val)],
        eval_metric=lgb_rmsle,  # Use custom RMSLE metric
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmsle'),  # Pruning based on validation RMSLE
                  lgb.early_stopping(100, verbose=False)]  # Early stopping
    )
    preds = model.predict(X_val[selected_features])
    
    # Calculate and return the validation RMSLE
    score = rmsle(y_val, preds)
    
    # Save feature importance for this trial
    trial.set_user_attr('feature_importance', 
                       {feat: imp for feat, imp in zip(selected_features, model.feature_importances_)})
    
    return score

def run_enhanced_model():
    """Run the enhanced model with optimized features and hyperparameter tuning."""
    logging.info("Loading data...")
    df, test, meal_info, center_info = load_data()
    
    logging.info("Preprocessing data...")
    df = preprocess_data(df, meal_info, center_info)
    test = preprocess_data(test, meal_info, center_info)
    
    # Add placeholder for num_orders in test for alignment
    if 'num_orders' not in test.columns:
        test['num_orders'] = np.nan

    logging.info("Creating enhanced features...")
    df = create_enhanced_features(df)
    test = create_enhanced_features(test)
    
    # Split train and validation data
    train_df, val_df = prepare_train_val_sets(df)
    
    # Prepare features and target
    exclude_cols = ["id", "week", "center_id", "meal_id", "checkout_price", 
                    "base_price", "emailer_for_promotion", "homepage_featured", 
                    "num_orders", "cuisine", "category", "center_type"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Check for missing values and handle them
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(train_df[col].median())
            val_df[col] = val_df[col].fillna(train_df[col].median())
            test[col] = test[col].fillna(train_df[col].median())
        else:
            logging.warning(f"Column {col} not found in train_df")
    
    # Prepare datasets
    X_train = train_df[feature_cols]
    y_train = train_df["num_orders"]
    X_val = val_df[feature_cols]
    y_val = val_df["num_orders"]
    
    # Log feature information
    logging.info(f"Training with {len(feature_cols)} features after enhancements")
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}")
      # Create and run Optuna study (using RMSLE as the objective)
    logging.info("Starting hyperparameter optimization with RMSLE objective...")
    study = optuna.create_study(direction='minimize', 
                               sampler=TPESampler(seed=SEED),
                               study_name=f"{OPTUNA_STUDY_NAME}")
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, feature_cols), 
                  n_trials=OPTUNA_TRIALS)
    
    # Get best hyperparameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    logging.info(f"Best trial: {best_trial.number}, RMSE: {best_trial.value:.4f}")
    logging.info("Best hyperparameters:")
    for param, value in best_params.items():
        logging.info(f"    {param}: {value}")
    
    # Train final model with best hyperparameters
    logging.info("Training final model with best hyperparameters...")
    best_sample_rate = best_params.pop('feature_sample_rate', 1.0)
    n_features = int(len(feature_cols) * best_sample_rate)
    
    if n_features < len(feature_cols):
        # Get feature importances across trials
        feature_imp_dict = {}
        for trial in study.trials:
            if 'feature_importance' in trial.user_attrs:
                for feat, imp in trial.user_attrs['feature_importance'].items():
                    if feat not in feature_imp_dict:
                        feature_imp_dict[feat] = []
                    feature_imp_dict[feat].append(imp)
        
        # Calculate average importance
        avg_importances = {feat: np.mean(imps) for feat, imps in feature_imp_dict.items() if len(imps) > 0}
        # Sort features by importance
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
        # Select top features
        selected_features = [f[0] for f in sorted_features[:n_features]]
        logging.info(f"Selected {len(selected_features)} features based on importance")
    else:
        selected_features = feature_cols
      # Train final model
    final_model = lgb.LGBMRegressor(**best_params, seed=SEED)
    final_model.fit(
        X_train[selected_features], y_train, 
        eval_set=[(X_val[selected_features], y_val)],
        eval_metric=lgb_rmsle,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
      # Evaluate on validation set
    val_preds = final_model.predict(X_val[selected_features])
    val_rmse = rmse(y_val, val_preds)
    val_rmsle = rmsle(y_val, val_preds)
    logging.info(f"Validation RMSE: {val_rmse:.4f}")
    logging.info(f"Validation RMSLE: {val_rmsle:.4f}")
    
    # Feature importance from final model
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    logging.info("Top 20 important features from final model:")
    logging.info(feature_importance.head(20))
    
    # Save feature importance
    feature_importance.to_csv(f"{SUBMISSION_FILE_PREFIX}_feature_importance.csv", index=False)
    
    # Generate SHAP values
    logging.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train[selected_features].iloc[:2000])  # Sample for efficiency
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train[selected_features].iloc[:2000], plot_type="bar", max_display=20)
    plt.savefig(f"{SUBMISSION_FILE_PREFIX}_shap_importance.png", bbox_inches='tight')
    plt.close()
    
    # Get feature importances from SHAP
    feature_importances = pd.DataFrame({
        'feature': selected_features,
        'mean_abs_shap': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    feature_importances.to_csv(f"{SUBMISSION_FILE_PREFIX}_shap_importances.csv", index=False)
    
    # SHAP values for analysis
    shap_values_df = pd.DataFrame(shap_values, columns=selected_features)
    shap_values_df.to_csv(f"{SUBMISSION_FILE_PREFIX}_shap_values.csv", index=False)
    
    logging.info("Top 20 SHAP feature importances:")
    logging.info(feature_importances.head(20))    # --- Recursive Prediction ---
    logging.info("Starting recursive prediction on the test set...")
    # Prepare the combined data history (training data + test structure)
    # We need the structure of test_df but will fill num_orders recursively
    
    # Important: make a copy of the raw dataframes before feature engineering
    raw_df = df.copy()
    raw_test = test.copy()
    
    # Extract original categorical columns before feature engineering transforms them
    category_cols = ['category', 'cuisine', 'center_type']
    original_cat_data = {}
    for col in category_cols:
        if col in raw_df.columns:
            # Store the mapping from id to category value
            original_cat_data[col] = pd.concat([
                raw_df[['id', col]], 
                raw_test[['id', col]]
            ]).drop_duplicates()
    
    # Combine train and test data for recursive prediction
    history_df = pd.concat([raw_df, raw_test], ignore_index=True).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    
    test_weeks = sorted(raw_test['week'].unique())

    for week_num in test_weeks:
        logging.info(f"Predicting for week {week_num}...")
        # Identify rows for the current week to predict
        current_week_mask = history_df['week'] == week_num
        
        # Create features for the entire history_df
        temp_df = create_enhanced_features(history_df)        # Prepare features for current week
        current_features = temp_df.loc[current_week_mask, selected_features]
        
        # Handle potential missing features
        missing_cols = [col for col in selected_features if col not in current_features.columns]
        if missing_cols:
            logging.warning(f"Missing columns during prediction for week {week_num}: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                current_features[col] = 0
        
        # Ensure features are in the right order
        current_features = current_features[selected_features]

        # Predict for the current week
        current_preds = final_model.predict(current_features)
        current_preds = np.clip(current_preds, 0, None) # Ensure predictions are non-negative

        # Update the 'num_orders' in history_df for the current week with predictions
        # This ensures the next iteration uses the predicted values for lag/rolling features
        history_df.loc[current_week_mask, 'num_orders'] = current_preds

    logging.info("Recursive prediction finished.")

    # Extract final predictions for the original test set IDs
    final_predictions_df = history_df.loc[history_df['id'].isin(test['id']), ['id', 'num_orders']].copy()
    final_predictions_df['num_orders'] = final_predictions_df['num_orders'].round().astype(int)  # Convert to integers
    
    # Create submission file
    submission_file = f"{SUBMISSION_FILE_PREFIX}_submission.csv"
    final_predictions_df.to_csv(submission_file, index=False)
    logging.info(f"Saved submission to {submission_file}")
    
    # Create performance report file
    performance = {
        "rmse": val_rmse,
        "rmsle": val_rmsle
    }
    
    pd.DataFrame([performance]).to_csv(f"{SUBMISSION_FILE_PREFIX}_performance_metrics.csv", index=False)
    logging.info(f"Saved performance metrics to {SUBMISSION_FILE_PREFIX}_performance_metrics.csv")
    
    return val_rmse, val_rmsle

if __name__ == "__main__":
    logging.info("Starting enhanced model training with RMSLE optimization...")
    rmse_score, rmsle_score = run_enhanced_model()
    logging.info(f"Enhanced model training completed with validation RMSE: {rmse_score:.4f}, RMSLE: {rmsle_score:.4f}")

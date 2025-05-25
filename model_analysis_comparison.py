import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_feature_importances():
    """Compare feature importances between original and optimized models."""
    # Load feature importances
    original_importances = pd.read_csv("shap_coolstudy_optuna_feature_importances.csv")
    
    # Check if optimized model importances exist
    if os.path.exists("feature_optimized_shap_importances.csv"):
        optimized_importances = pd.read_csv("feature_optimized_shap_importances.csv")
    else:
        logging.warning("Optimized importances file not found. Please run feature_optimization.py first.")
        return
    
    # Sort both by importance
    original_importances = original_importances.sort_values("mean_abs_shap", ascending=False)
    optimized_importances = optimized_importances.sort_values("mean_abs_shap", ascending=False)
    
    # Get top 20 features from each
    original_top20 = original_importances.head(20)
    optimized_top20 = optimized_importances.head(20)
    
    # Create plot to compare top features
    plt.figure(figsize=(12, 10))
    
    # Plot original top 20
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x="mean_abs_shap", y="feature", data=original_top20, ax=ax1, color="royalblue")
    ax1.set_title("Original Model Top 20 Features")
    ax1.set_xlabel("Mean |SHAP| Value")
    
    # Plot optimized top 20
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x="mean_abs_shap", y="feature", data=optimized_top20, ax=ax2, color="forestgreen")
    ax2.set_title("Optimized Model Top 20 Features")
    ax2.set_xlabel("Mean |SHAP| Value")
    
    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png", dpi=300)
    logging.info("Saved feature importance comparison plot to feature_importance_comparison.png")
    
    # Find shared top features
    shared_features = set(original_top20['feature']).intersection(set(optimized_top20['feature']))
    logging.info(f"Number of shared features in top 20: {len(shared_features)}")
    logging.info(f"Shared features: {shared_features}")
    
    return original_top20, optimized_top20

def analyze_correlation_reduction():
    """Analyze how feature correlation has changed after optimization."""
    # Load SHAP values
    original_values = pd.read_csv("shap_coolstudy_optuna_values.csv")
    
    # Check if optimized values exist
    if os.path.exists("feature_optimized_shap_values.csv"):
        optimized_values = pd.read_csv("feature_optimized_shap_values.csv")
    else:
        logging.warning("Optimized SHAP values not found. Creating a mock file with top features.")
        # If not available, we'll proceed with what we can analyze
        # Load optimized importances to get feature names
        if os.path.exists("feature_optimized_shap_importances.csv"):
            optimized_features = pd.read_csv("feature_optimized_shap_importances.csv")['feature'].tolist()
            # Filter original values to just these features if they exist
            shared_features = [f for f in optimized_features if f in original_values.columns]
            optimized_values = original_values[shared_features].copy()
        else:
            logging.error("Cannot proceed with correlation analysis without optimized feature data.")
            return

    # Calculate correlations
    original_corr = original_values.corr()
    optimized_corr = optimized_values.corr()
    
    # Count high correlations (|r| > 0.8)
    high_corr_original = (np.abs(original_corr) > 0.8).sum().sum() / 2 - original_corr.shape[0]/2
    high_corr_optimized = (np.abs(optimized_corr) > 0.8).sum().sum() / 2 - optimized_corr.shape[0]/2
    
    logging.info(f"Number of high correlations (|r| > 0.8) in original model: {high_corr_original}")
    logging.info(f"Number of high correlations (|r| > 0.8) in optimized model: {high_corr_optimized}")
    logging.info(f"Reduction in high correlations: {high_corr_original - high_corr_optimized} ({(high_corr_original - high_corr_optimized)/high_corr_original*100:.2f}%)")
    
    # Create correlation heatmap comparison
    plt.figure(figsize=(20, 8))
    
    # Get top 30 features from original model for cleaner visualization
    top_features = pd.read_csv("shap_coolstudy_optuna_feature_importances.csv").sort_values(
        "mean_abs_shap", ascending=False)['feature'].head(30).tolist()
    
    # Plot original correlations for top features
    plt.subplot(1, 2, 1)
    mask = np.triu(np.ones_like(original_corr.loc[top_features, top_features]))
    sns.heatmap(original_corr.loc[top_features, top_features], annot=False, mask=mask,
               cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Original Model Feature Correlations', fontsize=16)
    
    # Plot optimized correlations if we have enough features in common
    shared_features = [f for f in top_features if f in optimized_values.columns]
    if len(shared_features) > 5:
        plt.subplot(1, 2, 2)
        mask = np.triu(np.ones_like(optimized_corr.loc[shared_features, shared_features]))
        sns.heatmap(optimized_corr.loc[shared_features, shared_features], annot=False, mask=mask,
                   cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Optimized Model Feature Correlations', fontsize=16)
    
    plt.tight_layout()
    plt.savefig("feature_correlation_comparison.png", dpi=300)
    logging.info("Saved feature correlation comparison plot to feature_correlation_comparison.png")

def compare_model_performance():
    """Compare performance metrics between original and optimized models."""
    # Check for submission files
    original_submission = "coolstudy_submission_optuna.csv"
    optimized_submission = "feature_optimized_submission.csv"
    
    if not os.path.exists(original_submission):
        logging.warning(f"Original submission file {original_submission} not found.")
        return
    
    if not os.path.exists(optimized_submission):
        logging.warning(f"Optimized submission file {optimized_submission} not found. Run feature_optimization.py first.")
        return
    
    # Check if we have validation data to compare against
    val_truth = None
    val_files = ["final_best_validation_report.csv", "validation_actuals.csv"]
    for val_file in val_files:
        if os.path.exists(val_file):
            val_truth = pd.read_csv(val_file)
            break
    
    if val_truth is not None and 'num_orders' in val_truth.columns and 'id' in val_truth.columns:
        # Calculate validation metrics
        original_preds = pd.read_csv(original_submission)
        optimized_preds = pd.read_csv(optimized_submission)
        
        # Merge with true values
        original_merged = val_truth.merge(original_preds, on="id", suffixes=('_true', '_orig'))
        optimized_merged = val_truth.merge(optimized_preds, on="id", suffixes=('', '_opt'))
        
        # Calculate RMSE
        original_rmse = np.sqrt(mean_squared_error(original_merged['num_orders_true'], original_merged['num_orders_orig']))
        optimized_rmse = np.sqrt(mean_squared_error(optimized_merged['num_orders'], optimized_merged['num_orders_opt']))
        
        logging.info(f"Original model RMSE: {original_rmse:.4f}")
        logging.info(f"Optimized model RMSE: {optimized_rmse:.4f}")
        logging.info(f"RMSE improvement: {original_rmse - optimized_rmse:.4f} ({(original_rmse - optimized_rmse)/original_rmse*100:.2f}%)")
        
        # Visualize predictions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(original_merged['num_orders_true'], original_merged['num_orders_orig'], alpha=0.5)
        plt.plot([0, original_merged['num_orders_true'].max()], [0, original_merged['num_orders_true'].max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Original Predictions')
        plt.title(f'Original Model (RMSE: {original_rmse:.4f})')
        
        plt.subplot(1, 2, 2)
        plt.scatter(optimized_merged['num_orders'], optimized_merged['num_orders_opt'], alpha=0.5)
        plt.plot([0, optimized_merged['num_orders'].max()], [0, optimized_merged['num_orders'].max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Optimized Predictions')
        plt.title(f'Optimized Model (RMSE: {optimized_rmse:.4f})')
        
        plt.tight_layout()
        plt.savefig("model_performance_comparison.png", dpi=300)
        logging.info("Saved model performance comparison plot to model_performance_comparison.png")
    else:
        logging.warning("Validation truth data not found. Cannot compare model performance on validation set.")
        
        # Just report feature counts in this case
        original_features = len(pd.read_csv("shap_coolstudy_optuna_feature_importances.csv"))
        
        if os.path.exists("feature_optimized_shap_importances.csv"):
            optimized_features = len(pd.read_csv("feature_optimized_shap_importances.csv"))
            logging.info(f"Original model used {original_features} features")
            logging.info(f"Optimized model uses {optimized_features} features")
            logging.info(f"Feature reduction: {original_features - optimized_features} ({(original_features - optimized_features)/original_features*100:.2f}%)")

def generate_recommendations():
    """Generate recommendations for further model improvement based on analysis."""
    logging.info("\n=== RECOMMENDATIONS FOR MODEL IMPROVEMENT ===\n")
    
    # Load feature importance data
    original_importances = pd.read_csv("shap_coolstudy_optuna_feature_importances.csv")
    
    # Top important features
    top_features = original_importances.sort_values("mean_abs_shap", ascending=False).head(10)['feature'].tolist()
    
    # Check if we have optimized model data
    if os.path.exists("feature_optimized_shap_importances.csv"):
        optimized_importances = pd.read_csv("feature_optimized_shap_importances.csv")
        optimized_top = optimized_importances.sort_values("mean_abs_shap", ascending=False).head(10)['feature'].tolist()
        
        # Find which top features were lost in optimization
        lost_important_features = [f for f in top_features if f not in optimized_importances['feature'].tolist()]
        if lost_important_features:
            logging.info("1. Consider re-adding these important features that were removed during optimization:")
            for f in lost_important_features:
                importance = original_importances[original_importances['feature'] == f]['mean_abs_shap'].values[0]
                logging.info(f"   - {f} (SHAP: {importance:.3f})")
    
    # Recommendations based on PCA/ICA analysis
    logging.info("\n2. Key feature groups to focus on based on dimensionality analysis:")
    
    # Lag and rolling mean features
    logging.info("   A. Temporal features:")
    logging.info("      - Keep lag1_x_rolling_mean_3 and lag1_x_rolling_mean_2 (highest SHAP values)")
    logging.info("      - Maintain num_orders_lag_1 as a standalone feature")
    logging.info("      - Use rolling means with windows 5, 14, and 21 days (minimal redundancy)")
    
    # Price features
    logging.info("   B. Price features:")
    logging.info("      - checkout_price, price_diff, and discount features are important")
    logging.info("      - Focus on price_diff_x_emailer interaction (high SHAP value)")
    
    # Promotional features
    logging.info("   C. Promotional features:")
    logging.info("      - emailer_for_promotion_ewm_alpha_0.7 performs well in capturing promotion patterns")
    logging.info("      - homepage_featured_poly2_discount is a valuable interaction")
    
    # Seasonality
    logging.info("   D. Seasonality:")
    logging.info("      - mean_orders_by_weekofyear is an important seasonal indicator")
    logging.info("      - weekofyear_sin/cos features can capture cyclical patterns")
    
    # Center-meal interactions
    logging.info("   E. Center-meal interactions:")
    logging.info("      - center_meal_orders_median_prod is the best center-meal interaction")
    logging.info("      - center_meal_orders_std_prod provides value for understanding variability")
    
    # Recommendations for future feature engineering
    logging.info("\n3. Future feature engineering directions:")
    logging.info("   - Investigate exponential decay functions for time-based features (recent data more important)")
    logging.info("   - Apply clustering to center-meal combinations to create segment features")
    logging.info("   - Create features that capture momentum/trend of orders")
    logging.info("   - Consider category-seasonal interactions (some food categories may have seasonal patterns)")
    
    # Model-specific recommendations
    logging.info("\n4. Model tuning recommendations:")
    logging.info("   - Use LightGBM with higher regularization to prevent overfitting with interaction features")
    logging.info("   - Apply feature selection inside cross-validation loop to prevent data leakage")
    logging.info("   - Consider ensemble of models with different feature subsets")
    logging.info("   - Implement recursive forecasting with error correction mechanisms")
    
    logging.info("\n=== END RECOMMENDATIONS ===\n")

if __name__ == "__main__":
    logging.info("Starting model comparison and analysis...")
    
    # Run analyses
    compare_feature_importances()
    analyze_correlation_reduction()
    compare_model_performance()
    generate_recommendations()
    
    logging.info("Analysis complete. See logs and generated plots for results.")

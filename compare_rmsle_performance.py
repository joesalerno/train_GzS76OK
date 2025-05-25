import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    # Add 1 to avoid taking log of zero
    y_true = np.maximum(0, y_true) + 1
    y_pred = np.maximum(0, y_pred) + 1
    return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compare_model_performance():
    """Compare performance between original model and enhanced model."""
    logging.info("Comparing model performance...")
      # Check for submission files
    original_submissions = ["coolstudy_submission_optuna.csv", "newertest_submission_optuna.csv", "newtest_optuna.csv"]
    enhanced_submission = "enhanced_model_submission.csv"
    
    if not os.path.exists(enhanced_submission):
        logging.warning(f"Enhanced model submission file {enhanced_submission} not found. Please run enhanced_model.py first.")
        return
    
    # Check if we have validation data to compare against
    val_truth = None
    val_files = ["final_best_validation_report.csv", "validation_actuals.csv"]
    for val_file in val_files:
        if os.path.exists(val_file):
            val_truth = pd.read_csv(val_file)
            break
    
    if val_truth is None:
        logging.warning("Validation truth data not found. Cannot compare model performance on validation set.")
        return
    
    if 'num_orders' not in val_truth.columns or 'id' not in val_truth.columns:
        logging.warning("Validation data does not have required columns (id, num_orders).")
        return
    
    # Load enhanced model predictions
    enhanced_preds = pd.read_csv(enhanced_submission)
    
    # Try to find a valid original model submission
    original_preds = None
    original_submission = None
    
    for submission in original_submissions:
        if os.path.exists(submission):
            original_preds = pd.read_csv(submission)
            original_submission = submission
            break
    
    if original_preds is None:
        logging.warning("Original model submission file not found. Will only evaluate enhanced model.")
    
    # Calculate metrics
    results = []
    
    # Evaluate enhanced model
    merged = val_truth.merge(enhanced_preds, on="id", suffixes=('_true', '_pred'))
    enhanced_rmse = rmse(merged['num_orders_true'], merged['num_orders_pred'])
    enhanced_rmsle = rmsle(merged['num_orders_true'], merged['num_orders_pred'])
    
    results.append({
        'model': 'Enhanced Model (RMSLE Optimized)',
        'rmse': enhanced_rmse,
        'rmsle': enhanced_rmsle
    })
    
    logging.info(f"Enhanced Model - RMSE: {enhanced_rmse:.4f}, RMSLE: {enhanced_rmsle:.4f}")
    
    # If we have original model predictions, evaluate those too
    if original_preds is not None:
        merged = val_truth.merge(original_preds, on="id", suffixes=('_true', '_pred'))
        original_rmse = rmse(merged['num_orders_true'], merged['num_orders_pred'])
        original_rmsle = rmsle(merged['num_orders_true'], merged['num_orders_pred'])
        
        results.append({
            'model': f'Original Model ({original_submission})',
            'rmse': original_rmse,
            'rmsle': original_rmsle
        })
        
        logging.info(f"Original Model - RMSE: {original_rmse:.4f}, RMSLE: {original_rmsle:.4f}")
        
        # Calculate improvement
        rmse_improvement = (original_rmse - enhanced_rmse) / original_rmse * 100
        rmsle_improvement = (original_rmsle - enhanced_rmsle) / original_rmsle * 100
        
        logging.info(f"RMSE improvement: {rmse_improvement:.2f}%")
        logging.info(f"RMSLE improvement: {rmsle_improvement:.2f}%")
    
    # Create results DataFrame and save it
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison_results.csv", index=False)
    logging.info("Saved comparison results to model_comparison_results.csv")
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # RMSE comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x='model', y='rmse', data=results_df)
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # RMSLE comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x='model', y='rmsle', data=results_df)
    plt.title('RMSLE Comparison')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("model_metric_comparison.png", dpi=300)
    logging.info("Saved comparison plot to model_metric_comparison.png")

if __name__ == "__main__":
    compare_model_performance()

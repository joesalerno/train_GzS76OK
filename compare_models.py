import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_submissions():
    """Compare submission files from different models."""
    try:
        logging.info("Comparing model predictions...")
        
        # Check for submission files
        original_submissions = ["coolstudy_submission_optuna.csv", "newertest_submission_optuna.csv", "newtest_optuna.csv"]
        enhanced_submission = "enhanced_model_submission.csv"
        
        logging.info(f"Looking for enhanced submission: {enhanced_submission}")
        if not os.path.exists(enhanced_submission):
            logging.warning(f"Enhanced model submission file {enhanced_submission} not found.")
            return
        
        # Try to find a valid original model submission
        original_submission = None
        for submission in original_submissions:
            logging.info(f"Checking for original submission: {submission}")
            if os.path.exists(submission):
                original_submission = submission
                break
        
        if original_submission is None:
            logging.warning("Original model submission file not found.")
            return
        
        # Load submissions
        logging.info(f"Loading enhanced predictions from: {enhanced_submission}")
        enhanced_preds = pd.read_csv(enhanced_submission)
        logging.info(f"Enhanced predictions shape: {enhanced_preds.shape}")
        
        logging.info(f"Loading original predictions from: {original_submission}")
        original_preds = pd.read_csv(original_submission)
        logging.info(f"Original predictions shape: {original_preds.shape}")
        
        # Merge for comparison
        logging.info("Merging predictions for comparison...")
        merged = original_preds.merge(enhanced_preds, on="id", suffixes=('_original', '_enhanced'))
        logging.info(f"Merged shape: {merged.shape}")
        
        # Calculate metrics
        logging.info("Calculating comparison metrics...")
        mean_abs_diff = np.mean(np.abs(merged['num_orders_original'] - merged['num_orders_enhanced']))
        mean_rel_diff = np.mean(np.abs(merged['num_orders_original'] - merged['num_orders_enhanced']) / 
                              (merged['num_orders_original'] + 1)) * 100  # Add 1 to avoid division by zero
        
        logging.info(f"Original submission: {original_submission}")
        logging.info(f"Enhanced submission: {enhanced_submission}")
        logging.info(f"Mean absolute difference: {mean_abs_diff:.2f} orders")
        logging.info(f"Mean relative difference: {mean_rel_diff:.2f}%")
        
        # Plot comparison
        plt.figure(figsize=(10, 8))
        plt.scatter(merged['num_orders_original'], merged['num_orders_enhanced'], alpha=0.3)
        
        # Add perfect prediction line
        max_val = max(merged['num_orders_original'].max(), merged['num_orders_enhanced'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        plt.title(f'Original vs Enhanced Model Predictions\nMean Abs Diff: {mean_abs_diff:.2f}, Mean Rel Diff: {mean_rel_diff:.2f}%')
        plt.xlabel('Original Model Predictions')
        plt.ylabel('Enhanced Model Predictions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        
        # Check distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(merged['num_orders_original'], kde=True, color='blue', bins=50)
        plt.title('Original Model Predictions')
        plt.xlabel('Number of Orders')
        
        plt.subplot(1, 2, 2)
        sns.histplot(merged['num_orders_enhanced'], kde=True, color='green', bins=50)
        plt.title('Enhanced Model Predictions')
        plt.xlabel('Number of Orders')
        
        plt.tight_layout()
        plt.savefig('prediction_distributions.png')
        
        logging.info("Comparison plots saved to model_comparison.png and prediction_distributions.png")
        
        # Calculate correlation
        corr = merged['num_orders_original'].corr(merged['num_orders_enhanced'])
        logging.info(f"Correlation between predictions: {corr:.4f}")
        
        # Calculate performance metrics on our validation set
        valid_metrics = pd.read_csv("enhanced_model_performance_metrics.csv")
        logging.info(f"Enhanced model performance metrics:")
        logging.info(f"  RMSE: {valid_metrics['rmse'].values[0]:.4f}")
        logging.info(f"  RMSLE: {valid_metrics['rmsle'].values[0]:.4f}")
    except Exception as e:
        logging.error(f"Error during comparison: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        compare_submissions()
    except Exception as e:
        logging.error(f"Error during model comparison: {e}", exc_info=True)

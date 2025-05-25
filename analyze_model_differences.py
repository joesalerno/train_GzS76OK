import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_prediction_differences():
    """Perform detailed analysis of differences between model predictions."""
    try:
        logging.info("Analyzing prediction differences between models...")
        
        # Check for submission files
        original_submission = "coolstudy_submission_optuna.csv"
        enhanced_submission = "enhanced_model_submission.csv"
        
        if not os.path.exists(enhanced_submission) or not os.path.exists(original_submission):
            logging.warning(f"One or more submission files not found.")
            return
        
        # Load test data to get additional features for analysis
        test_df = pd.read_csv("test.csv")
        meal_info = pd.read_csv("meal_info.csv")
        center_info = pd.read_csv("fulfilment_center_info.csv")
        
        # Merge test with additional info
        test_df = test_df.merge(meal_info, on="meal_id", how="left")
        test_df = test_df.merge(center_info, on="center_id", how="left")
        
        # Load predictions
        enhanced_preds = pd.read_csv(enhanced_submission)
        original_preds = pd.read_csv(original_submission)
        
        # Merge predictions
        merged = original_preds.merge(enhanced_preds, on="id", suffixes=('_original', '_enhanced'))
        
        # Add additional information for analysis
        merged = merged.merge(test_df, on="id", how="left")
        
        # Calculate difference
        merged['abs_diff'] = np.abs(merged['num_orders_original'] - merged['num_orders_enhanced'])
        merged['rel_diff'] = merged['abs_diff'] / (merged['num_orders_original'] + 1) * 100
        
        # Group by different categories and analyze differences
        logging.info("\nAnalysis by center type:")
        center_analysis = merged.groupby('center_type').agg({
            'abs_diff': 'mean',
            'rel_diff': 'mean',
            'num_orders_original': 'mean',
            'num_orders_enhanced': 'mean'
        }).reset_index()
        logging.info(center_analysis)
        
        logging.info("\nAnalysis by meal category:")
        category_analysis = merged.groupby('category').agg({
            'abs_diff': 'mean',
            'rel_diff': 'mean',
            'num_orders_original': 'mean',
            'num_orders_enhanced': 'mean'
        }).reset_index()
        logging.info(category_analysis)
        
        logging.info("\nAnalysis by cuisine:")
        cuisine_analysis = merged.groupby('cuisine').agg({
            'abs_diff': 'mean',
            'rel_diff': 'mean',
            'num_orders_original': 'mean',
            'num_orders_enhanced': 'mean'
        }).reset_index()
        logging.info(cuisine_analysis)
        
        # Analyze by week
        logging.info("\nAnalysis by week:")
        week_analysis = merged.groupby('week').agg({
            'abs_diff': 'mean',
            'rel_diff': 'mean',
            'num_orders_original': 'mean',
            'num_orders_enhanced': 'mean'
        }).reset_index()
        logging.info(week_analysis)
        
        # Plot week trends
        plt.figure(figsize=(12, 6))
        plt.plot(week_analysis['week'], week_analysis['num_orders_original'], 'b-', label='Original Model')
        plt.plot(week_analysis['week'], week_analysis['num_orders_enhanced'], 'g-', label='Enhanced Model')
        plt.fill_between(
            week_analysis['week'], 
            week_analysis['num_orders_original'] - week_analysis['abs_diff'],
            week_analysis['num_orders_original'] + week_analysis['abs_diff'],
            alpha=0.2, color='r'
        )
        plt.legend()
        plt.title('Prediction Trends by Week')
        plt.xlabel('Week')
        plt.ylabel('Average Number of Orders')
        plt.grid(True, alpha=0.3)
        plt.savefig('weekly_prediction_trends.png')
        
        # Identify largest differences
        logging.info("\nTop 10 largest differences:")
        largest_diff = merged.sort_values('abs_diff', ascending=False).head(10)
        logging.info(largest_diff[['id', 'center_id', 'meal_id', 'category', 'cuisine', 'center_type', 
                                  'num_orders_original', 'num_orders_enhanced', 'abs_diff', 'rel_diff']])
        
        # Create a correlation heatmap based on different variables
        plt.figure(figsize=(12, 10))
        numeric_cols = merged.select_dtypes(include=['float64', 'int64']).columns
        correlation = merged[numeric_cols].corr()
        
        # Plot only the interesting correlations
        mask = np.triu(np.ones_like(correlation))
        sns.heatmap(
            correlation, 
            annot=False, 
            mask=mask, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            linewidths=0.5
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('differences_correlation_heatmap.png')
        
        # Create comparison report
        report = pd.DataFrame({
            'Metric': [
                'Overall Mean Absolute Difference',
                'Overall Mean Relative Difference',
                'Correlation between Predictions',
                'Original Model Average Prediction',
                'Enhanced Model Average Prediction',
                'Original Model Prediction Std',
                'Enhanced Model Prediction Std'
            ],
            'Value': [
                f"{merged['abs_diff'].mean():.2f} orders",
                f"{merged['rel_diff'].mean():.2f}%",
                f"{merged['num_orders_original'].corr(merged['num_orders_enhanced']):.4f}",
                f"{merged['num_orders_original'].mean():.2f}",
                f"{merged['num_orders_enhanced'].mean():.2f}",
                f"{merged['num_orders_original'].std():.2f}",
                f"{merged['num_orders_enhanced'].std():.2f}"
            ]
        })
        
        report.to_csv('model_comparison_report.csv', index=False)
        logging.info("\nComparison report saved to model_comparison_report.csv")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    analyze_prediction_differences()

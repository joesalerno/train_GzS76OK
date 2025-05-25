import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_weekly_trends():
    """Analyze weekly trends in predictions."""
    try:
        logging.info("Analyzing weekly prediction trends...")
        
        # Check for submission files
        original_submission = "coolstudy_submission_optuna.csv"
        enhanced_submission = "enhanced_model_submission.csv"
        
        if not os.path.exists(enhanced_submission) or not os.path.exists(original_submission):
            logging.warning(f"One or more submission files not found.")
            return
        
        # Load test data to get weeks
        test_df = pd.read_csv("test.csv")
        
        # Load predictions
        enhanced_preds = pd.read_csv(enhanced_submission)
        original_preds = pd.read_csv(original_submission)
        
        # Merge predictions with test data
        enhanced_with_week = enhanced_preds.merge(test_df[['id', 'week']], on='id')
        original_with_week = original_preds.merge(test_df[['id', 'week']], on='id')
        
        # Calculate weekly metrics
        weekly_enhanced = enhanced_with_week.groupby('week')['num_orders'].agg(['mean', 'std', 'median', 'count']).reset_index()
        weekly_enhanced.columns = ['week', 'mean_enhanced', 'std_enhanced', 'median_enhanced', 'count']
        
        weekly_original = original_with_week.groupby('week')['num_orders'].agg(['mean', 'std', 'median']).reset_index()
        weekly_original.columns = ['week', 'mean_original', 'std_original', 'median_original']
        
        # Merge weekly metrics
        weekly_comparison = weekly_enhanced.merge(weekly_original, on='week')
        
        # Calculate differences
        weekly_comparison['mean_diff'] = weekly_comparison['mean_enhanced'] - weekly_comparison['mean_original']
        weekly_comparison['mean_rel_diff'] = (weekly_comparison['mean_diff'] / weekly_comparison['mean_original']) * 100
        weekly_comparison['std_diff'] = weekly_comparison['std_enhanced'] - weekly_comparison['std_original']
        
        # Plot weekly trends
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Mean predictions by week
        plt.subplot(2, 2, 1)
        plt.plot(weekly_comparison['week'], weekly_comparison['mean_original'], 'b-o', label='Original Model')
        plt.plot(weekly_comparison['week'], weekly_comparison['mean_enhanced'], 'g-o', label='Enhanced Model')
        plt.title('Average Orders by Week')
        plt.xlabel('Week')
        plt.ylabel('Average Number of Orders')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation by week
        plt.subplot(2, 2, 2)
        plt.plot(weekly_comparison['week'], weekly_comparison['std_original'], 'b-o', label='Original Model')
        plt.plot(weekly_comparison['week'], weekly_comparison['std_enhanced'], 'g-o', label='Enhanced Model')
        plt.title('Standard Deviation of Orders by Week')
        plt.xlabel('Week')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Mean difference
        plt.subplot(2, 2, 3)
        bars = plt.bar(weekly_comparison['week'], weekly_comparison['mean_diff'])
        for i, bar in enumerate(bars):
            if weekly_comparison['mean_diff'].iloc[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Difference in Average Predictions (Enhanced - Original)')
        plt.xlabel('Week')
        plt.ylabel('Difference in Average Orders')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Mean relative difference
        plt.subplot(2, 2, 4)
        bars = plt.bar(weekly_comparison['week'], weekly_comparison['mean_rel_diff'])
        for i, bar in enumerate(bars):
            if weekly_comparison['mean_rel_diff'].iloc[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Relative Difference in Predictions (%)')
        plt.xlabel('Week')
        plt.ylabel('Relative Difference (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weekly_prediction_analysis.png')
        logging.info("Weekly prediction analysis saved to weekly_prediction_analysis.png")
        
        # Print weekly comparison table
        logging.info("\nWeekly prediction comparison:")
        logging.info(weekly_comparison[['week', 'mean_original', 'mean_enhanced', 'mean_diff', 'mean_rel_diff',
                                       'std_original', 'std_enhanced']])
        
        # Save to CSV
        weekly_comparison.to_csv('weekly_prediction_comparison.csv', index=False)
        logging.info("Weekly comparison saved to weekly_prediction_comparison.csv")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    analyze_weekly_trends()

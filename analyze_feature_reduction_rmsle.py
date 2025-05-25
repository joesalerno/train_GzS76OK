import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    # Add 1 to avoid taking log of zero
    y_true = np.maximum(0, y_true) + 1
    y_pred = np.maximum(0, y_pred) + 1
    return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))

def analyze_feature_reduction_impact():
    """Analyze how feature reduction affects RMSLE performance."""
    logging.info("Analyzing impact of feature reduction on RMSLE...")
    
    # Load feature counts
    original_feature_count = 171  # From our previous analysis
    
    # Check if enhanced model has feature importance file
    enhanced_importance_file = "enhanced_model_feature_importance.csv"
    if os.path.exists(enhanced_importance_file):
        enhanced_features = pd.read_csv(enhanced_importance_file)
        enhanced_feature_count = len(enhanced_features)
    else:
        logging.warning(f"Enhanced model feature importance file not found: {enhanced_importance_file}")
        enhanced_feature_count = 60  # Approximation from our code
    
    # Load performance metrics
    metrics_file = "enhanced_model_performance_metrics.csv"
    if os.path.exists(metrics_file):
        metrics = pd.read_csv(metrics_file)
        enhanced_rmsle = metrics['rmsle'].values[0]
    else:
        logging.warning(f"Performance metrics file not found: {metrics_file}")
        enhanced_rmsle = None
    
    # Load comparison results if available
    comparison_file = "model_comparison_results.csv"
    if os.path.exists(comparison_file):
        comparison = pd.read_csv(comparison_file)
        
        original_model = comparison[comparison['model'].str.contains('Original')].iloc[0] if any(comparison['model'].str.contains('Original')) else None
        enhanced_model = comparison[comparison['model'].str.contains('Enhanced')].iloc[0] if any(comparison['model'].str.contains('Enhanced')) else None
        
        if original_model is not None and enhanced_model is not None:
            original_rmsle = original_model['rmsle']
            improved_rmsle = enhanced_model['rmsle']
            
            rmsle_improvement = (original_rmsle - improved_rmsle) / original_rmsle * 100
            
            # Create visualization comparing feature count vs performance
            plt.figure(figsize=(12, 6))
            
            # Feature count comparison
            ax1 = plt.subplot(1, 2, 1)
            models = ['Original Model', 'Enhanced Model']
            feature_counts = [original_feature_count, enhanced_feature_count]
            
            ax1.bar(models, feature_counts, color=['#3274A1', '#1A9641'])
            ax1.set_title('Feature Count Comparison', fontsize=14)
            ax1.set_ylabel('Number of Features', fontsize=12)
            
            # Add percentage reduction
            reduction_pct = (original_feature_count - enhanced_feature_count) / original_feature_count * 100
            ax1.text(0.5, 0.5, f"{reduction_pct:.1f}% Reduction",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add counts on bars
            for i, count in enumerate(feature_counts):
                ax1.text(i, count + 3, str(count), ha='center', fontweight='bold')
            
            # RMSLE comparison
            ax2 = plt.subplot(1, 2, 2)
            rmsles = [original_rmsle, improved_rmsle]
            
            ax2.bar(models, rmsles, color=['#C51B7D', '#4D9221'])
            ax2.set_title('RMSLE Comparison', fontsize=14)
            ax2.set_ylabel('RMSLE (lower is better)', fontsize=12)
            
            # Add improvement percentage
            ax2.text(0.5, 0.5, f"{rmsle_improvement:.1f}% Improvement",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add values on bars
            for i, val in enumerate(rmsles):
                ax2.text(i, val + 0.01, f"{val:.4f}", ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig("feature_reduction_vs_rmsle.png", dpi=300)
            logging.info("Saved feature reduction vs RMSLE comparison plot to feature_reduction_vs_rmsle.png")
            
            # Summary
            logging.info(f"Feature reduction: {original_feature_count} → {enhanced_feature_count} ({reduction_pct:.1f}% reduction)")
            logging.info(f"RMSLE improvement: {original_rmsle:.4f} → {improved_rmsle:.4f} ({rmsle_improvement:.2f}% improvement)")
        else:
            logging.warning("Could not find both original and enhanced model data in comparison results.")
    else:
        logging.warning(f"Comparison results file not found: {comparison_file}")

if __name__ == "__main__":
    analyze_feature_reduction_impact()

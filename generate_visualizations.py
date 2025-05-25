import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def plot_feature_reduction_summary():
    """Create a visual summary of the feature reduction results."""
    # Original vs optimized feature counts
    original_count = 171
    optimized_count = 60
    
    # High correlation counts
    original_high_corr = 183
    optimized_high_corr = 6
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Feature count comparison
    labels = ['Original Model', 'Optimized Model']
    counts = [original_count, optimized_count]
    
    ax1.bar(labels, counts, color=['#3274A1', '#1A9641'])
    ax1.set_title('Feature Count Comparison', fontsize=14)
    ax1.set_ylabel('Number of Features', fontsize=12)
    
    # Add percentage reduction text
    reduction_pct = (original_count - optimized_count) / original_count * 100
    ax1.text(0.5, 0.5, f"{reduction_pct:.1f}% Reduction",
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    for i, count in enumerate(counts):
        ax1.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    # 2. High correlation count comparison
    labels = ['Original Model', 'Optimized Model']
    corr_counts = [original_high_corr, optimized_high_corr]
    
    ax2.bar(labels, corr_counts, color=['#C51B7D', '#4D9221'])
    ax2.set_title('High Correlation Pairs (|r| > 0.8)', fontsize=14)
    ax2.set_ylabel('Number of Highly Correlated Pairs', fontsize=12)
    
    # Add percentage reduction text
    corr_reduction_pct = (original_high_corr - optimized_high_corr) / original_high_corr * 100
    ax2.text(0.5, 0.5, f"{corr_reduction_pct:.1f}% Reduction",
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    for i, count in enumerate(corr_counts):
        ax2.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_reduction_summary.png', dpi=300)
    plt.close()
    
    print("Created feature reduction summary visualization")

def plot_pca_variance_explained():
    """Create a plot showing cumulative variance explained by PCA components."""
    # Load SHAP values to perform PCA
    shap_values_file = "shap_coolstudy_optuna_values.csv"
    
    if not os.path.exists(shap_values_file):
        print(f"Error: {shap_values_file} not found. Cannot create PCA variance plot.")
        return
    
    # Load the data
    shap_values = pd.read_csv(shap_values_file)
    
    # Get top 50 features for the analysis
    shap_importances_file = "shap_coolstudy_optuna_feature_importances.csv"
    if os.path.exists(shap_importances_file):
        shap_importances = pd.read_csv(shap_importances_file)
        top_features = shap_importances.sort_values('mean_abs_shap', ascending=False)['feature'].tolist()[:50]
        X = shap_values[top_features]
    else:
        X = shap_values
        
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    n_components = min(20, X.shape[1])  # Up to 20 components or max available
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-', color='#3274A1', linewidth=2)
    ax.set_title('Cumulative Explained Variance by Principal Components', fontsize=14)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Variance Threshold')
    
    # Add threshold crossing point
    components_for_90 = np.argmax(cumulative_variance >= 0.9) + 1
    ax.plot(components_for_90, cumulative_variance[components_for_90-1], 'ro')
    ax.text(components_for_90+0.5, cumulative_variance[components_for_90-1], 
            f'{components_for_90} components\nexplain 90% variance', 
            verticalalignment='center')
    
    # Set y-axis limit
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add percentage explained by component on the plot
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):  # Show first 5
        ax.text(i+1.1, cumulative_variance[i]-0.05, f"{var*100:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('pca_variance_explained.png', dpi=300)
    plt.close()
    
    print("Created PCA variance explained visualization")

def plot_top_feature_comparison():
    """Plot a comparison of SHAP values for the top 10 features."""
    
    # Load feature importance data
    orig_file = "shap_coolstudy_optuna_feature_importances.csv"
    optimized_file = "feature_optimized_shap_importances.csv"
    
    if not os.path.exists(orig_file):
        print(f"Error: {orig_file} not found. Cannot create top feature comparison.")
        return
    
    # Load original importances
    orig_importances = pd.read_csv(orig_file)
    orig_top10 = orig_importances.sort_values('mean_abs_shap', ascending=False).head(10)
    
    # Check if optimized model results exist
    if os.path.exists(optimized_file):
        opt_importances = pd.read_csv(optimized_file)
        opt_top10 = opt_importances.sort_values('mean_abs_shap', ascending=False).head(10)
        
        # Create combined visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for combined plot
        features = list(set(orig_top10['feature']).union(set(opt_top10['feature'])))[:15]  # Top 15 max
        
        # Get values for each model
        orig_values = []
        opt_values = []
        
        for feature in features:
            # Original model
            orig_val = orig_importances[orig_importances['feature'] == feature]['mean_abs_shap'].values
            orig_values.append(float(orig_val[0]) if len(orig_val) > 0 else 0)
            
            # Optimized model
            opt_val = opt_importances[opt_importances['feature'] == feature]['mean_abs_shap'].values
            opt_values.append(float(opt_val[0]) if len(opt_val) > 0 else 0)
        
        # Sort features by max importance in either model
        combined = pd.DataFrame({
            'feature': features,
            'original': orig_values,
            'optimized': opt_values,
            'max_val': [max(a, b) for a, b in zip(orig_values, opt_values)]
        })
        combined = combined.sort_values('max_val', ascending=False).head(10)
        
        # Create grouped bar chart
        x = np.arange(len(combined))
        width = 0.35
        
        ax.bar(x - width/2, combined['original'], width, label='Original Model', color='#4575B4')
        ax.bar(x + width/2, combined['optimized'], width, label='Optimized Model', color='#D73027')
        
        ax.set_title('Feature Importance Comparison: Original vs Optimized Model', fontsize=14)
        ax.set_ylabel('SHAP Importance Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(combined['feature'], rotation=45, ha='right')
        
        # Add a legend
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('top_feature_comparison.png', dpi=300)
        plt.close()
        
        print("Created top feature comparison visualization")
    else:
        # Create visualization for just original model
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Top 10 features
        features = orig_top10['feature'].tolist()
        values = orig_top10['mean_abs_shap'].tolist()
        
        # Create bar chart
        ax.barh(features[::-1], values[::-1], color='#4575B4')
        
        ax.set_title('Top 10 Features by SHAP Importance', fontsize=14)
        ax.set_xlabel('SHAP Importance Value', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('top_feature_importance.png', dpi=300)
        plt.close()
        
        print("Created top feature importance visualization")

def plot_correlation_matrix_difference():
    """Create a visual comparison of feature correlations before and after optimization."""
    # Load SHAP values to compute correlations
    orig_file = "shap_coolstudy_optuna_values.csv"
    
    if not os.path.exists(orig_file):
        print(f"Error: {orig_file} not found. Cannot create correlation comparison.")
        return
    
    # Load original values
    orig_values = pd.read_csv(orig_file)
    
    # Get top 20 features for a cleaner visualization
    top_features_file = "shap_coolstudy_optuna_feature_importances.csv"
    if os.path.exists(top_features_file):
        top_importances = pd.read_csv(top_features_file)
        top_features = top_importances.sort_values('mean_abs_shap', ascending=False)['feature'].head(20).tolist()
        
        # Calculate correlation matrix for top features
        corr_matrix = orig_values[top_features].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix))
        sns.heatmap(corr_matrix, annot=False, mask=mask, cmap='coolwarm', 
                    vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Feature Correlation Matrix for Top 20 Features', fontsize=16)
        plt.tight_layout()
        plt.savefig('feature_correlation_matrix.png', dpi=300)
        plt.close()
        
        print("Created feature correlation matrix visualization")
        
        # Create a visualization showing the number of strong correlations
        corr_threshold = 0.8
        strong_corrs = np.sum(np.abs(corr_matrix) > corr_threshold)
        
        # Count unique high correlations (exclude self-correlations and double-counting)
        high_corr_count = (np.sum(np.abs(corr_matrix) > corr_threshold) - len(top_features)) / 2
        
        # Count correlations by range
        ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts = []
        
        for i in range(len(ranges)-1):
            lower = ranges[i]
            upper = ranges[i+1]
            # Count correlations in this range (absolute value)
            count = np.sum((np.abs(corr_matrix) > lower) & (np.abs(corr_matrix) <= upper))
            # Remove self-correlations for the last range
            if i == len(ranges)-2:
                count -= len(top_features)
            # Divide by 2 to avoid double counting
            counts.append(count / 2)
        
        # Plot distribution of correlations
        plt.figure(figsize=(10, 6))
        bars = plt.bar(['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], 
                counts, color=['#91CF60', '#91CF60', '#FFFFBF', '#FC8D59', '#D73027'])
        plt.title('Distribution of Feature Correlations (Absolute Values)', fontsize=14)
        plt.xlabel('Correlation Range', fontsize=12)
        plt.ylabel('Count of Feature Pairs', fontsize=12)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('correlation_distribution.png', dpi=300)
        plt.close()
        
        print("Created correlation distribution visualization")

if __name__ == "__main__":
    print("Generating visualizations for dimensionality reduction analysis...")
    
    # Create visualizations
    plot_feature_reduction_summary()
    plot_pca_variance_explained()
    plot_top_feature_comparison()
    plot_correlation_matrix_difference()
    
    print("All visualizations completed.")

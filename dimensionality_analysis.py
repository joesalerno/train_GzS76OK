import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import os

# Set plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load SHAP values and feature importance data
shap_feature_importances = pd.read_csv('shap_coolstudy_optuna_feature_importances.csv')
shap_values = pd.read_csv('shap_coolstudy_optuna_values.csv')

print(f"Features in SHAP values: {shap_values.shape[1]}")
print(f"Features in importance data: {shap_feature_importances.shape[0]}")

# Sort features by importance
shap_feature_importances = shap_feature_importances.sort_values('mean_abs_shap', ascending=False)
top_features = shap_feature_importances['feature'].tolist()

# Get top 50 features for our analysis
top_50_features = top_features[:50]
print("\nTop 10 most important features:")
print(shap_feature_importances.head(10))

# Calculate correlation matrix for top features
print("\nCalculating correlation matrix for top features...")
feature_correlation = shap_values[top_50_features].corr()

# Identify highly correlated feature pairs (correlation > 0.8)
corr_threshold = 0.8
high_corr_pairs = []

for i in range(len(top_50_features)):
    for j in range(i + 1, len(top_50_features)):
        if abs(feature_correlation.iloc[i, j]) >= corr_threshold:
            feature1 = top_50_features[i]
            feature2 = top_50_features[j]
            corr_value = feature_correlation.iloc[i, j]
            imp1 = shap_feature_importances[shap_feature_importances['feature'] == feature1]['mean_abs_shap'].values[0]
            imp2 = shap_feature_importances[shap_feature_importances['feature'] == feature2]['mean_abs_shap'].values[0]
            high_corr_pairs.append((feature1, feature2, corr_value, imp1, imp2))

# Sort by absolute correlation value
high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nHighly correlated feature pairs (|correlation| > 0.8):")
print("Feature 1 | Feature 2 | Correlation | SHAP Importance 1 | SHAP Importance 2")
for f1, f2, corr, imp1, imp2 in high_corr_pairs[:20]:  # Show top 20 correlations
    print(f"{f1} | {f2} | {corr:.3f} | {imp1:.3f} | {imp2:.3f}")

# Save correlation matrix heatmap
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(feature_correlation))
sns.heatmap(feature_correlation, annot=False, mask=mask, cmap='coolwarm', 
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Matrix for Top 50 Features', fontsize=16)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png', dpi=300)

# Perform PCA on the top features
print("\nPerforming PCA analysis...")
scaler = StandardScaler()
X_top = shap_values[top_50_features]
X_scaled = scaler.fit_transform(X_top)

# Choose number of components to explain 90% of variance
pca = PCA(n_components=0.9)
pca_result = pca.fit_transform(X_scaled)

print(f"Number of PCA components to explain 90% variance: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

# Get feature loadings
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=top_50_features
)

# Identify top contributing features for each principal component
n_top_features = 5
top_loadings = {}

for i in range(pca.n_components_):
    pc = f'PC{i+1}'
    # Get the top positive and negative loadings
    pc_loadings = loadings[pc].abs().sort_values(ascending=False)
    top_loadings[pc] = pc_loadings.head(n_top_features).index.tolist()
    
    print(f"\nTop {n_top_features} features in {pc} (explains {pca.explained_variance_ratio_[i]:.4f} variance):")
    for feat in top_loadings[pc]:
        loading_value = loadings.loc[feat, pc]
        print(f"  - {feat}: {loading_value:.4f}")

# Perform ICA for comparison
print("\nPerforming ICA analysis...")
ica = FastICA(n_components=10, random_state=42)
ica_result = ica.fit_transform(X_scaled)

# Get ICA components mixing matrix
ica_components = pd.DataFrame(
    ica.components_, 
    columns=top_50_features,
    index=[f'IC{i+1}' for i in range(10)]
)

# Find the most influential features in each independent component
n_top_features = 5
for i in range(10):
    ic = f'IC{i+1}'
    # Get the absolute values to find the most influential features regardless of direction
    ic_influences = ica_components.loc[ic].abs().sort_values(ascending=False)
    top_features_ic = ic_influences.head(n_top_features).index.tolist()
    
    print(f"\nTop {n_top_features} features in {ic}:")
    for feat in top_features_ic:
        influence_value = ica_components.loc[ic, feat]
        print(f"  - {feat}: {influence_value:.4f}")

# Generate recommendations for feature engineering
print("\nRECOMMENDATIONS FOR MODEL IMPROVEMENT:")
print("1. Feature Reduction Recommendations:")
# Group similar features together
feature_groups = {}
for f1, f2, corr, imp1, imp2 in high_corr_pairs:
    if f1 not in feature_groups and f2 not in feature_groups:
        feature_groups[f1] = [f1, f2]
    elif f1 in feature_groups:
        if f2 not in feature_groups[f1]:
            feature_groups[f1].append(f2)
    elif f2 in feature_groups:
        if f1 not in feature_groups[f2]:
            feature_groups[f2].append(f1)

# Print feature groups and recommend which to keep
print("   Highly correlated feature groups:")
for key, group in feature_groups.items():
    if len(group) > 1:  # Only print groups with more than one feature
        print(f"   Group related to {key}:")
        for feat in group:
            imp = shap_feature_importances[shap_feature_importances['feature'] == feat]['mean_abs_shap'].values[0]
            print(f"     - {feat} (SHAP: {imp:.3f})")
        
        # Find the feature with highest importance in the group
        importances = [shap_feature_importances[shap_feature_importances['feature'] == feat]['mean_abs_shap'].values[0] for feat in group]
        max_imp_idx = importances.index(max(importances))
        print(f"     → Consider keeping only: {group[max_imp_idx]} and removing others in this group")

# Recommend removing or keeping interaction features
interaction_features = [f for f in top_features if '_x_' in f]
beneficial_interactions = []
harmful_interactions = []

for feat in interaction_features:
    parts = feat.split('_x_')
    if len(parts) == 2:
        base1, base2 = parts[0], parts[1]
        # Try to find base features
        base1_matches = [f for f in top_features if f == base1 or f.startswith(base1 + '_')]
        base2_matches = [f for f in top_features if f == base2 or f.startswith(base2 + '_')]
        
        imp_interaction = shap_feature_importances[shap_feature_importances['feature'] == feat]['mean_abs_shap'].values[0] if feat in shap_feature_importances['feature'].values else 0
        
        # Check if base features exist and have importance
        base1_imp = 0
        if base1_matches:
            base1_exact = [f for f in base1_matches if f == base1]
            if base1_exact:
                base1_imp = shap_feature_importances[shap_feature_importances['feature'] == base1]['mean_abs_shap'].values[0] if base1 in shap_feature_importances['feature'].values else 0
        
        base2_imp = 0
        if base2_matches:
            base2_exact = [f for f in base2_matches if f == base2]
            if base2_exact:
                base2_imp = shap_feature_importances[shap_feature_importances['feature'] == base2]['mean_abs_shap'].values[0] if base2 in shap_feature_importances['feature'].values else 0
        
        # Simple heuristic for whether interaction is beneficial
        if imp_interaction > (base1_imp + base2_imp) * 0.5:
            beneficial_interactions.append((feat, imp_interaction, base1, base1_imp, base2, base2_imp))
        else:
            harmful_interactions.append((feat, imp_interaction, base1, base1_imp, base2, base2_imp))

print("\n2. Interaction Feature Recommendations:")
print("   Beneficial interaction features to keep:")
for feat, imp, base1, imp1, base2, imp2 in sorted(beneficial_interactions, key=lambda x: x[1], reverse=True)[:10]:
    print(f"   - {feat} (SHAP: {imp:.3f}) [Base features: {base1}({imp1:.3f}), {base2}({imp2:.3f})]")

print("\n   Interaction features to consider removing:")
for feat, imp, base1, imp1, base2, imp2 in sorted(harmful_interactions, key=lambda x: x[1])[:10]:
    print(f"   - {feat} (SHAP: {imp:.3f}) [Base features: {base1}({imp1:.3f}), {base2}({imp2:.3f})]")

print("\n3. PCA-Based Recommendations:")
print("   Consider creating composite features based on these key principal components:")
for i in range(min(3, pca.n_components_)):
    pc = f'PC{i+1}'
    print(f"   - {pc} (Variance explained: {pca.explained_variance_ratio_[i]:.4f}):")
    for feat in top_loadings[pc]:
        loading_value = loadings.loc[feat, pc]
        imp = shap_feature_importances[shap_feature_importances['feature'] == feat]['mean_abs_shap'].values[0]
        print(f"     * {feat} (Loading: {loading_value:.4f}, SHAP: {imp:.3f})")

# Save loadings DataFrame for further analysis
loadings.to_csv('pca_feature_loadings.csv')
print("\nPCA feature loadings saved to 'pca_feature_loadings.csv'")

# Final recommendations
print("\nFINAL IMPROVEMENT RECOMMENDATIONS:")
print("1. Remove highly correlated features with lower importance scores")
print("2. Keep only beneficial interaction features and remove harmful ones")
print("3. Consider creating new composite features based on principal component loadings")
print("4. Focus feature engineering efforts on the top 20 most important features")
print("5. Explore creating simpler, more robust interaction features using the most important base features")

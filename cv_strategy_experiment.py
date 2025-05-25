import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_squared_log_error
import warnings
import os

# Optional: iterative-stratification
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, IterativeStratification
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False
    print("iterative-stratification not installed. Skipping those CV types.")

warnings.filterwarnings('ignore')

# --- Load Data ---
df = pd.read_csv('train.csv')

# --- Feature Engineering (reuse your pipeline) ---
from recursive_hybrid_forecast import apply_feature_engineering, FEATURES, TARGET
train_df, weekofyear_means, month_means = apply_feature_engineering(df, is_train=True)

# --- Prepare features/target ---
X = train_df[FEATURES]
y = train_df[TARGET]

# --- CV Strategies ---
cv_strategies = {
    'KFold': KFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'TimeSeriesSplit': TimeSeriesSplit(n_splits=5),
    'GroupKFold(center_id)': GroupKFold(n_splits=5),
}
if HAS_ITERSTRAT:
    cv_strategies['MultilabelStratifiedKFold'] = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_strategies['IterativeStratification'] = IterativeStratification(n_splits=5, order=1)

# --- Helper for stratification/grouping ---
def get_stratify_labels():
    # Use category or cuisine if available, else bin target
    if 'category' in train_df.columns:
        return train_df['category']
    elif 'cuisine' in train_df.columns:
        return train_df['cuisine']
    else:
        return pd.qcut(y, q=5, labels=False, duplicates='drop')

def get_groups():
    if 'center_id' in train_df.columns:
        return train_df['center_id']
    else:
        return None

def get_multilabel():
    # Example: use one-hot of category and cuisine if available
    cols = []
    for c in ['category', 'cuisine']:
        if c in train_df.columns:
            cols.append(pd.get_dummies(train_df[c], prefix=c))
    if cols:
        return pd.concat(cols, axis=1).values
    else:
        return None

# --- Run CV Experiment ---
results = []
for name, cv in cv_strategies.items():
    print(f"\nRunning CV: {name}")
    rmsle_scores = []
    if name == 'StratifiedKFold':
        stratify_labels = get_stratify_labels()
        splits = cv.split(X, stratify_labels)
    elif name == 'GroupKFold(center_id)':
        groups = get_groups()
        splits = cv.split(X, y, groups)
    elif name == 'MultilabelStratifiedKFold' and HAS_ITERSTRAT:
        multilabel = get_multilabel()
        if multilabel is not None:
            splits = cv.split(X, multilabel)
        else:
            print("No multilabels available, skipping.")
            continue
    elif name == 'IterativeStratification' and HAS_ITERSTRAT:
        multilabel = get_multilabel()
        if multilabel is not None:
            splits = cv.split(X, multilabel)
        else:
            print("No multilabels available, skipping.")
            continue
    else:
        splits = cv.split(X, y)
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmsle = np.sqrt(mean_squared_log_error(np.maximum(0, y_val), np.maximum(0, y_pred)))
        rmsle_scores.append(rmsle)
        print(f"  Fold {fold+1}: RMSLE={rmsle:.4f}")
    results.append({'cv': name, 'mean_rmsle': np.mean(rmsle_scores), 'std_rmsle': np.std(rmsle_scores), 'folds': rmsle_scores})
    print(f"{name}: Mean RMSLE={np.mean(rmsle_scores):.4f}, Std={np.std(rmsle_scores):.4f}")

# --- Plot Results ---
plt.figure(figsize=(10,6))
for r in results:
    plt.plot(r['folds'], marker='o', label=f"{r['cv']} (mean={r['mean_rmsle']:.4f})")
plt.xlabel('Fold')
plt.ylabel('RMSLE')
plt.title('CV Strategy Comparison: Fold RMSLEs')
plt.legend()
plt.tight_layout()
plt.savefig('cv_strategy_comparison.png')
plt.show()

# Save results to CSV
pd.DataFrame(results).to_csv('cv_strategy_comparison_results.csv', index=False)
print("\nAll results saved to cv_strategy_comparison_results.csv and cv_strategy_comparison.png")

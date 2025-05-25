import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import warnings

warnings.filterwarnings('ignore')

# --- Load Data ---
df = pd.read_csv('train.csv')

# --- Target ---
TARGET = 'num_orders'


# --- Feature List ---
continuous_features = set()
# Always include these
for col in ['checkout_price', 'base_price', 'price_diff', 'discount_pct']:
    if col in df.columns:
        continuous_features.add(col)
# Add all rolling means/stds
for col in df.columns:
    if col.startswith('num_orders_rolling_mean_') or col.startswith('num_orders_rolling_std_'):
        continuous_features.add(col)
# Add group aggregates
for col in df.columns:
    if any(col.startswith(prefix) for prefix in ['center_orders_', 'meal_orders_', 'category_orders_']):
        continuous_features.add(col)
# Add any other continuous (float) columns, excluding the target
for col in df.select_dtypes(include=[np.number]).columns:
    if col != TARGET:
        continuous_features.add(col)
continuous_features = list(continuous_features)


# --- Transforms to test ---
def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

def safe_log(x):
    return np.log1p(np.maximum(x, 0))

def identity(x):
    return x

def safe_square(x):
    return x ** 2

def safe_cube(x):
    return x ** 3

def rank_transform(x):
    return pd.Series(x).rank(method='average').values

def quantile_transform(x):
    qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
    return qt.fit_transform(x.reshape(-1, 1)).flatten()

def yeo_johnson_transform(x):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    return pt.fit_transform(x.reshape(-1, 1)).flatten()

TRANSFORMS = {
    'none': identity,
    'square': safe_square,
    'cube': safe_cube,
    'log': safe_log,
    'symlog': symlog,
    'rank': rank_transform,
    'quantile': quantile_transform,
    'yeo_johnson': yeo_johnson_transform,
}

results = []

for feat in continuous_features:
    if feat not in df.columns:
        continue
    X = df[[feat]].copy()
    y = df[TARGET]
    # Remove rows with NaN in feature or target
    mask = (~X[feat].isna()) & (~y.isna())
    X = X[mask]
    y = y[mask]
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    best_rmsle = float('inf')
    best_transform = None
    for tname, tfunc in TRANSFORMS.items():
        try:
            X_train_t = tfunc(X_train[feat].values)
            X_val_t = tfunc(X_val[feat].values)
            # If all values are nan or inf, skip
            if np.any(np.isnan(X_train_t)) or np.any(np.isnan(X_val_t)) or np.any(np.isinf(X_train_t)) or np.any(np.isinf(X_val_t)):
                continue
            model = LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_t.reshape(-1, 1), y_train)
            y_pred = model.predict(X_val_t.reshape(-1, 1))
            # RMSLE
            rmsle = np.sqrt(mean_squared_log_error(np.maximum(0, y_val), np.maximum(0, y_pred)))
            results.append({'feature': feat, 'transform': tname, 'rmsle': rmsle})
            if rmsle < best_rmsle:
                best_rmsle = rmsle
                best_transform = tname
        except Exception as e:
            continue
    print(f'Feature: {feat:30s}  Best Transform: {best_transform:8s}  RMSLE: {best_rmsle:.4f}')

# Save all results
df_results = pd.DataFrame(results)
df_results.to_csv('empirical_transform_results.csv', index=False)
print('\nAll results saved to empirical_transform_results.csv')

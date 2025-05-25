import os
import pandas as pd
import numpy as np
import optuna
import shap
import logging
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Configuration ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [2, 3, 5, 10, 14, 21]
VALIDATION_WEEKS = 8
OPTUNA_TRIALS = 50
SHAP_SAMPLES = 2000
ENSEMBLE_WINDOW_SIZE = 12

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load and Preprocess Data ---
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    meal_info = pd.read_csv(MEAL_INFO_PATH)
    center_info = pd.read_csv(CENTER_INFO_PATH)
    
    def preprocess(df):
        df = df.merge(meal_info, on="meal_id", how="left")
        df = df.merge(center_info, on="center_id", how="left")
        df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
        return df
    
    df = preprocess(df)
    test = preprocess(test)
    test['num_orders'] = np.nan  # Placeholder for prediction
    return df, test

# --- Feature Engineering ---
def create_features(df):
    group_cols = ["center_id", "meal_id"]
    df = df.copy()
    
    # Lag features
    for lag in LAG_WEEKS:
        df[f"num_orders_lag_{lag}"] = df.groupby(group_cols)["num_orders"].shift(lag)
    
    # Rolling features
    shifted = df.groupby(group_cols)["num_orders"].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"num_orders_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
        df[f"num_orders_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std()
    
    # Price features
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = df["discount"] / df["base_price"].replace(0, 1e-308)
    
    # Temporal features
    df["weekofyear"] = df["week"] % 52
    df["month"] = ((df["week"] - 1) // 4) % 12 + 1
    
    return df

# --- Advanced Ensemble Model ---
class AdvancedEnsemble:
    def __init__(self, features, target, config):
        self.features = features
        self.target = target
        self.config = config
        self.base_models = [
            LGBMRegressor(**config['lgb_params']),
            CatBoostRegressor(**config['cat_params']),
            Ridge(alpha=1.0)
        ]
        self.meta_model = None
        self.residual_model = None
        self.scaler = StandardScaler()
        
    def _create_dynamic_weights(self, X):
        # Simplified version - in practice use SHAP values
        weights = np.ones((X.shape[0], len(self.base_models)))
        return weights
    
    def _build_residual_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                input_shape=(self.config['window_size'], len(self.features))),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X_train, y_train, X_val, y_val):
        # Train base models
        for model in self.base_models:
            model.fit(X_train, y_train)
        
        # Generate base predictions
        base_preds = np.array([model.predict(X_train) for model in self.base_models]).T
        val_base_preds = np.array([model.predict(X_val) for model in self.base_models]).T
        
        # Create meta-features
        meta_features = np.concatenate([
            base_preds,
            self._create_dynamic_weights(X_train),
            np.abs(base_preds - y_train.values.reshape(-1, 1)),
            X_train[['checkout_price', 'discount', 'weekofyear']].values
        ], axis=1)
        
        val_meta_features = np.concatenate([
            val_base_preds,
            self._create_dynamic_weights(X_val),
            np.abs(val_base_preds - y_val.values.reshape(-1, 1)),
            X_val[['checkout_price', 'discount', 'weekofyear']].values
        ], axis=1)
        
        # Train meta-model
        self.meta_model = Ridge(alpha=self.config['meta_params']['alpha'])
        self.meta_model.fit(meta_features, y_train)
        
        # Train residual correction model
        residuals = y_train.values - self.meta_model.predict(meta_features)
        scaled_features = self.scaler.fit_transform(X_train[self.features])
        
        X_seq, y_seq = [], []
        for i in range(len(residuals) - self.config['window_size']):
            X_seq.append(scaled_features[i:i+self.config['window_size']])
            y_seq.append(residuals[i+self.config['window_size']])
            
        self.residual_model = self._build_residual_model()
        self.residual_model.fit(np.array(X_seq), np.array(y_seq), 
                               epochs=30, batch_size=32, verbose=0)
        
    def predict(self, X, history=None):
        base_preds = np.array([model.predict(X) for model in self.base_models]).T
        
        meta_features = np.concatenate([
            base_preds,
            self._create_dynamic_weights(X),
            np.zeros_like(base_preds),
            X[['checkout_price', 'discount', 'weekofyear']].values
        ], axis=1)
        
        meta_pred = self.meta_model.predict(meta_features)
        
        if history is not None and len(history) >= self.config['window_size']:
            scaled_history = self.scaler.transform(history[-self.config['window_size']:][self.features])
            correction = self.residual_model.predict(scaled_history.reshape(1, self.config['window_size'], -1))[0][0]
            return meta_pred + correction
        
        return meta_pred

# --- Main Execution ---
def main():
    # Load and preprocess data
    df, test_df = load_and_preprocess()
    df = create_features(df)
    test_df = create_features(test_df)
    
    # Split data
    max_week = df["week"].max()
    valid_df = df[df["week"] > max_week - VALIDATION_WEEKS]
    train_df = df[df["week"] <= max_week - VALIDATION_WEEKS]
    
    FEATURES = [col for col in df.columns if col != "num_orders" and col != "id"]
    TARGET = "num_orders"
    
    # Configure and train ensemble
    ensemble_config = {
        'lgb_params': {
            'n_estimators': 3000,
            'learning_rate': 0.015,
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 0.1
        },
        'cat_params': {
            'iterations': 3000,
            'learning_rate': 0.02,
            'depth': 8,
            'l2_leaf_reg': 10,
            'bootstrap_type': 'Bayesian',
            'subsample': 0.8
        },
        'window_size': ENSEMBLE_WINDOW_SIZE,
        'meta_params': {
            'alpha': 0.1
        }
    }
    
    ensemble = AdvancedEnsemble(FEATURES, TARGET, ensemble_config)
    ensemble.fit(train_df[FEATURES], train_df[TARGET], valid_df[FEATURES], valid_df[TARGET])
    
    # Generate predictions
    history = df.copy()
    test_df['num_orders'] = 0  # Initialize
    
    for week in sorted(test_df['week'].unique()):
        current = test_df[test_df['week'] == week]
        pred = ensemble.predict(current[FEATURES], history[TARGET].values[-ENSEMBLE_WINDOW_SIZE:])
        test_df.loc[test_df['week'] == week, 'num_orders'] = np.clip(pred, 0, None).round()
        history = pd.concat([history, current.assign(num_orders=pred)])
    
    # Create submission
    submission = test_df[['id', 'num_orders']]
    submission.to_csv("advanced_ensemble_submission.csv", index=False)
    logging.info("Submission file created")

if __name__ == "__main__":
    main()
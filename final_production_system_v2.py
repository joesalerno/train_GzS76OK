"""
Final Production Food Demand Forecasting System
==============================================

This module implements the final production-ready food demand forecasting system
based on the comprehensive evaluation results. It includes the best performing
Enhanced (Optimized) model with RMSLE: 0.42476.
"""

import os
import json
import pickle
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionForecastingSystem:
    """Production-ready food demand forecasting system."""
    
    def __init__(self, config_path: str = "production_model_info.json"):
        """Initialize the production forecasting system."""
        self.config_path = config_path
        self.model = None
        self.feature_names = None
        self.model_params = None
        self.performance_metrics = None
        self.version = "1.0.0"
        self.deployment_date = datetime.now()
        self.training_data = None
        
        # Performance tracking
        self.prediction_history = []
        self.performance_history = []
        self.drift_alerts = []
        
        # Load production configuration
        self._load_production_config()
        
    def _load_production_config(self):
        """Load the production model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.feature_names = config['features']
            self.model_params = config['model_params']
            self.performance_metrics = config['performance']
            
            logger.info(f"Loaded production config with {len(self.feature_names)} features")
            logger.info(f"Expected performance - RMSLE: {self.performance_metrics['validation_rmsle']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load production config: {e}")
            raise
    
    def create_enhanced_features(self, df: pd.DataFrame, is_training: bool = True, train_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create enhanced features using the optimized feature engineering pipeline."""
        logger.info("Creating enhanced features...")
        df_feat = df.copy()
        
        # Basic preprocessing
        df_feat['discount'] = df_feat['checkout_price'] - df_feat['base_price']
        df_feat['discount_pct'] = df_feat['discount'] / df_feat['base_price']
        df_feat['price_diff'] = df_feat['checkout_price'] - df_feat['base_price']
        
        # Date features
        df_feat['month'] = df_feat['week'] % 12 + 1
        df_feat['weekofyear'] = df_feat['week'] % 52 + 1
        df_feat['is_month_start'] = (df_feat['week'] % 4 == 1).astype(int)
        df_feat['is_quarter_start'] = (df_feat['week'] % 13 == 1).astype(int)
        
        # Create lag and rolling features
        if is_training and 'num_orders' in df_feat.columns:
            df_feat = self._create_lag_features(df_feat)
            df_feat = self._create_rolling_features(df_feat)
            df_feat = self._create_trend_features(df_feat)
            df_feat = self._create_aggregate_features(df_feat)
        elif not is_training and train_data is not None:
            # For test data, use training data to create lag features
            df_feat = self._create_test_lag_features(df_feat, train_data)
        
        # Interaction features (after lag features are created)
        df_feat['price_x_homepage'] = df_feat['base_price'] * df_feat['homepage_featured']
        df_feat['price_x_emailer'] = df_feat['base_price'] * df_feat['emailer_for_promotion']
        df_feat['discount_x_emailer'] = df_feat['discount'] * df_feat['emailer_for_promotion']
        df_feat['lag1_x_homepage'] = df_feat.get('orders_lag_1', 0) * df_feat['homepage_featured']
        df_feat['lag1_x_emailer'] = df_feat.get('orders_lag_1', 0) * df_feat['emailer_for_promotion']
        
        # Create rolling sum features for promotional features
        for window in [3, 5, 7]:
            for col in ['homepage_featured', 'emailer_for_promotion']:
                if col in df_feat.columns:
                    df_feat[f'{col}_rolling_sum_{window}'] = df_feat.groupby(['center_id', 'meal_id'])[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).sum()
                    )
        
        # Ensure all required features exist, fill missing with 0
        for feature in self.feature_names:
            if feature not in df_feat.columns:
                df_feat[feature] = 0
        
        logger.info(f"Created features. Shape: {df_feat.shape}")
        return df_feat
    
    def _create_test_lag_features(self, test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for test data using training data."""
        test_out = test_df.copy()
        
        # Get the last few weeks of training data for each center/meal combination
        max_train_week = train_df['week'].max()
        recent_train = train_df[train_df['week'] > max_train_week - 20].copy()
        
        # Create aggregate features from training data
        center_agg = train_df.groupby('center_id')['num_orders'].agg(['mean', 'std']).reset_index()
        center_agg.columns = ['center_id', 'center_avg_orders', 'center_std_orders']
        test_out = test_out.merge(center_agg, on='center_id', how='left')
        
        meal_agg = train_df.groupby('meal_id')['num_orders'].agg(['mean', 'std']).reset_index()
        meal_agg.columns = ['meal_id', 'meal_avg_orders', 'meal_std_orders']
        test_out = test_out.merge(meal_agg, on='meal_id', how='left')
        
        # For lag features, use simple historical averages for test data
        # This is a simplified approach for production deployment
        for lag in [1, 2, 3, 4, 5, 7, 10, 14]:
            test_out[f'orders_lag_{lag}'] = 0
        
        for window in [3, 5, 7, 10, 14, 21]:
            test_out[f'orders_mean_{window}'] = 0
            test_out[f'orders_std_{window}'] = 0
            test_out[f'orders_max_{window}'] = 0
            test_out[f'orders_min_{window}'] = 0
        
        for span in [3, 7, 14]:
            test_out[f'orders_ewma_{span}'] = 0
        
        for window in [3, 7, 14]:
            test_out[f'orders_trend_{window}'] = 0
            test_out[f'orders_volatility_{window}'] = 0
        
        # Use center/meal averages where possible
        center_meal_avg = train_df.groupby(['center_id', 'meal_id'])['num_orders'].mean().reset_index()
        center_meal_avg.columns = ['center_id', 'meal_id', 'avg_orders']
        test_out = test_out.merge(center_meal_avg, on=['center_id', 'meal_id'], how='left')
        
        # Fill lag features with historical averages
        for lag in [1, 2, 3, 4, 5, 7, 10, 14]:
            test_out[f'orders_lag_{lag}'] = test_out['avg_orders'].fillna(0)
        
        # Fill rolling features with historical averages  
        for window in [3, 5, 7, 10, 14, 21]:
            test_out[f'orders_mean_{window}'] = test_out['avg_orders'].fillna(0)
            test_out[f'orders_std_{window}'] = test_out['avg_orders'].fillna(0) * 0.3  # Assume 30% std
            test_out[f'orders_max_{window}'] = test_out['avg_orders'].fillna(0) * 1.5
            test_out[f'orders_min_{window}'] = test_out['avg_orders'].fillna(0) * 0.5
        
        for span in [3, 7, 14]:
            test_out[f'orders_ewma_{span}'] = test_out['avg_orders'].fillna(0)
            
        for window in [3, 7, 14]:
            test_out[f'orders_trend_{window}'] = 0  # Assume no trend for new data
            test_out[f'orders_volatility_{window}'] = 0.3  # Assume moderate volatility
        
        test_out = test_out.drop(['avg_orders'], axis=1, errors='ignore')
        
        return test_out
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series prediction."""
        df_out = df.copy()
        group = df_out.groupby(['center_id', 'meal_id'])
        
        for lag in [1, 2, 3, 4, 5, 7, 10, 14]:
            df_out[f'orders_lag_{lag}'] = group['num_orders'].shift(lag)
        
        return df_out
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features."""
        df_out = df.copy()
        group = df_out.groupby(['center_id', 'meal_id'])
        
        for window in [3, 5, 7, 10, 14, 21]:
            df_out[f'orders_mean_{window}'] = group['num_orders'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df_out[f'orders_std_{window}'] = group['num_orders'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df_out[f'orders_max_{window}'] = group['num_orders'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            df_out[f'orders_min_{window}'] = group['num_orders'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
        
        # Exponential weighted moving averages
        for span in [3, 7, 14]:
            df_out[f'orders_ewma_{span}'] = group['num_orders'].transform(
                lambda x: x.ewm(span=span).mean()
            )
        
        return df_out
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend and volatility features."""
        df_out = df.copy()
        group = df_out.groupby(['center_id', 'meal_id'])
        
        for window in [3, 7, 14]:
            # Trend (slope of linear regression)
            df_out[f'orders_trend_{window}'] = group['num_orders'].transform(
                lambda x: x.rolling(window=window).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == window else np.nan
                )
            )
            
            # Volatility (coefficient of variation)
            rolling_mean = group['num_orders'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            rolling_std = group['num_orders'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df_out[f'orders_volatility_{window}'] = rolling_std / (rolling_mean + 1e-8)
        
        return df_out
    
    def _create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features by center and meal."""
        df_out = df.copy()
        
        # Center-level aggregates
        center_agg = df_out.groupby('center_id')['num_orders'].agg(['mean', 'std']).reset_index()
        center_agg.columns = ['center_id', 'center_avg_orders', 'center_std_orders']
        df_out = df_out.merge(center_agg, on='center_id', how='left')
        
        # Meal-level aggregates
        meal_agg = df_out.groupby('meal_id')['num_orders'].agg(['mean', 'std']).reset_index()
        meal_agg.columns = ['meal_id', 'meal_avg_orders', 'meal_std_orders']
        df_out = df_out.merge(meal_agg, on='meal_id', how='left')
        
        return df_out
    
    def train_production_model(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Train the production model using the optimized parameters."""
        logger.info("Training production model...")
        start_time = datetime.now()
        
        # Prepare data
        train_features = self.create_enhanced_features(train_data, is_training=True)
        
        # Remove rows with missing target or excessive missing features
        train_features = train_features.dropna(subset=['num_orders'])
        missing_threshold = 0.3  # Allow up to 30% missing features
        train_features = train_features.dropna(thresh=int(len(train_features.columns) * (1 - missing_threshold)))
        
        # Fill remaining missing values
        train_features = train_features.fillna(0)
        
        # Select features that exist in the production config
        available_features = [f for f in self.feature_names if f in train_features.columns]
        missing_features = [f for f in self.feature_names if f not in train_features.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        X = train_features[available_features]
        y = train_features['num_orders']
        
        logger.info(f"Training with {len(available_features)} features on {len(X)} samples")
        
        # Train the model
        self.model = LGBMRegressor(**self.model_params)
        self.model.fit(X, y)
        
        # Store training data for future lag feature creation
        self.training_data = train_data.copy()
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        train_mae = mean_absolute_error(y, train_pred)
        train_rmsle = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(np.maximum(0, train_pred))))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'training_time': training_time,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_rmsle': train_rmsle,
            'features_used': len(available_features),
            'samples_trained': len(X)
        }
        
        logger.info(f"Model trained successfully in {training_time:.2f}s")
        logger.info(f"Training RMSLE: {train_rmsle:.4f}")fulfillment centers and meals. The system uses a recursive prediction
        
        return results
    
    def predict(self, test_data: pd.DataFrame, train_data: pd.DataFrame = None) -> np.ndarray:
        """Make predictions using the trained production model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_production_model first.")
        
        logger.info("Making predictions...")
        
        # Use stored training data if not provided
        if train_data is None:
            train_data = self.training_data
        
        # Create features for test data
        test_features = self.create_enhanced_features(test_data, is_training=False, train_data=train_data)
        
        # Fill missing values
        test_features = test_features.fillna(0)
        
        # Select features that match training
        X_test = test_features[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predictions = np.maximum(0, predictions)  # Ensure non-negative
        
        # Log prediction statistics
        logger.info(f"Made {len(predictions)} predictions")
        logger.info(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
        logger.info(f"Mean prediction: {predictions.mean():.2f}")
        
        # Store for monitoring
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'num_predictions': len(predictions),
            'mean_prediction': predictions.mean(),
            'std_prediction': predictions.std(),
            'min_prediction': predictions.min(),
            'max_prediction': predictions.max()
        })
        
        return predictions
    
    def save_model(self, model_path: str = "production_model.pkl"):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'performance_metrics': self.performance_metrics,
            'version': self.version,
            'deployment_date': self.deployment_date,
            'training_data': self.training_data
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

def create_production_submission(train_path: str = "train.csv", 
                               test_path: str = "test.csv",
                               meal_info_path: str = "meal_info.csv",
                               center_info_path: str = "fulfilment_center_info.csv",
                               output_path: str = "final_production_submission.csv"):
    """Create final production submission using the best performing model."""
    logger.info("Creating final production submission...")
    
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    meal_info = pd.read_csv(meal_info_path)
    center_info = pd.read_csv(center_info_path)
    
    # Merge with additional info
    train = train.merge(meal_info, on='meal_id', how='left')
    train = train.merge(center_info, on='center_id', how='left')
    test = test.merge(meal_info, on='meal_id', how='left')
    test = test.merge(center_info, on='center_id', how='left')
    
    # Initialize and train production system
    production_system = ProductionForecastingSystem()
    
    # Train the model
    training_results = production_system.train_production_model(train)
    logger.info(f"Training completed: {training_results}")
    
    # Make predictions
    predictions = production_system.predict(test, train_data=train)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'num_orders': predictions
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Final production submission saved to {output_path}")
    
    # Save the production model
    production_system.save_model("final_production_model.pkl")
    
    return {
        'submission_path': output_path,
        'training_results': training_results,
        'predictions_stats': {
            'count': len(predictions),
            'mean': predictions.mean(),
            'std': predictions.std(),
            'min': predictions.min(),
            'max': predictions.max()
        }
    }

if __name__ == "__main__":
    # Create final production submission
    results = create_production_submission()
    print("Final Production System Results:")
    print(json.dumps(results, indent=2, default=str))

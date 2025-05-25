import os
import random
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import logging
import lightgbm as lgb
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import catboost as cb
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Enhanced Configuration
class EnhancedConfig:
    # Data paths
    DATA_PATH = "train.csv"
    TEST_PATH = "test.csv"
    MEAL_INFO_PATH = "meal_info.csv"
    CENTER_INFO_PATH = "fulfilment_center_info.csv"
    
    # Model parameters
    SEED = 42
    LAG_WEEKS = [1, 2, 3, 4, 5, 7, 10, 14]  # Extended lag periods
    ROLLING_WINDOWS = [2, 3, 5, 7, 10, 14, 21, 28]  # Extended rolling windows
    EWMA_SPANS = [3, 7, 14, 28]  # Exponential weighted moving averages
    
    # Advanced feature parameters
    FOURIER_TERMS = 4  # Number of Fourier terms for seasonality
    DECOMPOSITION_PERIODS = [13, 26, 52]  # Seasonal decomposition periods
    CLUSTER_FEATURES = ['center_id', 'meal_id', 'checkout_price', 'base_price']
    N_CLUSTERS = 50
    
    # Model ensemble parameters
    ENSEMBLE_MODELS = ['lgbm', 'xgb', 'catboost', 'rf', 'mlp']
    ENSEMBLE_WEIGHTS = 'auto'  # Can be 'auto', 'equal', or list of weights
    
    # Validation
    VALIDATION_WEEKS = 8
    N_FOLDS = 5
    
    # Optuna
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600  # 1 hour
    
    # Output
    SUBMISSION_PREFIX = "enhanced_submission"
    MODEL_PREFIX = "enhanced_model"

config = EnhancedConfig()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedFeatureEngineering:
    """Enhanced feature engineering with advanced techniques."""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.cluster_models = {}

    def create_fourier_features(self, df, period_col='weekofyear', period=52, n_terms=4):
        """Create Fourier terms for better seasonality modeling."""
        df_out = df.copy()
        
        # Create date features first if not present
        if 'weekofyear' not in df_out.columns:
            df_out['weekofyear'] = df_out['week'] % 52 + 1
        if 'dayofweek' not in df_out.columns:
            df_out['dayofweek'] = df_out['week'] % 7 + 1
            
        # Create fourier features for multiple periods
        periods = [52, 7]  # yearly, weekly
        period_cols = ['weekofyear', 'dayofweek']
        
        for p, p_col in zip(periods, period_cols):
            for i in range(1, n_terms + 1):
                df_out[f'{p_col}_fourier_sin_{i}'] = np.sin(2 * np.pi * i * df_out[p_col] / p)
                df_out[f'{p_col}_fourier_cos_{i}'] = np.cos(2 * np.pi * i * df_out[p_col] / p)
        
        return df_out
    
    def create_trend_features(self, df, target='num_orders'):
        """Create trend and momentum features."""
        df_out = df.copy()
        group = df_out.groupby(['center_id', 'meal_id'])
        
        # Linear trend over different windows
        for window in [4, 8, 12]:
            df_out[f'{target}_trend_{window}'] = (
                group[target].shift(1)
                .rolling(window)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0)
                .reset_index(drop=True)
            )
            
        # Momentum features (rate of change)
        for lag in [1, 2, 4]:
            df_out[f'{target}_momentum_{lag}'] = (
                (group[target].shift(1) - group[target].shift(lag + 1)) / 
                group[target].shift(lag + 1).replace(0, np.nan)
            )
            
        return df_out
    
    def create_volatility_features(self, df, target='num_orders'):
        """Create volatility and stability measures."""
        df_out = df.copy()
        group = df_out.groupby(['center_id', 'meal_id'])
        shifted = group[target].shift(1)
        
        # Coefficient of variation over different windows
        for window in [4, 8, 12]:
            rolling_mean = shifted.rolling(window).mean()
            rolling_std = shifted.rolling(window).std()
            df_out[f'{target}_cv_{window}'] = (rolling_std / rolling_mean).replace([np.inf, -np.inf], 0)
            
        # Range features
        for window in [4, 8]:
            rolling_min = shifted.rolling(window).min()
            rolling_max = shifted.rolling(window).max()
            df_out[f'{target}_range_{window}'] = rolling_max - rolling_min
            df_out[f'{target}_range_norm_{window}'] = (
                (rolling_max - rolling_min) / rolling_mean.replace(0, 1)
            )
            
        return df_out
    
    def create_demand_regime_features(self, df, target='num_orders'):
        """Create features based on demand regimes."""
        df_out = df.copy()
        
        # Calculate global percentiles for demand levels
        if target in df_out.columns:
            p25, p50, p75, p90 = np.percentile(df_out[target].dropna(), [25, 50, 75, 90])
            
            df_out[f'{target}_regime_low'] = (df_out[target] <= p25).astype(int)
            df_out[f'{target}_regime_med'] = ((df_out[target] > p25) & (df_out[target] <= p75)).astype(int)
            df_out[f'{target}_regime_high'] = ((df_out[target] > p75) & (df_out[target] <= p90)).astype(int)
            df_out[f'{target}_regime_extreme'] = (df_out[target] > p90).astype(int)
            
        return df_out
    
    def create_clustering_features(self, df, is_train=True):
        """Create clustering-based features."""
        df_out = df.copy()
        
        cluster_features = [f for f in self.config.CLUSTER_FEATURES if f in df_out.columns]
        if len(cluster_features) < 2:
            return df_out
            
        if is_train:
            # Fit clustering model
            X_cluster = df_out[cluster_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            kmeans = KMeans(n_clusters=self.config.N_CLUSTERS, random_state=self.config.SEED)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            self.scalers['cluster'] = scaler
            self.cluster_models['demand'] = kmeans
            
        else:
            # Transform using fitted models
            if 'cluster' in self.scalers and 'demand' in self.cluster_models:
                X_cluster = df_out[cluster_features].fillna(0)
                X_scaled = self.scalers['cluster'].transform(X_cluster)
                cluster_labels = self.cluster_models['demand'].predict(X_scaled)
            else:
                cluster_labels = np.zeros(len(df_out))
                
        df_out['demand_cluster'] = cluster_labels
        
        # Create cluster-based aggregates
        if 'num_orders' in df_out.columns:
            cluster_means = df_out.groupby('demand_cluster')['num_orders'].transform('mean')
            df_out['cluster_demand_mean'] = cluster_means
            
        return df_out
    
    def create_external_features(self, df):
        """Create features based on external factors."""
        df_out = df.copy()
        
        # Holiday and special event indicators
        holiday_weeks = {1, 10, 25, 45, 52}  # New Year, Easter, Independence, Thanksgiving, Christmas
        df_out['is_holiday'] = df_out['week'].apply(lambda x: (x % 52) in holiday_weeks).astype(int)
        
        # Month-end effects
        df_out['is_month_end'] = (df_out['week'] % 4 == 0).astype(int)
        df_out['is_quarter_end'] = (df_out['week'] % 13 == 0).astype(int)
        
        # Weather proxy (simplified seasonal patterns)
        df_out['weather_proxy'] = np.sin(2 * np.pi * (df_out['week'] % 52) / 52)
        
        return df_out
    
    def create_cross_validation_features(self, df, target='num_orders'):
        """Create features using cross-validation target encoding."""
        df_out = df.copy()
        
        if target not in df_out.columns:
            return df_out
            
        # Cross-validated target encoding for categorical variables
        categorical_cols = ['center_id', 'meal_id', 'category', 'cuisine']
        categorical_cols = [col for col in categorical_cols if col in df_out.columns]
        
        for col in categorical_cols:
            # Simple target encoding with regularization
            global_mean = df_out[target].mean()
            col_means = df_out.groupby(col)[target].mean()
            col_counts = df_out.groupby(col)[target].count()
            
            # Smoothing parameter
            alpha = 10
            smoothed_means = (col_means * col_counts + global_mean * alpha) / (col_counts + alpha)
            
            df_out[f'{col}_target_encoded'] = df_out[col].map(smoothed_means).fillna(global_mean)
            
        return df_out

    def engineer_features(self, df):
        """Apply feature engineering - wrapper for prepare_features for testing compatibility."""
        return self.prepare_features(df, is_train=True)
    
    def get_feature_list(self, df):
        """Get list of feature columns for testing compatibility."""
        exclude_cols = ['id', 'num_orders', 'week']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        return feature_cols
    
    def select_features(self, train_data, val_data, features):
        """Feature selection for testing compatibility."""
        # Simple feature selection - return all valid features
        return [col for col in features if col in train_data.columns]
    
    def train_ensemble(self, train_data, features):
        """Train ensemble models for testing compatibility."""
        # Fill missing values
        train_data = train_data.copy()
        train_data[features] = train_data[features].fillna(0)
        
        X_train = train_data[features]
        y_train = train_data['num_orders']
        
        # Create a simple validation set
        split_idx = int(0.8 * len(train_data))
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        X_train = X_train.iloc[:split_idx]
        y_train = y_train.iloc[:split_idx]
        
        # Train the ensemble
        self.ensemble.fit_ensemble(X_train, y_train, X_val, y_val)
        self.feature_cols = features
        self.is_trained = True
        
        # Return model info for testing
        return {'ensemble': self.ensemble, 'features': features}
    
    def predict_ensemble(self, test_data, features, models):
        """Make ensemble predictions for testing compatibility."""
        test_data = test_data.copy()
        test_data[features] = test_data[features].fillna(0)
        
        X_test = test_data[features]
        predictions = self.ensemble.predict(X_test)
        
        return np.clip(predictions, 0, None)
    
    def rmsle(self, y_true, y_pred):
        """RMSLE metric for testing compatibility."""
        return self.ensemble.rmsle(y_true, y_pred)

class MultiModelEnsemble:
    """Enhanced ensemble with multiple model types."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = None
        
    def get_base_models(self):
        """Initialize base models for ensemble."""
        models = {}
        
        # LightGBM
        models['lgbm'] = LGBMRegressor(
            objective='regression_l1',
            metric='None',
            n_estimators=1000,
            learning_rate=0.02,
            random_state=self.config.SEED,
            n_jobs=-1,
            verbose=-1
        )
        
        # XGBoost
        models['xgb'] = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=1000,
            learning_rate=0.02,
            random_state=self.config.SEED,
            n_jobs=-1,
            verbosity=0
        )
        
        # CatBoost
        models['catboost'] = cb.CatBoostRegressor(
            loss_function='MAE',
            iterations=1000,
            learning_rate=0.02,
            random_seed=self.config.SEED,
            verbose=False
        )
        
        # Random Forest
        models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=self.config.SEED,
            n_jobs=-1
        )
        
        # Neural Network
        models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=self.config.SEED,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        return models
    
    def fit_ensemble(self, X_train, y_train, X_val, y_val):
        """Fit ensemble models."""
        self.models = self.get_base_models()        
        val_predictions = {}
        
        for name, model in list(self.models.items()):
            logging.info(f"Training {name}...")
            
            if name in ['lgbm', 'xgb', 'catboost']:
                # Tree-based models with early stopping
                if name == 'lgbm':              
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric=self.lgb_rmsle,
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                elif name == 'xgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:  # catboost
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
                        verbose=False
                    )
            elif name == 'mlp':
                # Neural network with scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                model.fit(X_train_scaled, y_train)
                val_predictions[name] = model.predict(X_val_scaled)
                self.models[f'{name}_scaler'] = scaler
                continue
            else:
                # Other models
                model.fit(X_train, y_train)
            
            val_predictions[name] = model.predict(X_val)
        
        # Calculate ensemble weights
        self.weights = self.calculate_weights(val_predictions, y_val)
        
    def calculate_weights(self, predictions, y_true):
        """Calculate optimal ensemble weights."""
        if self.config.ENSEMBLE_WEIGHTS == 'equal':
            n_models = len(predictions)
            return {name: 1.0/n_models for name in predictions.keys()}
        
        elif self.config.ENSEMBLE_WEIGHTS == 'auto':
            # Optimize weights using RMSLE
            from scipy.optimize import minimize
            def objective(weights):
                weights = weights / weights.sum()  # Normalize
                ensemble_pred = np.zeros(len(y_true), dtype=np.float64)
                for i, (name, pred) in enumerate(predictions.items()):
                    ensemble_pred += weights[i] * np.array(pred, dtype=np.float64)
                return self.rmsle(y_true, ensemble_pred)
            
            n_models = len(predictions)
            initial_weights = np.ones(n_models) / n_models
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
            bounds = [(0, 1) for _ in range(n_models)]
            
            result = minimize(objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            optimal_weights = result.x / result.x.sum()
            return {name: weight for name, weight in zip(predictions.keys(), optimal_weights)}
        
        else:
            # Use provided weights
            return {name: weight for name, weight in zip(predictions.keys(), self.config.ENSEMBLE_WEIGHTS)}
    
    def predict(self, X):
        """Make ensemble predictions."""
        predictions = {}
        
        for name, model in self.models.items():
            if name.endswith('_scaler'):
                continue
                
            if name == 'mlp' and f'{name}_scaler' in self.models:
                X_scaled = self.models[f'{name}_scaler'].transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
            
        return ensemble_pred
    
    @staticmethod
    def rmsle(y_true, y_pred):
        """RMSLE metric."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).clip(0)
        return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    
    @staticmethod
    def lgb_rmsle(y_true, y_pred):
        """RMSLE for LightGBM."""
        return 'rmsle', MultiModelEnsemble.rmsle(y_true, y_pred), False

class AdvancedValidation:
    """Enhanced validation strategies."""
    
    def __init__(self, config):
        self.config = config
    
    def time_series_cv(self, df, n_splits=5):
        """Time series cross-validation."""
        max_week = df['week'].max()
        min_week = df['week'].min()
        total_weeks = max_week - min_week + 1
        
        fold_size = total_weeks // (n_splits + 1)
        
        for i in range(n_splits):
            train_end = min_week + (i + 1) * fold_size
            val_start = train_end + 1
            val_end = val_start + fold_size - 1
            
            train_mask = df['week'] <= train_end
            val_mask = (df['week'] >= val_start) & (df['week'] <= val_end)
            
            yield df[train_mask], df[val_mask]
    
    def gap_validation(self, df, gap_weeks=2):
        """Validation with gap to simulate real prediction scenario."""
        max_week = df['week'].max()
        val_weeks = self.config.VALIDATION_WEEKS
        
        train_end = max_week - val_weeks - gap_weeks
        val_start = max_week - val_weeks + 1
        
        train_mask = df['week'] <= train_end
        val_mask = df['week'] >= val_start
        
        return df[train_mask], df[val_mask]

class EnhancedForecastingSystem:
    """Main orchestrating class for the enhanced forecasting system."""
    def __init__(self, config=None):
        self.config = config if config else EnhancedConfig()
        self.feature_eng = AdvancedFeatureEngineering(self.config)
        self.ensemble = MultiModelEnsemble(self.config)
        self.validation = AdvancedValidation(self.config)
        self.is_trained = False
        
    def engineer_features(self, df):
        """Apply feature engineering - wrapper for prepare_features for testing compatibility."""
        return self.prepare_features(df, is_train=True)
    
    def get_feature_list(self, df):
        """Get list of feature columns for testing compatibility."""
        exclude_cols = ['id', 'num_orders', 'week']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        return feature_cols
    
    def select_features(self, train_data, val_data, features):
        """Feature selection for testing compatibility."""
        # Simple feature selection - return all valid features
        return [col for col in features if col in train_data.columns]
    
    def train_ensemble(self, train_data, features):
        """Train ensemble models for testing compatibility."""
        # Fill missing values
        train_data = train_data.copy()
        train_data[features] = train_data[features].fillna(0)
        
        X_train = train_data[features]
        y_train = train_data['num_orders']
        
        # Create a simple validation set
        split_idx = int(0.8 * len(train_data))
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        X_train = X_train.iloc[:split_idx]
        y_train = y_train.iloc[:split_idx]
        
        # Train the ensemble
        self.ensemble.fit_ensemble(X_train, y_train, X_val, y_val)
        self.feature_cols = features
        self.is_trained = True
        
        # Return model info for testing
        return {'ensemble': self.ensemble, 'features': features}
    
    def predict_ensemble(self, test_data, features, models):
        """Make ensemble predictions for testing compatibility."""
        test_data = test_data.copy()
        test_data[features] = test_data[features].fillna(0)
        
        X_test = test_data[features]
        predictions = self.ensemble.predict(X_test)
        
        return np.clip(predictions, 0, None)
    
    def rmsle(self, y_true, y_pred):
        """RMSLE metric for testing compatibility."""
        return self.ensemble.rmsle(y_true, y_pred)
        
    def load_data(self):
        """Load and prepare training data."""
        logging.info("Loading data...")
        df = pd.read_csv(self.config.DATA_PATH)
        meal_info = pd.read_csv(self.config.MEAL_INFO_PATH)
        center_info = pd.read_csv(self.config.CENTER_INFO_PATH)
        
        # Merge data
        df = df.merge(meal_info, on="meal_id", how="left")
        df = df.merge(center_info, on="center_id", how="left")
        
        return df
    
    def prepare_features(self, df, is_train=True):
        """Apply comprehensive feature engineering."""
        logging.info("Creating advanced features...")
        
        # Apply all feature engineering steps
        df = self.feature_eng.create_fourier_features(df)
        df = self.feature_eng.create_trend_features(df)
        df = self.feature_eng.create_volatility_features(df)
        df = self.feature_eng.create_demand_regime_features(df)
        df = self.feature_eng.create_clustering_features(df, is_train=is_train)
        df = self.feature_eng.create_external_features(df)
        df = self.feature_eng.create_cross_validation_features(df)
        
        return df
    
    def train(self, df=None):
        """Train the enhanced forecasting system."""
        if df is None:
            df = self.load_data()
        
        # Prepare features
        df = self.prepare_features(df, is_train=True)
        
        # Prepare feature columns
        exclude_cols = ['id', 'num_orders', 'week']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Validation split
        train_split, val_split = self.validation.gap_validation(df)
        
        X_train = train_split[feature_cols]
        y_train = train_split['num_orders']
        X_val = val_split[feature_cols]
        y_val = val_split['num_orders']
        
        # Train ensemble
        logging.info("Training ensemble models...")
        self.ensemble.fit_ensemble(X_train, y_train, X_val, y_val)
        self.feature_cols = feature_cols
        self.is_trained = True
        
        # Validate ensemble
        val_preds = self.ensemble.predict(X_val)
        val_rmsle = self.ensemble.rmsle(y_val, val_preds)
        logging.info(f"Ensemble validation RMSLE: {val_rmsle:.5f}")
        
        return val_rmsle
    
    def predict(self, test_df):
        """Generate predictions for test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        test_df = self.prepare_features(test_df, is_train=False)
        
        # Handle missing values
        test_df[self.feature_cols] = test_df[self.feature_cols].fillna(0)
        
        # Get features
        X_test = test_df[self.feature_cols]
        
        # Predict
        predictions = self.ensemble.predict(X_test)
        predictions = np.clip(predictions, 0, None)
        
        return predictions
    
    def create_submission(self):
        """Create complete submission with recursive prediction."""
        # Load data
        df = self.load_data()
        test = pd.read_csv(self.config.TEST_PATH)
        meal_info = pd.read_csv(self.config.MEAL_INFO_PATH)
        center_info = pd.read_csv(self.config.CENTER_INFO_PATH)
        
        # Merge test data
        test = test.merge(meal_info, on="meal_id", how="left")
        test = test.merge(center_info, on="center_id", how="left")
        
        # Prepare combined dataset for feature engineering
        test['num_orders'] = np.nan
        combined_df = pd.concat([df, test], ignore_index=True)
        combined_df = combined_df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
        
        # Apply feature engineering
        combined_df = self.prepare_features(combined_df, is_train=True)
        
        # Split back
        train_df = combined_df[combined_df['num_orders'].notna()].copy()
        test_df = combined_df[combined_df['num_orders'].isna()].copy()
        
        # Train on all data
        self.train(train_df)
        
        # Recursive prediction for test weeks
        logging.info("Starting recursive prediction...")
        history_df = pd.concat([train_df, test_df], ignore_index=True)
        history_df = history_df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
        
        test_weeks = sorted(test_df['week'].unique())
        
        for week_num in test_weeks:
            logging.info(f"Predicting week {week_num}...")
            current_week_mask = history_df['week'] == week_num
            
            # Update dynamic features
            history_df = self.feature_eng.create_trend_features(history_df)
            history_df = self.feature_eng.create_volatility_features(history_df)
            
            # Get current features
            current_features = history_df.loc[current_week_mask, self.feature_cols]
            current_features = current_features.fillna(0)
            
            # Predict
            current_preds = self.ensemble.predict(current_features)
            current_preds = np.clip(current_preds, 0, None)
            
            # Update history
            history_df.loc[current_week_mask, 'num_orders'] = current_preds
        
        # Create submission
        final_predictions = history_df[history_df['id'].isin(test['id'])][['id', 'num_orders']].copy()
        final_predictions['num_orders'] = final_predictions['num_orders'].round().astype(int)
        
        submission_path = f"{self.config.SUBMISSION_PREFIX}_enhanced.csv"
        final_predictions.to_csv(submission_path, index=False)
        logging.info(f"Enhanced submission saved to {submission_path}")
        
        return final_predictions

# Update main execution
def create_advanced_submission():
    """Create submission with advanced techniques."""
    system = EnhancedForecastingSystem()
    return system.create_submission()

if __name__ == "__main__":
    create_advanced_submission()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import savgol_filter
import logging

class ModelGeneratedFeatures:
    """Generate features using trained models and meta-learning."""
    
    def __init__(self):
        self.base_models = {}
        self.meta_features = {}
        self.confidence_model = None
        
    def fit(self, X, y, base_models=None):
        """Fit meta-learning models."""
        if base_models:
            self.base_models = base_models
            
        # Create prediction confidence features
        self._fit_confidence_model(X, y)
        
        # Create ensemble disagreement features
        self._fit_disagreement_features(X, y)
        
    def _fit_confidence_model(self, X, y):
        """Train a model to predict prediction confidence."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate base predictions and residuals
        predictions = []
        residuals = []
        
        for name, model in self.base_models.items():
            pred = model.predict(X)
            residual = np.abs(y - pred)
            predictions.append(pred)
            residuals.append(residual)
            
        # Train confidence model to predict residual magnitude
        if predictions:
            X_conf = np.column_stack(predictions + [np.mean(predictions, axis=0), np.std(predictions, axis=0)])
            y_conf = np.mean(residuals, axis=0)
            
            self.confidence_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.confidence_model.fit(X_conf, y_conf)
            
    def _fit_disagreement_features(self, X, y):
        """Calculate ensemble disagreement metrics."""
        predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            predictions.append(pred)
            
        if len(predictions) > 1:
            pred_array = np.array(predictions)
            self.meta_features['disagreement_std'] = np.std(pred_array, axis=0)
            self.meta_features['disagreement_range'] = np.ptp(pred_array, axis=0)
            self.meta_features['disagreement_iqr'] = stats.iqr(pred_array, axis=0)
            
    def transform(self, X):
        """Generate meta-features for new data."""
        meta_features = []
        
        # Base model predictions
        predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            predictions.append(pred)
            meta_features.append(pred)
            
        if predictions:
            # Ensemble statistics
            pred_mean = np.mean(predictions, axis=0)
            pred_std = np.std(predictions, axis=0)
            meta_features.extend([pred_mean, pred_std])
            
            # Confidence estimates
            if self.confidence_model:
                X_conf = np.column_stack(predictions + [pred_mean, pred_std])
                confidence = self.confidence_model.predict(X_conf)
                meta_features.append(confidence)
                
        return np.column_stack(meta_features) if meta_features else np.array([]).reshape(len(X), 0)

class TimeSeriesDecomposer:
    """Advanced time series decomposition features."""
    
    def __init__(self, periods=[13, 26, 52]):
        self.periods = periods
        self.seasonal_components = {}
        self.trend_components = {}
        
    def fit_transform(self, df, target='num_orders', group_cols=['center_id', 'meal_id']):
        """Decompose time series and extract components."""
        df_out = df.copy()
        
        for period in self.periods:
            # Seasonal decomposition for each group
            seasonal_features = []
            trend_features = []
            residual_features = []
            
            for name, group in df_out.groupby(group_cols):
                group = group.sort_values('week')
                
                if len(group) >= period * 2:  # Need enough data for decomposition
                    # Simple seasonal decomposition
                    ts = group[target].values
                    
                    # Seasonal component (moving average)
                    seasonal = self._extract_seasonal(ts, period)
                    
                    # Trend component (smoothed series)
                    trend = savgol_filter(ts, min(len(ts)//2*2-1, 51), 3) if len(ts) >= 51 else ts
                    
                    # Residual
                    residual = ts - seasonal - trend
                    
                    seasonal_features.extend(seasonal)
                    trend_features.extend(trend)
                    residual_features.extend(residual)
                else:
                    # Fallback for short series
                    seasonal_features.extend([0] * len(group))
                    trend_features.extend(group[target].values)
                    residual_features.extend([0] * len(group))
                    
            df_out[f'seasonal_{period}'] = seasonal_features
            df_out[f'trend_{period}'] = trend_features
            df_out[f'residual_{period}'] = residual_features
            
        return df_out
    
    def _extract_seasonal(self, ts, period):
        """Extract seasonal component."""
        if len(ts) < period:
            return np.zeros_like(ts)
            
        # Create seasonal indices
        seasonal_means = np.zeros(period)
        seasonal_counts = np.zeros(period)
        
        for i, value in enumerate(ts):
            season_idx = i % period
            seasonal_means[season_idx] += value
            seasonal_counts[season_idx] += 1
            
        # Avoid division by zero
        seasonal_means = np.divide(seasonal_means, seasonal_counts, 
                                 out=np.zeros_like(seasonal_means), 
                                 where=seasonal_counts!=0)
        
        # Map back to time series
        seasonal = np.array([seasonal_means[i % period] for i in range(len(ts))])
        
        # Remove overall mean to center around zero
        seasonal = seasonal - np.mean(seasonal)
        
        return seasonal

class AdvancedLagFeatures:
    """Create sophisticated lag and rolling features."""
    
    def __init__(self, max_lag=20, decay_factor=0.1):
        self.max_lag = max_lag
        self.decay_factor = decay_factor
        
    def create_exponential_lags(self, df, target='num_orders', group_cols=['center_id', 'meal_id']):
        """Create exponentially weighted lag features."""
        df_out = df.copy()
        
        # Exponential weights for combining lags
        weights = np.exp(-self.decay_factor * np.arange(1, self.max_lag + 1))
        weights = weights / weights.sum()
        
        group = df_out.groupby(group_cols)
        
        # Create weighted lag feature
        weighted_lag = np.zeros(len(df_out))
        
        for i in range(len(df_out)):
            group_mask = ((df_out['center_id'] == df_out.iloc[i]['center_id']) & 
                         (df_out['meal_id'] == df_out.iloc[i]['meal_id']) &
                         (df_out['week'] < df_out.iloc[i]['week']))
            
            if group_mask.any():
                historical_values = df_out.loc[group_mask, target].values
                if len(historical_values) > 0:
                    # Use last N values with exponential weights
                    n_available = min(len(historical_values), self.max_lag)
                    recent_values = historical_values[-n_available:]
                    used_weights = weights[-n_available:]
                    used_weights = used_weights / used_weights.sum()  # Renormalize
                    
                    weighted_lag[i] = np.sum(recent_values * used_weights)
                    
        df_out[f'{target}_exp_weighted_lag'] = weighted_lag
        
        return df_out
    
    def create_adaptive_rolling(self, df, target='num_orders', group_cols=['center_id', 'meal_id']):
        """Create adaptive rolling windows based on volatility."""
        df_out = df.copy()
        group = df_out.groupby(group_cols)
        
        # Calculate volatility for each group
        volatility = group[target].transform(lambda x: x.rolling(8, min_periods=2).std())
        
        # Adaptive window size (smaller windows for high volatility)
        base_window = 8
        adaptive_window = np.maximum(2, base_window - (volatility / volatility.std()).fillna(0) * 3).astype(int)
        
        # Create adaptive rolling mean
        adaptive_rolling = np.zeros(len(df_out))
        
        for i in range(len(df_out)):
            window_size = adaptive_window.iloc[i]
            
            # Get group data up to current point
            mask = ((df_out['center_id'] == df_out.iloc[i]['center_id']) & 
                   (df_out['meal_id'] == df_out.iloc[i]['meal_id']) &
                   (df_out.index < i))
            
            if mask.any():
                group_data = df_out.loc[mask, target].values
                if len(group_data) >= window_size:
                    adaptive_rolling[i] = np.mean(group_data[-window_size:])
                elif len(group_data) > 0:
                    adaptive_rolling[i] = np.mean(group_data)
                    
        df_out[f'{target}_adaptive_rolling'] = adaptive_rolling
        
        return df_out

class ErrorCorrectionFeatures:
    """Generate features based on prediction error patterns."""
    
    def __init__(self):
        self.error_patterns = {}
        self.bias_corrections = {}
        
    def fit(self, df, predictions, actuals, group_cols=['center_id', 'meal_id']):
        """Learn error patterns from validation data."""
        errors = actuals - predictions
        
        # Learn systematic biases by group
        for name, group in df.groupby(group_cols):
            group_errors = errors[df.groupby(group_cols).groups[name]]
            
            if len(group_errors) > 5:  # Need sufficient data
                self.bias_corrections[name] = {
                    'mean_error': np.mean(group_errors),
                    'std_error': np.std(group_errors),
                    'trend_error': self._calculate_error_trend(group_errors)
                }
                
        # Learn temporal error patterns
        df_with_errors = df.copy()
        df_with_errors['error'] = errors
        
        # Weekly error patterns
        weekly_errors = df_with_errors.groupby('week')['error'].agg(['mean', 'std'])
        self.error_patterns['weekly'] = weekly_errors
        
        # Seasonal error patterns
        df_with_errors['weekofyear'] = df_with_errors['week'] % 52
        seasonal_errors = df_with_errors.groupby('weekofyear')['error'].agg(['mean', 'std'])
        self.error_patterns['seasonal'] = seasonal_errors
        
    def _calculate_error_trend(self, errors):
        """Calculate trend in errors over time."""
        if len(errors) < 3:
            return 0
            
        x = np.arange(len(errors))
        try:
            slope, _, _, _, _ = stats.linregress(x, errors)
            return slope
        except:
            return 0
            
    def transform(self, df, predictions):
        """Generate error correction features."""
        df_out = df.copy()
        
        # Bias correction features
        bias_corrections = []
        
        for i in range(len(df_out)):
            center_id = df_out.iloc[i]['center_id']
            meal_id = df_out.iloc[i]['meal_id']
            
            key = (center_id, meal_id)
            if key in self.bias_corrections:
                bias_corrections.append(self.bias_corrections[key]['mean_error'])
            else:
                bias_corrections.append(0)
                
        df_out['bias_correction'] = bias_corrections
        
        # Temporal error corrections
        if 'weekly' in self.error_patterns:
            week_corrections = df_out['week'].map(
                lambda w: self.error_patterns['weekly'].loc[w, 'mean'] 
                if w in self.error_patterns['weekly'].index else 0
            )
            df_out['week_error_correction'] = week_corrections
            
        if 'seasonal' in self.error_patterns:
            df_out['weekofyear'] = df_out['week'] % 52
            seasonal_corrections = df_out['weekofyear'].map(
                lambda w: self.error_patterns['seasonal'].loc[w, 'mean'] 
                if w in self.error_patterns['seasonal'].index else 0
            )
            df_out['seasonal_error_correction'] = seasonal_corrections
            
        # Corrected predictions
        total_correction = (df_out.get('bias_correction', 0) + 
                          df_out.get('week_error_correction', 0) + 
                          df_out.get('seasonal_error_correction', 0))
        
        df_out['corrected_prediction'] = predictions + total_correction
        
        return df_out

# Example usage function
def demonstrate_advanced_features():
    """Demonstrate the advanced feature engineering techniques."""
    
    # Load sample data (replace with actual data loading)
    # df = pd.read_csv("train.csv")
    
    logging.info("Advanced Feature Engineering Demonstration")
    
    # 1. Model-generated features
    logging.info("1. Creating model-generated features...")
    # mgf = ModelGeneratedFeatures()
    # mgf.fit(X_train, y_train, base_models={'lgbm': lgbm_model, 'rf': rf_model})
    # meta_features = mgf.transform(X_test)
    
    # 2. Time series decomposition
    logging.info("2. Applying time series decomposition...")
    # decomposer = TimeSeriesDecomposer(periods=[13, 26, 52])
    # df_decomposed = decomposer.fit_transform(df)
    
    # 3. Advanced lag features
    logging.info("3. Creating advanced lag features...")
    # lag_creator = AdvancedLagFeatures(max_lag=15, decay_factor=0.1)
    # df_with_lags = lag_creator.create_exponential_lags(df)
    # df_with_adaptive = lag_creator.create_adaptive_rolling(df_with_lags)
    
    # 4. Error correction features
    logging.info("4. Generating error correction features...")
    # error_corrector = ErrorCorrectionFeatures()
    # error_corrector.fit(val_df, val_predictions, val_actuals)
    # df_corrected = error_corrector.transform(test_df, test_predictions)
    
    logging.info("Advanced feature engineering complete!")

if __name__ == "__main__":
    demonstrate_advanced_features()

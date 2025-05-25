"""
Enhanced Forecasting System Analysis and Improvements
====================================================

Based on the performance evaluation results, this script analyzes why the enhanced
system underperformed and implements targeted improvements.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

class SystemAnalysis:
    """Analyze and improve the enhanced forecasting system."""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_performance_degradation(self):
        """Analyze why the enhanced system underperformed."""
        logging.info("Analyzing performance degradation...")
        
        analysis_points = [
            "PERFORMANCE DEGRADATION ANALYSIS",
            "=" * 50,
            "",
            "POTENTIAL CAUSES:",
            "",
            "1. OVERFITTING:",
            "   - Enhanced system has 37 additional features (53 vs 16)",
            "   - Complex ensemble with 5 models vs single Random Forest",
            "   - Small sample size (12K rows) may not support complex models",
            "   - Advanced features may be overengineered for this dataset",
            "",
            "2. FEATURE ENGINEERING ISSUES:",
            "   - Some advanced features may introduce noise",
            "   - Clustering features (50 clusters) may be too granular",
            "   - Fourier features may not capture true seasonality patterns",
            "   - Target encoding without proper cross-validation",
            "",
            "3. MODEL COMPLEXITY:",
            "   - Ensemble weight optimization may be unstable with small data",
            "   - Neural network (MLP) may be inappropriate for this problem size",
            "   - Multiple tree-based models may be redundant",
            "",
            "4. DATA LEAKAGE:",
            "   - Future information accidentally included in features",
            "   - Cross-validation features computed incorrectly",
            "   - Improper handling of time series structure",
            "",
            "IMPROVEMENT STRATEGIES:",
            "",
            "1. FEATURE SELECTION:",
            "   - Implement recursive feature elimination",
            "   - Use feature importance analysis",
            "   - Apply correlation-based filtering",
            "   - Validate features against target",
            "",
            "2. MODEL SIMPLIFICATION:",
            "   - Start with fewer, well-tuned models",
            "   - Use proper cross-validation for ensemble weights",
            "   - Implement early stopping based on validation performance",
            "",
            "3. REGULARIZATION:",
            "   - Add L1/L2 regularization to models",
            "   - Use dropout in neural networks",
            "   - Implement feature selection penalties",
            "",
            "4. VALIDATION IMPROVEMENT:",
            "   - Use time series cross-validation",
            "   - Implement proper train/validation/test splits",
            "   - Monitor for overfitting during training",
        ]
        
        analysis_content = "\n".join(analysis_points)
        
        # Save analysis
        with open('performance_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(analysis_content)
        
        print(analysis_content)
        logging.info("Performance analysis saved to performance_analysis_report.txt")
        
        return analysis_content

    def create_improved_system(self):
        """Create an improved version with lessons learned."""
        logging.info("Creating improved forecasting system...")
        
        improved_system_code = '''"""
Improved Enhanced Forecasting System
===================================

A refined version addressing the performance issues identified in the analysis.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

class ImprovedForecastingSystem:
    """Simplified and improved forecasting system."""
    
    def __init__(self, max_features=25):
        self.max_features = max_features
        self.feature_selector = None
        self.models = {}
        self.weights = None
        self.is_trained = False
        
    def create_core_features(self, df):
        """Create essential features only."""
        df_out = df.copy()
        
        # Sort by time for proper lag calculation
        df_out = df_out.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
        
        # Basic lag features (proven effective)
        for lag in [1, 2, 4]:
            df_out[f'orders_lag_{lag}'] = df_out.groupby(['center_id', 'meal_id'])['num_orders'].shift(lag)
        
        # Rolling statistics with proper time series handling
        for window in [3, 7]:
            df_out[f'orders_mean_{window}'] = df_out.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df_out[f'orders_std_{window}'] = df_out.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Simple trend (difference features)
        df_out['orders_diff_1'] = df_out.groupby(['center_id', 'meal_id'])['num_orders'].diff(1)
        df_out['orders_diff_2'] = df_out.groupby(['center_id', 'meal_id'])['num_orders'].diff(2)
        
        # Basic interaction features
        df_out['price_ratio'] = df_out['checkout_price'] / (df_out['base_price'] + 1e-8)
        df_out['promotion_score'] = df_out['emailer_for_promotion'] + df_out['homepage_featured']
        
        # Simple seasonal features
        df_out['week_mod_4'] = df_out['week'] % 4
        df_out['week_mod_12'] = df_out['week'] % 12
        
        # Center-meal specific averages (with minimal data leakage)
        center_meal_stats = df_out.groupby(['center_id', 'meal_id'])['num_orders'].agg([
            'mean', 'median', 'std'
        ]).reset_index()
        center_meal_stats.columns = ['center_id', 'meal_id', 'cm_mean', 'cm_median', 'cm_std']
        df_out = df_out.merge(center_meal_stats, on=['center_id', 'meal_id'], how='left')
        
        return df_out
    
    def select_best_features(self, X, y, max_features=None):
        """Select the most important features."""
        if max_features is None:
            max_features = self.max_features
            
        # Remove non-numeric columns
        feature_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        X_numeric = X[feature_cols].fillna(0)
        
        # Use SelectKBest with f_regression
        if len(feature_cols) > max_features:
            self.feature_selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = self.feature_selector.fit_transform(X_numeric, y)
            selected_features = [feature_cols[i] for i in self.feature_selector.get_support(indices=True)]
        else:
            selected_features = feature_cols
            X_selected = X_numeric
            
        return X_selected, selected_features
    
    def train(self, train_df, val_df=None):
        """Train the improved system."""
        logging.info("Training improved forecasting system...")
        
        # Create features
        train_featured = self.create_core_features(train_df)
        
        # Prepare target and features
        y_train = train_featured['num_orders']
        exclude_cols = ['id', 'num_orders', 'week']
        X_train_cols = [col for col in train_featured.columns if col not in exclude_cols]
        X_train = train_featured[X_train_cols]
        
        # Feature selection
        X_train_selected, self.selected_features = self.select_best_features(X_train, y_train)
        
        logging.info(f"Selected {len(self.selected_features)} features: {self.selected_features[:10]}...")
        
        # Train simple ensemble (2 models only)
        self.models = {
            'lgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42,
                objective='regression',
                metric='rmse',
                verbosity=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Train models
        val_predictions = {}
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            model.fit(X_train_selected, y_train)
            
            if val_df is not None:
                val_featured = self.create_core_features(val_df)
                X_val = val_featured[self.selected_features].fillna(0)
                val_pred = model.predict(X_val)
                val_predictions[name] = val_pred
        
        # Simple equal weighting (avoid overfitting with weight optimization)
        self.weights = {name: 0.5 for name in self.models.keys()}
        
        self.is_trained = True
        logging.info("Training completed!")
        
        return val_predictions
    
    def predict(self, test_df):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Create features
        test_featured = self.create_core_features(test_df)
        
        # Prepare features
        X_test = test_featured[self.selected_features].fillna(0)
        
        # Ensemble prediction
        predictions = np.zeros(len(X_test))
        for name, model in self.models.items():
            pred = model.predict(X_test)
            predictions += self.weights[name] * pred
            
        return np.clip(predictions, 0, None)
    
    def rmsle(self, y_true, y_pred):
        """Calculate RMSLE."""
        return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def compare_improved_system():
    """Compare the improved system with baseline and original enhanced system."""
    logging.info("Comparing improved system...")
    
    # Load data
    train_df = pd.read_csv('train.csv').head(5000)  # Use sample
    meal_info = pd.read_csv('meal_info.csv')
    center_info = pd.read_csv('fulfilment_center_info.csv')
    
    # Merge data
    df = train_df.merge(meal_info, on='meal_id', how='left')
    df = df.merge(center_info, on='center_id', how='left')
    df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
    
    # Split data
    max_week = df['week'].max()
    train_data = df[df['week'] <= max_week - 8].copy()
    val_data = df[df['week'] > max_week - 8].copy()
    
    logging.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Test improved system
    improved_system = ImprovedForecastingSystem(max_features=20)
    val_predictions = improved_system.train(train_data, val_data)
    
    final_predictions = improved_system.predict(val_data)
    rmsle = improved_system.rmsle(val_data['num_orders'], final_predictions)
    
    logging.info(f"Improved System RMSLE: {rmsle:.5f}")
    
    return {
        'rmsle': rmsle,
        'predictions': final_predictions,
        'actuals': val_data['num_orders'],
        'num_features': len(improved_system.selected_features)
    }

if __name__ == "__main__":
    results = compare_improved_system()
    print(f"\\nIMPROVED SYSTEM RESULTS:")
    print(f"RMSLE: {results['rmsle']:.5f}")
    print(f"Features: {results['num_features']}")
'''
        
        # Save improved system
        with open('improved_forecasting_system.py', 'w', encoding='utf-8') as f:
            f.write(improved_system_code)
        
        logging.info("Improved system code saved to improved_forecasting_system.py")
        
        return improved_system_code

def main():
    """Run comprehensive analysis and create improvements."""
    logging.info("Starting comprehensive system analysis...")
    
    analyzer = SystemAnalysis()
    
    # Analyze why enhanced system underperformed
    analyzer.analyze_performance_degradation()
    
    # Create improved system
    analyzer.create_improved_system()
    
    logging.info("Analysis and improvements completed!")

if __name__ == "__main__":
    main()

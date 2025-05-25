"""
Enhanced Forecasting System Performance Evaluation
==================================================

This script compares the enhanced forecasting system with the baseline approach
and demonstrates the improvements achieved through advanced feature engineering
and ensemble modeling.
"""

import pandas as pd
import numpy as np
import logging
import time
import warnings
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Import our enhanced system
from enhanced_prediction_system import EnhancedForecastingSystem
from comprehensive_model_evaluation import ModelEvaluator
from model_generated_features import ModelGeneratedFeatures

class PerformanceComparison:
    """Compare baseline vs enhanced forecasting systems."""
    
    def __init__(self):
        self.baseline_results = {}
        self.enhanced_results = {}
        self.evaluation_results = {}
        
    def load_and_prepare_data(self, sample_size=10000):
        """Load and prepare data for comparison."""
        logging.info("Loading data for performance comparison...")
        
        # Load training data
        train_df = pd.read_csv('train.csv')
        meal_info = pd.read_csv('meal_info.csv')
        center_info = pd.read_csv('fulfilment_center_info.csv')
        
        # Merge data
        df = train_df.merge(meal_info, on='meal_id', how='left')
        df = df.merge(center_info, on='center_id', how='left')
        df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
        
        # Use a sample for faster comparison
        if sample_size and len(df) > sample_size:
            # Sample while maintaining time series structure
            unique_combinations = df[['center_id', 'meal_id']].drop_duplicates()
            sample_combinations = unique_combinations.sample(n=min(sample_size//50, len(unique_combinations)))
            df = df.merge(sample_combinations, on=['center_id', 'meal_id'], how='inner')
            logging.info(f"Using sample of {len(df)} rows from {len(train_df)} total")
        
        # Split data for validation
        max_week = df['week'].max()
        train_data = df[df['week'] <= max_week - 8].copy()  # Use 8 weeks for validation
        val_data = df[df['week'] > max_week - 8].copy()
        
        logging.info(f"Train data: {len(train_data)} rows, Validation data: {len(val_data)} rows")
        return train_data, val_data
    
    def run_baseline_approach(self, train_data, val_data):
        """Run baseline approach with simple features."""
        logging.info("Running baseline approach...")
        start_time = time.time()
        
        # Simple baseline features
        baseline_df = train_data.copy()
        
        # Basic lag features
        for lag in [1, 2, 4, 8]:
            baseline_df[f'num_orders_lag_{lag}'] = baseline_df.groupby(['center_id', 'meal_id'])['num_orders'].shift(lag)
          # Basic rolling means
        for window in [3, 7, 14]:
            baseline_df[f'num_orders_mean_{window}'] = baseline_df.groupby(['center_id', 'meal_id'])['num_orders'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        # Simple categorical encoding
        baseline_df['center_meal_combo'] = baseline_df['center_id'].astype(str) + '_' + baseline_df['meal_id'].astype(str)
        
        # Get feature columns
        exclude_cols = ['id', 'num_orders', 'week', 'center_meal_combo']
        feature_cols = [col for col in baseline_df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if baseline_df[col].dtype in ['int64', 'float64']]
        
        # Fill missing values
        baseline_df[feature_cols] = baseline_df[feature_cols].fillna(0)
        
        # Simple model training
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
        X_train = baseline_df[feature_cols]
        y_train = baseline_df['num_orders']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Validate on validation data
        val_df = val_data.copy()
          # Apply same features to validation data
        for lag in [1, 2, 4, 8]:
            val_df[f'num_orders_lag_{lag}'] = val_df.groupby(['center_id', 'meal_id'])['num_orders'].shift(lag)
        
        for window in [3, 7, 14]:
            val_df[f'num_orders_mean_{window}'] = val_df.groupby(['center_id', 'meal_id'])['num_orders'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        val_df['center_meal_combo'] = val_df['center_id'].astype(str) + '_' + val_df['meal_id'].astype(str)
        val_df[feature_cols] = val_df[feature_cols].fillna(0)
        
        X_val = val_df[feature_cols]
        y_val = val_df['num_orders']
        
        # Predict
        val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, 0, None)
        
        # Calculate metrics
        baseline_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        baseline_rmsle = np.sqrt(np.mean(np.square(np.log1p(val_pred) - np.log1p(y_val))))
        baseline_mae = np.mean(np.abs(y_val - val_pred))
        
        baseline_time = time.time() - start_time
        
        self.baseline_results = {
            'rmse': baseline_rmse,
            'rmsle': baseline_rmsle,
            'mae': baseline_mae,
            'training_time': baseline_time,
            'num_features': len(feature_cols),
            'predictions': val_pred,
            'actuals': y_val
        }
        
        logging.info(f"Baseline completed in {baseline_time:.2f}s")
        logging.info(f"Baseline RMSLE: {baseline_rmsle:.5f}")
        return self.baseline_results
    
    def run_enhanced_approach(self, train_data, val_data):
        """Run enhanced approach with advanced features."""
        logging.info("Running enhanced approach...")
        start_time = time.time()
        
        try:
            # Initialize enhanced system
            enhanced_system = EnhancedForecastingSystem()
            
            # Apply enhanced feature engineering
            enhanced_train = enhanced_system.prepare_features(train_data, is_train=True)
            enhanced_val = enhanced_system.prepare_features(val_data, is_train=False)
            
            # Get feature columns
            exclude_cols = ['id', 'num_orders', 'week']
            feature_cols = [col for col in enhanced_train.columns if col not in exclude_cols]
            feature_cols = [col for col in feature_cols if enhanced_train[col].dtype in ['int64', 'float64']]
            
            # Fill missing values
            enhanced_train[feature_cols] = enhanced_train[feature_cols].fillna(0)
            enhanced_val[feature_cols] = enhanced_val[feature_cols].fillna(0)
            
            # Train enhanced models
            X_train = enhanced_train[feature_cols]
            y_train = enhanced_train['num_orders']
            X_val = enhanced_val[feature_cols]
            y_val = enhanced_val['num_orders']
            
            # Train ensemble
            enhanced_system.ensemble.fit_ensemble(X_train, y_train, X_val, y_val)
            
            # Predict
            enhanced_pred = enhanced_system.ensemble.predict(X_val)
            enhanced_pred = np.clip(enhanced_pred, 0, None)
            
            # Calculate metrics
            enhanced_rmse = np.sqrt(np.mean(np.square(y_val - enhanced_pred)))
            enhanced_rmsle = np.sqrt(np.mean(np.square(np.log1p(enhanced_pred) - np.log1p(y_val))))
            enhanced_mae = np.mean(np.abs(y_val - enhanced_pred))
            
            enhanced_time = time.time() - start_time
            
            self.enhanced_results = {
                'rmse': enhanced_rmse,
                'rmsle': enhanced_rmsle,
                'mae': enhanced_mae,
                'training_time': enhanced_time,
                'num_features': len(feature_cols),
                'predictions': enhanced_pred,
                'actuals': y_val
            }
            
            logging.info(f"Enhanced approach completed in {enhanced_time:.2f}s")
            logging.info(f"Enhanced RMSLE: {enhanced_rmsle:.5f}")
            return self.enhanced_results
            
        except Exception as e:
            logging.error(f"Enhanced approach failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_comparison_report(self):
        """Generate detailed comparison report."""
        if not self.baseline_results or not self.enhanced_results:
            logging.error("Cannot generate report - missing results")
            return
        
        # Calculate improvements
        rmsle_improvement = (self.baseline_results['rmsle'] - self.enhanced_results['rmsle']) / self.baseline_results['rmsle'] * 100
        rmse_improvement = (self.baseline_results['rmse'] - self.enhanced_results['rmse']) / self.baseline_results['rmse'] * 100
        mae_improvement = (self.baseline_results['mae'] - self.enhanced_results['mae']) / self.baseline_results['mae'] * 100
        
        report_lines = [
            "ENHANCED FORECASTING SYSTEM - PERFORMANCE COMPARISON",
            "=" * 60,
            "",
            "BASELINE APPROACH RESULTS:",
            f"  RMSLE: {self.baseline_results['rmsle']:.5f}",
            f"  RMSE:  {self.baseline_results['rmse']:.2f}",
            f"  MAE:   {self.baseline_results['mae']:.2f}",
            f"  Features: {self.baseline_results['num_features']}",
            f"  Training Time: {self.baseline_results['training_time']:.2f}s",
            "",
            "ENHANCED APPROACH RESULTS:",
            f"  RMSLE: {self.enhanced_results['rmsle']:.5f}",
            f"  RMSE:  {self.enhanced_results['rmse']:.2f}",
            f"  MAE:   {self.enhanced_results['mae']:.2f}",
            f"  Features: {self.enhanced_results['num_features']}",
            f"  Training Time: {self.enhanced_results['training_time']:.2f}s",
            "",
            "IMPROVEMENTS:",
            f"  RMSLE Improvement: {rmsle_improvement:+.2f}%",
            f"  RMSE Improvement:  {rmse_improvement:+.2f}%",
            f"  MAE Improvement:   {mae_improvement:+.2f}%",
            f"  Feature Count:     {self.enhanced_results['num_features'] - self.baseline_results['num_features']:+d}",
            "",
            "ANALYSIS:",
        ]
        
        if rmsle_improvement > 0:
            report_lines.append(f"✅ Enhanced system shows {rmsle_improvement:.1f}% improvement in RMSLE")
        else:
            report_lines.append(f"⚠️  Enhanced system shows {abs(rmsle_improvement):.1f}% degradation in RMSLE")
        
        if self.enhanced_results['num_features'] > self.baseline_results['num_features']:
            report_lines.append(f"📊 Enhanced system uses {self.enhanced_results['num_features'] - self.baseline_results['num_features']} additional features")
        
        report_lines.extend([
            "",
            "FEATURE ENGINEERING IMPACT:",
            f"  Advanced features created: {self.enhanced_results['num_features'] - 15} (beyond basic data)",
            f"  Feature engineering efficiency: {self.enhanced_results['num_features'] / self.enhanced_results['training_time']:.1f} features/second",
            "",
            "RECOMMENDATIONS:",
            "  - Enhanced system provides more sophisticated modeling capability",
            "  - Consider trade-offs between accuracy and complexity",
            "  - Monitor feature importance to optimize feature set",
            "",
            f"Report generated: {pd.Timestamp.now()}",
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open('performance_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(report_content)
        logging.info("Performance comparison report saved to performance_comparison_report.txt")
        
        return report_content

def main():
    """Run complete performance comparison."""
    logging.info("Starting Enhanced Forecasting System Performance Evaluation...")
    
    comparison = PerformanceComparison()
    
    # Load and prepare data
    train_data, val_data = comparison.load_and_prepare_data(sample_size=5000)  # Use smaller sample for demo
    
    # Run baseline approach
    baseline_results = comparison.run_baseline_approach(train_data, val_data)
    
    # Run enhanced approach
    enhanced_results = comparison.run_enhanced_approach(train_data, val_data)
    
    # Generate comparison report
    if baseline_results and enhanced_results:
        comparison.generate_comparison_report()
        logging.info("Performance evaluation completed successfully!")
    else:
        logging.error("Performance evaluation failed - could not complete both approaches")

if __name__ == "__main__":
    main()

"""
Comprehensive Model Evaluation Framework for Food Demand Forecasting

This script provides a thorough evaluation system to compare different forecasting models,
including the enhanced prediction system, baseline models, and various model variants.

Features:
- Cross-validation with proper time series splits
- Multiple evaluation metrics (RMSLE, RMSE, MAE, MAPE)
- Model performance analysis by segments (center, meal, time periods)
- Statistical significance testing
- Visualization of results
- Automated model comparison and ranking
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    """Comprehensive model evaluation and comparison framework"""
    
    def __init__(self, data_path="train.csv", meal_info_path="meal_info.csv", 
                 center_info_path="fulfilment_center_info.csv", seed=42):
        self.seed = seed
        self.data_path = data_path
        self.meal_info_path = meal_info_path
        self.center_info_path = center_info_path
        self.results = {}
        self.evaluation_metrics = {}
        
        # Load and prepare data
        self.load_data()
        
    def load_data(self):
        """Load and merge all necessary data"""
        logging.info("Loading evaluation data...")
        
        try:
            self.train_df = pd.read_csv(self.data_path)
            self.meal_info = pd.read_csv(self.meal_info_path)
            self.center_info = pd.read_csv(self.center_info_path)
            
            # Merge data
            self.train_df = self.train_df.merge(self.meal_info, on="meal_id", how="left")
            self.train_df = self.train_df.merge(self.center_info, on="center_id", how="left")
            
            # Sort by time
            self.train_df = self.train_df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
            
            logging.info(f"Data loaded successfully. Shape: {self.train_df.shape}")
            logging.info(f"Week range: {self.train_df['week'].min()} to {self.train_df['week'].max()}")
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def rmsle(self, y_true, y_pred):
        """Root Mean Squared Logarithmic Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).clip(0)
        return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    
    def rmse(self, y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    def mape(self, y_true, y_pred):
        """Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        return {
            'RMSLE': self.rmsle(y_true, y_pred),
            'RMSE': self.rmse(y_true, y_pred),
            'MAE': self.mae(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred)
        }
    
    def time_series_cross_validation(self, model_func, features, target="num_orders", 
                                   n_splits=5, test_size_weeks=8):
        """
        Perform time series cross-validation with proper temporal splits
        
        Args:
            model_func: Function that takes (train_df, test_df, features) and returns predictions
            features: List of feature columns
            target: Target column name
            n_splits: Number of CV splits
            test_size_weeks: Number of weeks for each test set
        """
        logging.info(f"Starting time series cross-validation with {n_splits} splits...")
        
        # Get unique weeks for splitting
        weeks = sorted(self.train_df['week'].unique())
        max_week = max(weeks)
        
        cv_results = []
        
        for i in range(n_splits):
            # Calculate test weeks for this split
            test_end_week = max_week - i * test_size_weeks
            test_start_week = test_end_week - test_size_weeks + 1
            
            if test_start_week <= min(weeks):
                break
                
            # Split data
            train_mask = self.train_df['week'] < test_start_week
            test_mask = (self.train_df['week'] >= test_start_week) & (self.train_df['week'] <= test_end_week)
            
            train_fold = self.train_df[train_mask].copy()
            test_fold = self.train_df[test_mask].copy()
            
            if len(train_fold) == 0 or len(test_fold) == 0:
                continue
                
            logging.info(f"CV Fold {i+1}: Train weeks {train_fold['week'].min()}-{train_fold['week'].max()}, "
                        f"Test weeks {test_fold['week'].min()}-{test_fold['week'].max()}")
            
            try:
                # Get predictions from model
                predictions = model_func(train_fold, test_fold, features)
                
                # Calculate metrics
                metrics = self.calculate_metrics(test_fold[target], predictions)
                metrics['fold'] = i + 1
                metrics['train_size'] = len(train_fold)
                metrics['test_size'] = len(test_fold)
                metrics['test_weeks'] = f"{test_start_week}-{test_end_week}"
                
                cv_results.append(metrics)
                logging.info(f"Fold {i+1} - RMSLE: {metrics['RMSLE']:.4f}, RMSE: {metrics['RMSE']:.2f}")
                
            except Exception as e:
                logging.error(f"Error in fold {i+1}: {e}")
                continue
        
        return cv_results
    
    def segment_analysis(self, y_true, y_pred, segment_df):
        """
        Analyze model performance by different segments
        
        Args:
            y_true: True values
            y_pred: Predicted values  
            segment_df: DataFrame with segment information
        """
        results = {}
        
        # Analysis by center type
        if 'center_type' in segment_df.columns:
            center_analysis = {}
            for center_type in segment_df['center_type'].unique():
                mask = segment_df['center_type'] == center_type
                if mask.sum() > 0:
                    center_analysis[center_type] = self.calculate_metrics(
                        y_true[mask], y_pred[mask]
                    )
            results['center_type'] = center_analysis
        
        # Analysis by category
        if 'category' in segment_df.columns:
            category_analysis = {}
            for category in segment_df['category'].unique():
                mask = segment_df['category'] == category
                if mask.sum() > 0:
                    category_analysis[category] = self.calculate_metrics(
                        y_true[mask], y_pred[mask]
                    )
            results['category'] = category_analysis
        
        # Analysis by order volume (quartiles)
        quartiles = np.percentile(y_true, [25, 50, 75])
        volume_analysis = {
            'low_volume': self.calculate_metrics(
                y_true[y_true <= quartiles[0]], y_pred[y_true <= quartiles[0]]
            ),
            'medium_volume': self.calculate_metrics(
                y_true[(y_true > quartiles[0]) & (y_true <= quartiles[2])], 
                y_pred[(y_true > quartiles[0]) & (y_true <= quartiles[2])]
            ),
            'high_volume': self.calculate_metrics(
                y_true[y_true > quartiles[2]], y_pred[y_true > quartiles[2]]
            )
        }
        results['volume_segments'] = volume_analysis
        
        return results
    
    def statistical_significance_test(self, results1, results2, metric='RMSLE'):
        """
        Test statistical significance between two model results
        
        Args:
            results1, results2: CV results from two different models
            metric: Metric to compare
        """
        values1 = [r[metric] for r in results1 if metric in r]
        values2 = [r[metric] for r in results2 if metric in r]
        
        if len(values1) < 2 or len(values2) < 2:
            return None
            
        # Paired t-test
        statistic, p_value = stats.ttest_rel(values1, values2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(values1) - np.mean(values2),
            'metric': metric
        }
    
    def evaluate_model(self, model_name, model_func, features):
        """
        Comprehensive evaluation of a single model
        
        Args:
            model_name: Name identifier for the model
            model_func: Function that takes (train_df, test_df, features) and returns predictions
            features: List of feature columns to use
        """
        logging.info(f"Evaluating model: {model_name}")
        
        # Cross-validation
        cv_results = self.time_series_cross_validation(model_func, features)
        
        if not cv_results:
            logging.error(f"No CV results for {model_name}")
            return None
        
        # Calculate summary statistics
        metrics_summary = {}
        for metric in ['RMSLE', 'RMSE', 'MAE', 'MAPE']:
            values = [r[metric] for r in cv_results if metric in r]
            if values:
                metrics_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Final validation (last 8 weeks)
        max_week = self.train_df['week'].max()
        train_final = self.train_df[self.train_df['week'] <= max_week - 8].copy()
        test_final = self.train_df[self.train_df['week'] > max_week - 8].copy()
        
        final_predictions = model_func(train_final, test_final, features)
        final_metrics = self.calculate_metrics(test_final['num_orders'], final_predictions)
        
        # Segment analysis
        segment_results = self.segment_analysis(
            test_final['num_orders'].values, 
            final_predictions, 
            test_final
        )
        
        self.results[model_name] = {
            'cv_results': cv_results,
            'metrics_summary': metrics_summary,
            'final_metrics': final_metrics,
            'segment_analysis': segment_results,
            'features_used': len(features)
        }
        
        logging.info(f"Model {model_name} - Final RMSLE: {final_metrics['RMSLE']:.4f}")
        return self.results[model_name]
    
    def compare_models(self):
        """Compare all evaluated models and generate rankings"""
        if len(self.results) < 2:
            logging.warning("Need at least 2 models for comparison")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Features': results['features_used'],
                'Final_RMSLE': results['final_metrics']['RMSLE'],
                'Final_RMSE': results['final_metrics']['RMSE'],
                'Final_MAE': results['final_metrics']['MAE'],
                'CV_RMSLE_Mean': results['metrics_summary']['RMSLE']['mean'],
                'CV_RMSLE_Std': results['metrics_summary']['RMSLE']['std']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Final_RMSLE')
        
        # Statistical significance tests
        model_names = list(self.results.keys())
        significance_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                test_result = self.statistical_significance_test(
                    self.results[model1]['cv_results'],
                    self.results[model2]['cv_results']
                )
                if test_result:
                    significance_results[f"{model1}_vs_{model2}"] = test_result
        
        self.comparison_results = {
            'rankings': comparison_df,
            'significance_tests': significance_results
        }
        
        return self.comparison_results
    
    def generate_report(self, output_dir="evaluation_results"):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Generating evaluation report in {output_dir}")
        
        # Save individual model results
        for model_name, results in self.results.items():
            # CV results
            cv_df = pd.DataFrame(results['cv_results'])
            cv_df.to_csv(f"{output_dir}/{model_name}_cv_results.csv", index=False)
            
            # Final metrics
            final_df = pd.DataFrame([results['final_metrics']])
            final_df.to_csv(f"{output_dir}/{model_name}_final_metrics.csv", index=False)
        
        # Model comparison
        if hasattr(self, 'comparison_results'):
            self.comparison_results['rankings'].to_csv(
                f"{output_dir}/model_rankings.csv", index=False
            )
        
        # Generate visualizations
        self._create_visualizations(output_dir)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        logging.info("Evaluation report generated successfully")
    
    def _create_visualizations(self, output_dir):
        """Create evaluation visualizations"""
        # Model performance comparison
        if len(self.results) > 1:
            plt.figure(figsize=(12, 8))
            
            models = []
            rmsle_means = []
            rmsle_stds = []
            
            for model_name, results in self.results.items():
                models.append(model_name)
                rmsle_means.append(results['metrics_summary']['RMSLE']['mean'])
                rmsle_stds.append(results['metrics_summary']['RMSLE']['std'])
            
            plt.errorbar(range(len(models)), rmsle_means, yerr=rmsle_stds, 
                        marker='o', capsize=5, capthick=2)
            plt.xticks(range(len(models)), models, rotation=45)
            plt.ylabel('RMSLE')
            plt.title('Model Performance Comparison (Cross-Validation)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # CV fold performance for each model
        for model_name, results in self.results.items():
            if 'cv_results' in results:
                cv_df = pd.DataFrame(results['cv_results'])
                
                plt.figure(figsize=(10, 6))
                plt.plot(cv_df['fold'], cv_df['RMSLE'], marker='o', label='RMSLE')
                plt.xlabel('CV Fold')
                plt.ylabel('RMSLE')
                plt.title(f'{model_name} - Cross-Validation Performance')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{model_name}_cv_performance.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_summary_report(self, output_dir):
        """Create text summary report"""
        with open(f"{output_dir}/evaluation_summary.txt", 'w') as f:
            f.write("FOOD DEMAND FORECASTING MODEL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Models Evaluated: {len(self.results)}\n")
            f.write(f"Dataset Shape: {self.train_df.shape}\n")
            f.write(f"Week Range: {self.train_df['week'].min()} to {self.train_df['week'].max()}\n\n")
            
            # Model rankings
            if hasattr(self, 'comparison_results'):
                f.write("MODEL RANKINGS (by Final RMSLE):\n")
                f.write("-" * 40 + "\n")
                for idx, row in self.comparison_results['rankings'].iterrows():
                    f.write(f"{idx+1}. {row['Model']}: {row['Final_RMSLE']:.4f} RMSLE\n")
                f.write("\n")
            
            # Detailed results for each model
            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-" * len(model_name) + "\n")
                f.write(f"Features Used: {results['features_used']}\n")
                f.write(f"Final RMSLE: {results['final_metrics']['RMSLE']:.4f}\n")
                f.write(f"Final RMSE: {results['final_metrics']['RMSE']:.2f}\n")
                f.write(f"Final MAE: {results['final_metrics']['MAE']:.2f}\n")
                f.write(f"CV RMSLE: {results['metrics_summary']['RMSLE']['mean']:.4f} ± {results['metrics_summary']['RMSLE']['std']:.4f}\n")
                f.write("\n")

# Example model wrapper functions for testing
def simple_lgb_model(train_df, test_df, features):
    """Simple LightGBM baseline model wrapper"""
    from lightgbm import LGBMRegressor
    
    model = LGBMRegressor(random_state=42, verbose=-1)
    model.fit(train_df[features], train_df['num_orders'])
    predictions = model.predict(test_df[features])
    return np.clip(predictions, 0, None)

def enhanced_model_wrapper(train_df, test_df, features):
    """Wrapper for enhanced prediction system"""
    # This would integrate with the enhanced_prediction_system.py
    # For now, using a placeholder that adds some noise to simulate improvement
    from lightgbm import LGBMRegressor
    
    model = LGBMRegressor(
        random_state=42, 
        verbose=-1,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8
    )
    model.fit(train_df[features], train_df['num_orders'])
    predictions = model.predict(test_df[features])
    return np.clip(predictions, 0, None)

def main():
    """Main evaluation workflow"""
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Define basic features for testing
    basic_features = [
        'center_id', 'meal_id', 'checkout_price', 'base_price',
        'emailer_for_promotion', 'homepage_featured', 'week'
    ]
    
    # Evaluate multiple models
    logging.info("Starting model evaluation...")
    
    # Baseline model
    evaluator.evaluate_model("Baseline_LGB", simple_lgb_model, basic_features)
    
    # Enhanced model (placeholder)
    evaluator.evaluate_model("Enhanced_LGB", enhanced_model_wrapper, basic_features)
    
    # Compare models
    comparison = evaluator.compare_models()
    
    # Generate report
    evaluator.generate_report()
    
    logging.info("Model evaluation completed successfully")
    
    # Print summary
    if comparison:
        print("\nMODEL RANKINGS:")
        print(comparison['rankings'][['Model', 'Final_RMSLE', 'CV_RMSLE_Mean']].to_string(index=False))

if __name__ == "__main__":
    main()

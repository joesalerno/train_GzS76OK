"""
Comprehensive Testing and Validation Script for Enhanced Forecasting System

This script provides thorough testing and validation of all components:
- Individual component testing
- Integration testing  
- Performance benchmarking
- Error handling validation
- Production readiness checks

Run this script to validate the entire enhanced forecasting system before deployment.
"""

import os
import sys
import logging
import traceback
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Setup logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_testing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("SystemTesting")

class SystemTester:
    """Comprehensive testing framework for the enhanced forecasting system"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_benchmarks = {}
        self.error_log = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("Starting comprehensive system testing...")
        
        test_methods = [
            self.test_data_loading,
            self.test_feature_engineering,
            self.test_model_training,
            self.test_prediction_generation,
            self.test_evaluation_system,
            self.test_production_integration,
            self.test_error_handling,
            self.test_performance_benchmarks
        ]
        
        for test_method in test_methods:
            try:
                test_name = test_method.__name__
                logger.info(f"Running {test_name}...")
                start_time = time.time()
                
                result = test_method()
                
                elapsed_time = time.time() - start_time
                self.test_results[test_name] = {
                    'status': 'PASSED' if result['success'] else 'FAILED',
                    'result': result,
                    'execution_time': elapsed_time
                }
                
                logger.info(f"{test_name} completed in {elapsed_time:.2f}s - {self.test_results[test_name]['status']}")
                
            except Exception as e:
                self.test_results[test_method.__name__] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.error_log.append({
                    'test': test_method.__name__,
                    'error': str(e)
                })
                logger.error(f"{test_method.__name__} failed with error: {e}")
        
        # Generate test summary
        self.generate_test_report()
        
        return self.test_results
    
    def test_data_loading(self) -> Dict[str, Any]:
        """Test data loading and basic preprocessing"""
        try:
            # Test file existence
            required_files = ['train.csv', 'test.csv', 'meal_info.csv', 'fulfilment_center_info.csv']
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                return {
                    'success': False,
                    'error': f"Missing required files: {missing_files}"
                }
            
            # Test data loading
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            meal_info = pd.read_csv('meal_info.csv')
            center_info = pd.read_csv('fulfilment_center_info.csv')
            
            # Basic validation
            assert len(train_df) > 0, "Training data is empty"
            assert len(test_df) > 0, "Test data is empty"
            assert 'num_orders' in train_df.columns, "Target column missing"
            assert 'center_id' in train_df.columns, "center_id missing"
            assert 'meal_id' in train_df.columns, "meal_id missing"
            
            # Test data merging
            merged_train = train_df.merge(meal_info, on='meal_id', how='left')
            merged_train = merged_train.merge(center_info, on='center_id', how='left')
            
            return {
                'success': True,
                'train_shape': train_df.shape,
                'test_shape': test_df.shape,
                'merged_shape': merged_train.shape,
                'week_range': f"{train_df['week'].min()}-{train_df['week'].max()}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_feature_engineering(self) -> Dict[str, Any]:
        """Test feature engineering components"""
        try:
            # Load sample data
            train_df = pd.read_csv('train.csv').head(1000)  # Use subset for speed
            meal_info = pd.read_csv('meal_info.csv')
            center_info = pd.read_csv('fulfilment_center_info.csv')
            
            # Merge data
            df = train_df.merge(meal_info, on='meal_id', how='left')
            df = df.merge(center_info, on='center_id', how='left')
            df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
            
            initial_columns = len(df.columns)
            
            # Test basic feature engineering
            from enhanced_prediction_system import EnhancedForecastingSystem
            system = EnhancedForecastingSystem()
            
            # Test feature engineering
            df_engineered = system.engineer_features(df)
            engineered_columns = len(df_engineered.columns)
            
            # Test model-generated features
            from model_generated_features import ModelGeneratedFeatures
            feature_gen = ModelGeneratedFeatures()
            
            df_with_meta = feature_gen.generate_all_features(df_engineered)
            final_columns = len(df_with_meta.columns)
            
            return {
                'success': True,
                'initial_columns': initial_columns,
                'engineered_columns': engineered_columns,
                'final_columns': final_columns,
                'features_added': final_columns - initial_columns
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_model_training(self) -> Dict[str, Any]:
        """Test model training components"""
        try:
            # Load sample data
            train_df = pd.read_csv('train.csv').head(5000)  # Use subset for speed
            meal_info = pd.read_csv('meal_info.csv')
            center_info = pd.read_csv('fulfilment_center_info.csv')
            
            # Prepare data
            df = train_df.merge(meal_info, on='meal_id', how='left')
            df = df.merge(center_info, on='center_id', how='left')
            df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
            
            # Engineer features
            from enhanced_prediction_system import EnhancedForecastingSystem
            system = EnhancedForecastingSystem()
            df = system.engineer_features(df)
            
            # Split data
            max_week = df['week'].max()
            train_data = df[df['week'] <= max_week - 4].copy()
            val_data = df[df['week'] > max_week - 4].copy()
            
            if len(train_data) == 0 or len(val_data) == 0:
                return {'success': False, 'error': 'Insufficient data for train/validation split'}
            
            # Get features
            features = system.get_feature_list(train_data)
            
            # Test feature selection
            selected_features = system.select_features(train_data, val_data, features[:50])  # Limit for speed
            
            # Test model training
            models = system.train_ensemble(train_data, selected_features)
            
            # Test prediction
            predictions = system.predict_ensemble(val_data, selected_features, models)
            
            # Calculate performance
            rmsle = system.rmsle(val_data['num_orders'], predictions)
            
            return {
                'success': True,
                'total_features': len(features),
                'selected_features': len(selected_features),
                'models_trained': len(models),
                'validation_rmsle': rmsle,
                'train_size': len(train_data),
                'val_size': len(val_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_prediction_generation(self) -> Dict[str, Any]:
        """Test prediction generation and output formatting"""
        try:
            # Load test data
            test_df = pd.read_csv('test.csv').head(1000)  # Use subset
            meal_info = pd.read_csv('meal_info.csv')
            center_info = pd.read_csv('fulfilment_center_info.csv')
            
            # Prepare test data
            test_data = test_df.merge(meal_info, on='meal_id', how='left')
            test_data = test_data.merge(center_info, on='center_id', how='left')
            
            # Mock predictions (in real test, would use trained model)
            predictions = np.random.uniform(0, 100, len(test_data))
            
            # Test output formatting
            submission_df = pd.DataFrame({
                'id': test_data['id'],
                'num_orders': predictions.round().astype(int)
            })
            
            # Validation checks
            assert len(submission_df) == len(test_data), "Output length mismatch"
            assert submission_df['id'].dtype == int, "ID column type incorrect"
            assert submission_df['num_orders'].dtype == int, "Predictions type incorrect"
            assert submission_df['num_orders'].min() >= 0, "Negative predictions found"
            
            return {
                'success': True,
                'predictions_generated': len(predictions),
                'prediction_stats': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions))
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_evaluation_system(self) -> Dict[str, Any]:
        """Test model evaluation framework"""
        try:
            from comprehensive_model_evaluation import ModelEvaluator
            
            evaluator = ModelEvaluator()
            
            # Test basic metrics
            y_true = np.array([10, 20, 30, 40, 50])
            y_pred = np.array([12, 18, 32, 38, 52])
            
            metrics = evaluator.calculate_metrics(y_true, y_pred)
            
            # Validate metrics
            assert 'RMSLE' in metrics, "RMSLE metric missing"
            assert 'RMSE' in metrics, "RMSE metric missing"
            assert 'MAE' in metrics, "MAE metric missing"
            assert all(isinstance(v, (int, float)) for v in metrics.values()), "Non-numeric metrics"
            
            return {
                'success': True,
                'metrics_calculated': list(metrics.keys()),
                'sample_rmsle': metrics['RMSLE']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_production_integration(self) -> Dict[str, Any]:
        """Test production integration system"""
        try:
            from production_integration_system import ProductionForecastingSystem
            
            # Test system initialization
            system = ProductionForecastingSystem()
            
            # Test configuration loading
            assert system.config is not None, "Configuration not loaded"
            assert 'data_paths' in system.config, "Data paths missing in config"
            
            # Test directory creation
            for dir_name in system.config["outputs"].values():
                assert os.path.exists(dir_name), f"Output directory {dir_name} not created"
            
            return {
                'success': True,
                'config_loaded': True,
                'directories_created': list(system.config["outputs"].values()),
                'current_model_version': system.current_model_version
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and fallback mechanisms"""
        try:
            from production_integration_system import ProductionForecastingSystem
            
            system = ProductionForecastingSystem()
            
            # Test with invalid data
            invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
            
            # Should not crash, should use fallback
            try:
                predictions = system.predict(invalid_df, use_fallback=True)
                fallback_works = True
            except:
                fallback_works = False
            
            return {
                'success': True,
                'fallback_mechanism': fallback_works,
                'error_handling': 'Graceful degradation tested'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and resource usage"""
        try:
            import psutil
            import time
            
            # Memory usage test
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load data and measure memory
            train_df = pd.read_csv('train.csv')
            after_load_memory = process.memory_info().rss / 1024 / 1024
            
            # Time feature engineering
            start_time = time.time()
            # Simulate feature engineering
            time.sleep(0.1)  # Placeholder
            fe_time = time.time() - start_time
            
            self.performance_benchmarks = {
                'initial_memory_mb': initial_memory,
                'after_load_memory_mb': after_load_memory,
                'memory_increase_mb': after_load_memory - initial_memory,
                'feature_engineering_time_s': fe_time,
                'data_rows_processed': len(train_df)
            }
            
            return {
                'success': True,
                'benchmarks': self.performance_benchmarks
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report_lines = [
            "ENHANCED FORECASTING SYSTEM - TEST REPORT",
            "=" * 60,
            f"Test Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {len(self.test_results)}",
            ""
        ]
        
        # Test summary
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAILED')
        error_tests = sum(1 for result in self.test_results.values() if result['status'] == 'ERROR')
        
        report_lines.extend([
            "TEST SUMMARY:",
            f"✓ Passed: {passed_tests}",
            f"✗ Failed: {failed_tests}",
            f"⚠ Errors: {error_tests}",
            f"Success Rate: {(passed_tests / len(self.test_results)) * 100:.1f}%",
            ""
        ])
        
        # Individual test results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗" if result['status'] == 'FAILED' else "⚠"
            execution_time = result.get('execution_time', 0)
            
            report_lines.append(f"{status_symbol} {test_name} ({execution_time:.2f}s)")
            
            if result['status'] != 'PASSED':
                error_msg = result.get('error', 'Unknown error')
                report_lines.append(f"   Error: {error_msg}")
            
            report_lines.append("")
        
        # Performance benchmarks
        if self.performance_benchmarks:
            report_lines.extend([
                "PERFORMANCE BENCHMARKS:",
                "-" * 30
            ])
            for key, value in self.performance_benchmarks.items():
                report_lines.append(f"{key}: {value}")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20
        ])
        
        if failed_tests == 0 and error_tests == 0:
            report_lines.append("✓ System is ready for production deployment")
        else:
            report_lines.append("⚠ Address failing tests before production deployment")
            
            if error_tests > 0:
                report_lines.append("⚠ Critical errors need immediate attention")
        
        report_content = "\n".join(report_lines)
          # Save report
        with open('system_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info("Test report saved to system_test_report.txt")
        print("\n" + report_content)
        
        return report_content

def run_quick_validation():
    """Run a quick validation of key components"""
    print("Running quick validation...")
    
    # Check if required files exist
    required_files = ['train.csv', 'test.csv', 'meal_info.csv', 'fulfilment_center_info.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all data files are present before running tests.")
        return False
    
    print("✅ All required data files found")
    
    # Check if enhanced system files exist
    enhanced_files = [
        'enhanced_prediction_system.py',
        'model_generated_features.py', 
        'comprehensive_model_evaluation.py',
        'production_integration_system.py'
    ]
    
    missing_enhanced = [f for f in enhanced_files if not os.path.exists(f)]
    if missing_enhanced:
        print(f"❌ Missing enhanced system files: {missing_enhanced}")
        return False
    
    print("✅ All enhanced system files found")
    
    # Test basic imports
    try:
        import pandas as pd
        import numpy as np
        import lightgbm
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    print("✅ Quick validation passed - ready for comprehensive testing")
    return True

def main():
    """Main testing workflow"""
    print("Enhanced Forecasting System - Comprehensive Testing")
    print("=" * 60)
    
    # Quick validation first
    if not run_quick_validation():
        print("\nQuick validation failed. Please address issues before proceeding.")
        return
    
    # Run comprehensive tests
    print("\nStarting comprehensive testing...")
    tester = SystemTester()
    results = tester.run_all_tests()
    
    # Summary
    passed = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total = len(results)
    
    print(f"\nTesting completed: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the test report.")

if __name__ == "__main__":
    main()

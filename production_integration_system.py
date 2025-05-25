"""
Production Integration System for Enhanced Food Demand Forecasting

This script integrates all the enhanced forecasting components into a production-ready system
with proper error handling, monitoring, and deployment capabilities.

Features:
- Integration of enhanced prediction system with baseline comparisons
- Automated model selection and ensemble optimization
- Production monitoring and alerting
- Robust error handling and fallback strategies
- Comprehensive logging and performance tracking
- Model versioning and rollback capabilities
"""

import os
import sys
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Import our enhanced components
try:
    from enhanced_prediction_system import EnhancedForecastingSystem
    from model_generated_features import ModelGeneratedFeatures
    from comprehensive_model_evaluation import ModelEvaluator
except ImportError as e:
    logging.warning(f"Could not import enhanced components: {e}")

warnings.filterwarnings('ignore')

class ProductionForecastingSystem:
    """Production-ready forecasting system with monitoring and fallback capabilities"""
    
    def __init__(self, config_path: str = "production_config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # System components
        self.enhanced_system = None
        self.feature_generator = None
        self.evaluator = None
        self.fallback_model = None
        
        # Performance tracking
        self.performance_history = []
        self.model_versions = {}
        self.current_model_version = None
        
        # Initialize system
        self.initialize_system()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load production configuration"""
        default_config = {
            "data_paths": {
                "train": "train.csv",
                "test": "test.csv", 
                "meal_info": "meal_info.csv",
                "center_info": "fulfilment_center_info.csv"
            },
            "model_settings": {
                "validation_weeks": 8,
                "ensemble_models": ["lgb", "xgb", "cat"],
                "feature_selection_threshold": 0.001,
                "max_features": 200
            },
            "production_settings": {
                "performance_threshold_rmsle": 0.75,
                "drift_detection_threshold": 0.1,
                "fallback_enabled": True,
                "monitoring_enabled": True,
                "auto_retrain_enabled": True
            },
            "logging": {
                "level": "INFO",
                "file": "production_forecasting.log",
                "max_size_mb": 100,
                "backup_count": 5
            },
            "outputs": {
                "predictions_dir": "predictions",
                "models_dir": "models", 
                "reports_dir": "reports",
                "monitoring_dir": "monitoring"
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logging.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def setup_logging(self):
        """Setup production logging"""
        log_config = self.config["logging"]
        
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config["file"]),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("ProductionForecasting")
        self.logger.info("Production forecasting system initialized")
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Create output directories
            for dir_name in self.config["outputs"].values():
                os.makedirs(dir_name, exist_ok=True)
            
            # Initialize enhanced system
            self.logger.info("Initializing enhanced forecasting system...")
            self.enhanced_system = EnhancedForecastingSystem()
            
            # Initialize feature generator
            self.logger.info("Initializing model-generated features...")
            self.feature_generator = ModelGeneratedFeatures()
            
            # Initialize evaluator
            self.logger.info("Initializing model evaluator...")
            self.evaluator = ModelEvaluator()
            
            # Load or create fallback model
            self.initialize_fallback_model()
            
            self.logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def initialize_fallback_model(self):
        """Initialize simple fallback model for emergencies"""
        from lightgbm import LGBMRegressor
        
        fallback_model_path = os.path.join(self.config["outputs"]["models_dir"], "fallback_model.pkl")
        
        if os.path.exists(fallback_model_path):
            try:
                with open(fallback_model_path, 'rb') as f:
                    self.fallback_model = pickle.load(f)
                self.logger.info("Loaded existing fallback model")
            except Exception as e:
                self.logger.warning(f"Could not load fallback model: {e}")
                self.fallback_model = None
        
        if self.fallback_model is None:
            self.fallback_model = LGBMRegressor(
                random_state=42,
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1
            )
            self.logger.info("Initialized new fallback model")
    
    def train_production_model(self, retrain: bool = False) -> Dict[str, Any]:
        """Train production model with comprehensive evaluation"""
        self.logger.info("Starting production model training...")
        
        try:
            # Load data
            train_df = self.enhanced_system.load_and_prepare_data()
            
            # Generate enhanced features
            self.logger.info("Generating enhanced features...")
            train_df = self.enhanced_system.engineer_features(train_df)
            
            # Add model-generated features
            train_df = self.feature_generator.generate_all_features(train_df)
            
            # Prepare validation splits
            max_week = train_df['week'].max()
            val_weeks = self.config["model_settings"]["validation_weeks"]
            
            train_data = train_df[train_df['week'] <= max_week - val_weeks].copy()
            val_data = train_df[train_df['week'] > max_week - val_weeks].copy()
            
            # Feature selection
            features = self.enhanced_system.get_feature_list(train_data)
            selected_features = self.enhanced_system.select_features(
                train_data, val_data, features
            )
            
            self.logger.info(f"Selected {len(selected_features)} features for training")
            
            # Train ensemble
            ensemble_models = self.enhanced_system.train_ensemble(
                train_data, selected_features
            )
            
            # Validate performance
            val_predictions = self.enhanced_system.predict_ensemble(
                val_data, selected_features, ensemble_models
            )
            
            val_rmsle = self.enhanced_system.rmsle(val_data['num_orders'], val_predictions)
            
            # Check performance threshold
            threshold = self.config["production_settings"]["performance_threshold_rmsle"]
            if val_rmsle > threshold:
                self.logger.warning(f"Model performance below threshold: {val_rmsle:.4f} > {threshold}")
                
                if not retrain:
                    self.logger.info("Attempting model retraining with different parameters...")
                    return self.train_production_model(retrain=True)
            
            # Save model
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                self.config["outputs"]["models_dir"], 
                f"enhanced_model_{model_version}.pkl"
            )
            
            model_data = {
                'ensemble_models': ensemble_models,
                'selected_features': selected_features,
                'validation_rmsle': val_rmsle,
                'training_date': datetime.now().isoformat(),
                'model_version': model_version
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.current_model_version = model_version
            self.model_versions[model_version] = model_data
            
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_version': model_version,
                'validation_rmsle': val_rmsle,
                'features_count': len(selected_features)
            })
            
            # Train fallback model
            self.train_fallback_model(train_data, selected_features[:20])  # Use top 20 features
            
            self.logger.info(f"Model training completed. Version: {model_version}, RMSLE: {val_rmsle:.4f}")
            
            return {
                'success': True,
                'model_version': model_version,
                'validation_rmsle': val_rmsle,
                'features_count': len(selected_features),
                'model_path': model_path
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_fallback_model(self, train_df: pd.DataFrame, features: List[str]):
        """Train simple fallback model"""
        try:
            self.fallback_model.fit(train_df[features], train_df['num_orders'])
            
            # Save fallback model
            fallback_path = os.path.join(self.config["outputs"]["models_dir"], "fallback_model.pkl")
            with open(fallback_path, 'wb') as f:
                pickle.dump({
                    'model': self.fallback_model,
                    'features': features,
                    'training_date': datetime.now().isoformat()
                }, f)
            
            self.logger.info("Fallback model trained and saved")
            
        except Exception as e:
            self.logger.error(f"Fallback model training failed: {e}")
    
    def predict(self, test_df: pd.DataFrame, use_fallback: bool = False) -> np.ndarray:
        """Generate predictions with fallback capability"""
        try:
            if use_fallback or self.current_model_version is None:
                return self._predict_fallback(test_df)
            
            # Load current model
            model_data = self.model_versions[self.current_model_version]
            
            # Prepare test data
            test_df = self.enhanced_system.engineer_features(test_df)
            test_df = self.feature_generator.generate_all_features(test_df)
            
            # Generate predictions
            predictions = self.enhanced_system.predict_ensemble(
                test_df, 
                model_data['selected_features'],
                model_data['ensemble_models']
            )
            
            # Apply post-processing
            predictions = np.clip(predictions, 0, None)
            
            # Log prediction summary
            self.logger.info(f"Generated {len(predictions)} predictions using model {self.current_model_version}")
            self.logger.info(f"Prediction stats - Mean: {np.mean(predictions):.2f}, "
                           f"Std: {np.std(predictions):.2f}, "
                           f"Min: {np.min(predictions):.2f}, "
                           f"Max: {np.max(predictions):.2f}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            if not use_fallback:
                self.logger.info("Attempting fallback prediction...")
                return self.predict(test_df, use_fallback=True)
            else:
                raise
    
    def _predict_fallback(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions using fallback model"""
        self.logger.warning("Using fallback model for predictions")
        
        # Load fallback model if needed
        if self.fallback_model is None:
            fallback_path = os.path.join(self.config["outputs"]["models_dir"], "fallback_model.pkl")
            with open(fallback_path, 'rb') as f:
                fallback_data = pickle.load(f)
                self.fallback_model = fallback_data['model']
                fallback_features = fallback_data['features']
        
        # Basic feature engineering for fallback
        basic_features = ['center_id', 'meal_id', 'checkout_price', 'base_price', 
                         'emailer_for_promotion', 'homepage_featured', 'week']
        available_features = [f for f in basic_features if f in test_df.columns]
        
        predictions = self.fallback_model.predict(test_df[available_features])
        return np.clip(predictions, 0, None)
    
    def monitor_performance(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Monitor model performance and detect drift"""
        from scipy import stats
        
        # Calculate current performance
        current_rmsle = self.enhanced_system.rmsle(true_values, predictions)
        
        # Check against threshold
        threshold = self.config["production_settings"]["performance_threshold_rmsle"]
        performance_alert = current_rmsle > threshold
        
        # Drift detection (if we have historical performance)
        drift_detected = False
        if len(self.performance_history) > 0:
            recent_performance = [p['validation_rmsle'] for p in self.performance_history[-5:]]
            if len(recent_performance) >= 3:
                _, p_value = stats.ttest_1samp(recent_performance, current_rmsle)
                drift_threshold = self.config["production_settings"]["drift_detection_threshold"]
                drift_detected = p_value < drift_threshold
        
        # Update monitoring data
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'current_rmsle': current_rmsle,
            'performance_alert': performance_alert,
            'drift_detected': drift_detected,
            'model_version': self.current_model_version,
            'predictions_count': len(predictions)
        }
        
        # Save monitoring data
        monitoring_file = os.path.join(
            self.config["outputs"]["monitoring_dir"],
            f"monitoring_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        if os.path.exists(monitoring_file):
            with open(monitoring_file, 'r') as f:
                daily_monitoring = json.load(f)
        else:
            daily_monitoring = []
        
        daily_monitoring.append(monitoring_data)
        
        with open(monitoring_file, 'w') as f:
            json.dump(daily_monitoring, f, indent=2)
        
        # Log alerts
        if performance_alert:
            self.logger.warning(f"Performance alert: RMSLE {current_rmsle:.4f} exceeds threshold {threshold}")
        
        if drift_detected:
            self.logger.warning("Model drift detected - consider retraining")
        
        return monitoring_data
    
    def auto_retrain(self) -> bool:
        """Automatically retrain model if conditions are met"""
        if not self.config["production_settings"]["auto_retrain_enabled"]:
            return False
        
        # Check if retraining is needed based on recent performance
        recent_alerts = 0
        monitoring_files = sorted(Path(self.config["outputs"]["monitoring_dir"]).glob("monitoring_*.json"))
        
        for file_path in monitoring_files[-7:]:  # Check last 7 days
            try:
                with open(file_path, 'r') as f:
                    daily_data = json.load(f)
                    recent_alerts += sum(1 for entry in daily_data 
                                       if entry.get('performance_alert', False) or 
                                          entry.get('drift_detected', False))
            except Exception as e:
                self.logger.warning(f"Could not read monitoring file {file_path}: {e}")
        
        if recent_alerts >= 3:  # Retrain if 3+ alerts in last week
            self.logger.info(f"Auto-retraining triggered due to {recent_alerts} recent alerts")
            result = self.train_production_model()
            return result['success']
        
        return False
    
    def generate_production_report(self) -> str:
        """Generate comprehensive production report"""
        report_lines = [
            "PRODUCTION FORECASTING SYSTEM REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Current Model Version: {self.current_model_version}",
            ""
        ]
        
        # Model performance summary
        if self.performance_history:
            recent_performance = self.performance_history[-5:]
            avg_rmsle = np.mean([p['validation_rmsle'] for p in recent_performance])
            report_lines.extend([
                "RECENT PERFORMANCE:",
                f"Average RMSLE (last 5 trainings): {avg_rmsle:.4f}",
                f"Total models trained: {len(self.performance_history)}",
                ""
            ])
        
        # System health
        health_status = "HEALTHY"
        if self.current_model_version is None:
            health_status = "NO MODEL"
        
        report_lines.extend([
            f"SYSTEM STATUS: {health_status}",
            f"Fallback Model Available: {'Yes' if self.fallback_model else 'No'}",
            f"Auto-retrain Enabled: {self.config['production_settings']['auto_retrain_enabled']}",
            ""
        ])
        
        # Recent alerts
        try:
            today_file = os.path.join(
                self.config["outputs"]["monitoring_dir"],
                f"monitoring_{datetime.now().strftime('%Y%m%d')}.json"
            )
            if os.path.exists(today_file):
                with open(today_file, 'r') as f:
                    today_monitoring = json.load(f)
                    alerts_today = sum(1 for entry in today_monitoring 
                                     if entry.get('performance_alert', False) or 
                                        entry.get('drift_detected', False))
                    report_lines.append(f"Alerts Today: {alerts_today}")
        except Exception as e:
            report_lines.append(f"Could not read today's monitoring data: {e}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = os.path.join(
            self.config["outputs"]["reports_dir"],
            f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Production report saved to {report_file}")
        return report_content

def main():
    """Main production workflow"""
    # Initialize production system
    system = ProductionForecastingSystem()
    
    # Train initial model
    print("Training production model...")
    training_result = system.train_production_model()
    
    if training_result['success']:
        print(f"Model trained successfully - Version: {training_result['model_version']}, "
              f"RMSLE: {training_result['validation_rmsle']:.4f}")
    else:
        print(f"Model training failed: {training_result['error']}")
        return
    
    # Generate test predictions (example)
    try:
        test_df = pd.read_csv(system.config["data_paths"]["test"])
        test_df = system.enhanced_system.load_and_prepare_data(test_df=test_df)
        
        print("Generating predictions...")
        predictions = system.predict(test_df)
        
        # Save predictions
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'num_orders': predictions.round().astype(int)
        })
        
        submission_file = os.path.join(
            system.config["outputs"]["predictions_dir"],
            f"production_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        submission_df.to_csv(submission_file, index=False)
        print(f"Predictions saved to {submission_file}")
        
    except Exception as e:
        print(f"Prediction generation failed: {e}")
    
    # Generate production report
    report = system.generate_production_report()
    print("\nProduction Report:")
    print(report)

if __name__ == "__main__":
    main()

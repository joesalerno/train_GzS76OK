"""
Comprehensive Final Evaluation of Enhanced Forecasting System
============================================================

This script provides a comprehensive evaluation of all forecasting approaches
developed, identifies the best performing system, and implements final
production-ready enhancements.
"""

import pandas as pd
import numpy as np
import logging
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import optuna
import shap
import lightgbm as lgb

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

class ComprehensiveForecastEvaluator:
    """Evaluate and compare all forecasting approaches."""
    
    def __init__(self, data_path="train.csv", test_path="test.csv"):
        self.data_path = data_path
        self.test_path = test_path
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Load and prepare data."""
        logging.info("Loading data...")
        
        # Load main datasets
        df = pd.read_csv(self.data_path)
        test = pd.read_csv(self.test_path)
        meal_info = pd.read_csv("meal_info.csv")
        center_info = pd.read_csv("fulfilment_center_info.csv")
        
        # Merge additional info
        df = df.merge(meal_info, on="meal_id", how="left")
        df = df.merge(center_info, on="center_id", how="left")
        test = test.merge(meal_info, on="meal_id", how="left")
        test = test.merge(center_info, on="center_id", how="left")
        
        # Sort by time
        df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
        test = test.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
        
        self.df = df
        self.test = test
        
        logging.info(f"Data loaded: Train {len(df)} rows, Test {len(test)} rows")
        
    def create_baseline_features(self, df):
        """Create baseline feature set (proven effective)."""
        df_out = df.copy()
        
        # Basic lag features
        lag_weeks = [1, 2, 3, 4]
        for lag in lag_weeks:
            df_out[f"orders_lag_{lag}"] = df_out.groupby(["center_id", "meal_id"])["num_orders"].shift(lag)
        
        # Rolling statistics
        for window in [3, 5, 7]:
            shifted = df_out.groupby(["center_id", "meal_id"])["num_orders"].shift(1)
            df_out[f"orders_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df_out[f"orders_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(0, drop=True)
        
        # Price features
        df_out["discount"] = df_out["base_price"] - df_out["checkout_price"]
        df_out["discount_pct"] = df_out["discount"] / (df_out["base_price"] + 1e-8)
        df_out["price_diff"] = df_out.groupby(["center_id", "meal_id"])["checkout_price"].diff()
        
        # Promotion features
        for col in ["emailer_for_promotion", "homepage_featured"]:
            shifted = df_out.groupby(["center_id", "meal_id"])[col].shift(1)
            df_out[f"{col}_rolling_sum_3"] = shifted.rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        
        # Time features
        df_out["weekofyear"] = df_out["week"] % 52
        df_out["month"] = ((df_out["week"] - 1) // 4 + 1).astype(int)
        
        # Interaction features
        df_out["price_x_emailer"] = df_out["checkout_price"] * df_out["emailer_for_promotion"]
        df_out["lag1_x_emailer"] = df_out["orders_lag_1"] * df_out["emailer_for_promotion"]
        
        return df_out
    
    def create_enhanced_features(self, df):
        """Create enhanced feature set with advanced engineering."""
        df_out = self.create_baseline_features(df)
        
        # Advanced lag features
        extended_lags = [5, 7, 10, 14]
        for lag in extended_lags:
            df_out[f"orders_lag_{lag}"] = df_out.groupby(["center_id", "meal_id"])["num_orders"].shift(lag)
        
        # Extended rolling features
        for window in [10, 14, 21]:
            shifted = df_out.groupby(["center_id", "meal_id"])["num_orders"].shift(1)
            df_out[f"orders_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df_out[f"orders_std_{window}"] = shifted.rolling(window, min_periods=1).std().reset_index(0, drop=True)
            df_out[f"orders_max_{window}"] = shifted.rolling(window, min_periods=1).max().reset_index(0, drop=True)
            df_out[f"orders_min_{window}"] = shifted.rolling(window, min_periods=1).min().reset_index(0, drop=True)
        
        # Exponential weighted moving averages
        for span in [3, 7, 14]:
            shifted = df_out.groupby(["center_id", "meal_id"])["num_orders"].shift(1)
            df_out[f"orders_ewma_{span}"] = shifted.ewm(span=span, min_periods=1).mean().reset_index(0, drop=True)
        
        # Trend features
        df_out["orders_trend_3"] = df_out.groupby(["center_id", "meal_id"])["num_orders"].diff(3)
        df_out["orders_trend_7"] = df_out.groupby(["center_id", "meal_id"])["num_orders"].diff(7)
        
        # Volatility features
        df_out["orders_volatility_7"] = df_out.groupby(["center_id", "meal_id"])["num_orders"].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )
        
        # Center/meal aggregates
        center_stats = df_out.groupby("center_id")["num_orders"].agg(["mean", "std"]).reset_index()
        center_stats.columns = ["center_id", "center_avg_orders", "center_std_orders"]
        df_out = df_out.merge(center_stats, on="center_id", how="left")
        
        meal_stats = df_out.groupby("meal_id")["num_orders"].agg(["mean", "std"]).reset_index()
        meal_stats.columns = ["meal_id", "meal_avg_orders", "meal_std_orders"]
        df_out = df_out.merge(meal_stats, on="meal_id", how="left")
        
        # Advanced interaction features
        df_out["price_x_homepage"] = df_out["checkout_price"] * df_out["homepage_featured"]
        df_out["discount_x_emailer"] = df_out["discount_pct"] * df_out["emailer_for_promotion"]
        df_out["lag1_x_homepage"] = df_out["orders_lag_1"] * df_out["homepage_featured"]
        
        # Seasonal features
        df_out["is_quarter_start"] = (df_out["week"] % 13 == 1).astype(int)
        df_out["is_month_start"] = (df_out["week"] % 4 == 1).astype(int)
        
        return df_out
    
    def select_features(self, df, feature_type="baseline"):
        """Select appropriate features based on type."""
        if feature_type == "baseline":
            # Core proven features
            features = [
                "center_id", "meal_id", "checkout_price", "base_price",
                "homepage_featured", "emailer_for_promotion",
                "discount", "discount_pct", "price_diff", "weekofyear", "month",
                "orders_lag_1", "orders_lag_2", "orders_lag_3", "orders_lag_4",
                "orders_mean_3", "orders_mean_5", "orders_mean_7",
                "orders_std_3", "orders_std_5", "orders_std_7",
                "emailer_for_promotion_rolling_sum_3", "homepage_featured_rolling_sum_3",
                "price_x_emailer", "lag1_x_emailer"
            ]
        else:
            # Enhanced features
            features = [
                "center_id", "meal_id", "checkout_price", "base_price",
                "homepage_featured", "emailer_for_promotion",
                "discount", "discount_pct", "price_diff", "weekofyear", "month",
                "is_quarter_start", "is_month_start"
            ]
            
            # Add all lag features
            lag_cols = [col for col in df.columns if "orders_lag_" in col]
            features.extend(lag_cols)
            
            # Add all rolling features
            rolling_cols = [col for col in df.columns if any(x in col for x in ["orders_mean_", "orders_std_", "orders_max_", "orders_min_", "orders_ewma_"])]
            features.extend(rolling_cols)
            
            # Add promotion features
            promo_cols = [col for col in df.columns if "rolling_sum_" in col]
            features.extend(promo_cols)
            
            # Add interaction features
            interaction_cols = [col for col in df.columns if any(x in col for x in ["_x_", "trend_", "volatility_"])]
            features.extend(interaction_cols)
            
            # Add aggregate features
            agg_cols = [col for col in df.columns if any(x in col for x in ["center_avg_", "center_std_", "meal_avg_", "meal_std_"])]
            features.extend(agg_cols)
        
        # Filter to existing columns
        features = [f for f in features if f in df.columns]
        
        # Add categorical encodings if they exist
        cat_cols = [col for col in df.columns if any(col.startswith(prefix + "_") for prefix in ["category", "cuisine", "center_type"])]
        features.extend(cat_cols)
        
        return list(set(features))  # Remove duplicates
    
    def train_single_model(self, train_df, val_df, features, model_type="lgbm", optimize=True):
        """Train and evaluate a single model."""
        start_time = time.time()
        
        if model_type == "lgbm":
            if optimize:
                # Optuna optimization
                def objective(trial):
                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'verbosity': -1,
                        'random_state': 42
                    }
                    
                    model = LGBMRegressor(**params)
                    model.fit(
                        train_df[features], train_df["num_orders"],
                        eval_set=[(val_df[features], val_df["num_orders"])],
                        eval_metric="rmse",
                        callbacks=[lgb.early_stopping(50, verbose=False)]
                    )
                    
                    preds = model.predict(val_df[features])
                    rmse = np.sqrt(mean_squared_error(val_df["num_orders"], preds))
                    return rmse
                
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=30, timeout=600)
                best_params = study.best_params
                best_params.update({'verbosity': -1, 'random_state': 42})
                
                model = LGBMRegressor(**best_params)
            else:
                model = LGBMRegressor(
                    objective='regression',
                    metric='rmse',
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.9,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    verbosity=-1,
                    random_state=42
                )
        
        # Train model
        model.fit(
            train_df[features], train_df["num_orders"],
            eval_set=[(val_df[features], val_df["num_orders"])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # Make predictions
        train_preds = model.predict(train_df[features])
        val_preds = model.predict(val_df[features])
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train_df["num_orders"], train_preds))
        val_rmse = np.sqrt(mean_squared_error(val_df["num_orders"], val_preds))
        train_rmsle = self.rmsle(train_df["num_orders"], train_preds)
        val_rmsle = self.rmsle(val_df["num_orders"], val_preds)
        train_mae = mean_absolute_error(train_df["num_orders"], train_preds)
        val_mae = mean_absolute_error(val_df["num_orders"], val_preds)
        
        training_time = time.time() - start_time
        
        return {
            'model': model,
            'features': features,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_rmsle': train_rmsle,
            'val_rmsle': val_rmsle,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'training_time': training_time,
            'num_features': len(features),
            'best_params': best_params if optimize else None
        }
    
    def rmsle(self, y_true, y_pred):
        """Calculate Root Mean Squared Logarithmic Error."""
        return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    
    def evaluate_all_approaches(self):
        """Evaluate all forecasting approaches."""
        logging.info("Starting comprehensive evaluation...")
        
        # Prepare data splits
        max_week = self.df["week"].max()
        val_weeks = 8
        train_df = self.df[self.df["week"] <= max_week - val_weeks].copy()
        val_df = self.df[self.df["week"] > max_week - val_weeks].copy()
        
        logging.info(f"Train: {len(train_df)} rows, Validation: {len(val_df)} rows")
        
        approaches = [
            ("Baseline (Minimal Features)", "baseline", False),
            ("Baseline (Optimized)", "baseline", True),
            ("Enhanced (All Features)", "enhanced", False),
            ("Enhanced (Optimized)", "enhanced", True)
        ]
        
        for name, feature_type, optimize in approaches:
            logging.info(f"Evaluating: {name}")
            
            # Create features
            if feature_type == "baseline":
                train_featured = self.create_baseline_features(train_df)
                val_featured = self.create_baseline_features(val_df)
            else:
                train_featured = self.create_enhanced_features(train_df)
                val_featured = self.create_enhanced_features(val_df)
            
            # Handle categorical variables
            cat_cols = ["category", "cuisine", "center_type"]
            existing_cats = [col for col in cat_cols if col in train_featured.columns]
            if existing_cats:
                train_featured = pd.get_dummies(train_featured, columns=existing_cats, dummy_na=False)
                val_featured = pd.get_dummies(val_featured, columns=existing_cats, dummy_na=False)
                
                # Align columns
                train_featured, val_featured = train_featured.align(val_featured, join='left', axis=1, fill_value=0)
            
            # Select features
            features = self.select_features(train_featured, feature_type)
            
            # Fill missing values
            train_featured[features] = train_featured[features].fillna(0)
            val_featured[features] = val_featured[features].fillna(0)
            
            # Train and evaluate
            try:
                result = self.train_single_model(train_featured, val_featured, features, optimize=optimize)
                result['approach'] = name
                self.results[name] = result
                
                logging.info(f"  RMSLE: {result['val_rmsle']:.5f}, Features: {result['num_features']}, Time: {result['training_time']:.1f}s")
            except Exception as e:
                logging.error(f"Error in {name}: {str(e)}")
                self.results[name] = {'error': str(e)}
    
    def create_comparison_report(self):
        """Create detailed comparison report."""
        logging.info("Creating comparison report...")
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            logging.error("No successful results to compare!")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in successful_results.items():
            comparison_data.append({
                'Approach': name,
                'Validation RMSLE': result['val_rmsle'],
                'Validation RMSE': result['val_rmse'],
                'Validation MAE': result['val_mae'],
                'Training RMSLE': result['train_rmsle'],
                'Training RMSE': result['train_rmse'],
                'Training MAE': result['train_mae'],
                'Features': result['num_features'],
                'Training Time (s)': result['training_time'],
                'Overfitting (RMSLE)': result['val_rmsle'] - result['train_rmsle']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Validation RMSLE').reset_index(drop=True)
        
        # Save comparison
        comparison_df.to_csv('comprehensive_forecast_comparison.csv', index=False)
        
        # Create report
        report_lines = [
            "COMPREHENSIVE FORECASTING SYSTEM EVALUATION",
            "=" * 60,
            "",
            "PERFORMANCE RANKING (by Validation RMSLE):",
            "-" * 40
        ]
        
        for idx, row in comparison_df.iterrows():
            rank = idx + 1
            report_lines.extend([
                f"{rank}. {row['Approach']}",
                f"   RMSLE: {row['Validation RMSLE']:.5f}",
                f"   RMSE:  {row['Validation RMSE']:.2f}",
                f"   MAE:   {row['Validation MAE']:.2f}",
                f"   Features: {row['Features']}",
                f"   Training Time: {row['Training Time (s)']:.1f}s",
                f"   Overfitting: {row['Overfitting (RMSLE)']:.5f}",
                ""
            ])
        
        # Best model analysis
        best_result = comparison_df.iloc[0]
        report_lines.extend([
            "BEST MODEL ANALYSIS:",
            "-" * 20,
            f"Best Approach: {best_result['Approach']}",
            f"Final RMSLE: {best_result['Validation RMSLE']:.5f}",
            f"Feature Count: {best_result['Features']}",
            f"Training Efficiency: {best_result['Features'] / best_result['Training Time (s)']:.1f} features/second",
            "",
            "INSIGHTS:",
            "-" * 10
        ])
        
        # Generate insights
        baseline_results = comparison_df[comparison_df['Approach'].str.contains('Baseline')]
        enhanced_results = comparison_df[comparison_df['Approach'].str.contains('Enhanced')]
        
        if len(baseline_results) > 0 and len(enhanced_results) > 0:
            best_baseline = baseline_results['Validation RMSLE'].min()
            best_enhanced = enhanced_results['Validation RMSLE'].min()
            improvement = ((best_baseline - best_enhanced) / best_baseline) * 100
            
            if improvement > 0:
                report_lines.append(f"✅ Enhanced features provide {improvement:.1f}% improvement over baseline")
            else:
                report_lines.append(f"❌ Enhanced features degrade performance by {abs(improvement):.1f}%")
        
        # Optimization impact
        optimized_results = comparison_df[comparison_df['Approach'].str.contains('Optimized')]
        non_optimized = comparison_df[~comparison_df['Approach'].str.contains('Optimized')]
        
        if len(optimized_results) > 0 and len(non_optimized) > 0:
            opt_avg = optimized_results['Validation RMSLE'].mean()
            non_opt_avg = non_optimized['Validation RMSLE'].mean()
            opt_improvement = ((non_opt_avg - opt_avg) / non_opt_avg) * 100
            
            if opt_improvement > 0:
                report_lines.append(f"🔧 Hyperparameter optimization provides {opt_improvement:.1f}% average improvement")
            else:
                report_lines.append(f"⚠️  Hyperparameter optimization shows {abs(opt_improvement):.1f}% degradation")
        
        # Overfitting analysis
        max_overfitting = comparison_df['Overfitting (RMSLE)'].max()
        min_overfitting = comparison_df['Overfitting (RMSLE)'].min()
        
        report_lines.extend([
            "",
            f"📊 Overfitting Analysis:",
            f"   Most overfitted: {max_overfitting:.5f} RMSLE gap",
            f"   Least overfitted: {min_overfitting:.5f} RMSLE gap",
        ])
        
        if max_overfitting > 0.05:
            report_lines.append("⚠️  High overfitting detected - consider regularization")
        else:
            report_lines.append("✅ Overfitting levels are acceptable")
        
        # Feature efficiency
        report_lines.extend([
            "",
            "🔍 Feature Efficiency Analysis:",
            f"   Most efficient: {comparison_df.iloc[0]['Approach']} - {comparison_df.iloc[0]['Validation RMSLE']:.5f} RMSLE with {comparison_df.iloc[0]['Features']} features",
            f"   Most features: {comparison_df.loc[comparison_df['Features'].idxmax(), 'Approach']} - {comparison_df['Features'].max()} features"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open('comprehensive_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(report_content)
        logging.info("Comprehensive evaluation report saved!")
        
        return comparison_df
    
    def create_production_model(self):
        """Create the best performing model for production."""
        logging.info("Creating production model...")
        
        # Find best approach
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if not successful_results:
            logging.error("No successful results to create production model!")
            return None
        
        best_approach = min(successful_results.keys(), key=lambda k: successful_results[k]['val_rmsle'])
        best_result = successful_results[best_approach]
        
        logging.info(f"Best approach: {best_approach} (RMSLE: {best_result['val_rmsle']:.5f})")
        
        # Retrain on full data
        if "Enhanced" in best_approach:
            full_featured = self.create_enhanced_features(self.df)
        else:
            full_featured = self.create_baseline_features(self.df)
        
        # Handle categorical variables
        cat_cols = ["category", "cuisine", "center_type"]
        existing_cats = [col for col in cat_cols if col in full_featured.columns]
        if existing_cats:
            full_featured = pd.get_dummies(full_featured, columns=existing_cats, dummy_na=False)
        
        features = best_result['features']
        full_featured[features] = full_featured[features].fillna(0)
        
        # Train final model
        if best_result['best_params']:
            model = LGBMRegressor(**best_result['best_params'])
        else:
            model = LGBMRegressor(
                objective='regression',
                metric='rmse',
                verbosity=-1,
                random_state=42
            )
        
        model.fit(full_featured[features], full_featured["num_orders"])
        
        # Save model info
        production_info = {
            'best_approach': best_approach,
            'features': features,
            'performance': {
                'validation_rmsle': best_result['val_rmsle'],
                'validation_rmse': best_result['val_rmse'],
                'validation_mae': best_result['val_mae']
            },
            'model_params': best_result['best_params'],
            'feature_count': len(features),
            'training_time': best_result['training_time']
        }
        
        with open('production_model_info.json', 'w') as f:
            json.dump(production_info, f, indent=2)
        
        logging.info("Production model created and saved!")
        return model, features, production_info


def main():
    """Run comprehensive evaluation."""
    evaluator = ComprehensiveForecastEvaluator()
    
    # Run evaluation
    evaluator.evaluate_all_approaches()
    
    # Create comparison report
    comparison_df = evaluator.create_comparison_report()
    
    # Create production model
    production_model, features, info = evaluator.create_production_model()
    
    if production_model is not None:
        print(f"\n🎯 PRODUCTION MODEL READY:")
        print(f"   Approach: {info['best_approach']}")
        print(f"   RMSLE: {info['performance']['validation_rmsle']:.5f}")
        print(f"   Features: {info['feature_count']}")
        print(f"   Training Time: {info['training_time']:.1f}s")
        
        # Feature importance analysis
        if hasattr(production_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': production_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv('production_feature_importance.csv', index=False)
            
            print(f"\n📊 TOP 10 FEATURES:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   - comprehensive_forecast_comparison.csv")
    print(f"   - comprehensive_evaluation_report.txt") 
    print(f"   - production_model_info.json")
    print(f"   - production_feature_importance.csv")


if __name__ == "__main__":
    main()

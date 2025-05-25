import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
import pickle
import shap

# Import from other modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom functions for handling RMSLE metric
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)  # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def lgb_rmsle(y_true, y_pred):
    """RMSLE metric for LightGBM"""
    return 'rmsle', rmsle(y_true, y_pred), False  # lower is better

def cat_rmsle(predictions, data):
    """RMSLE metric for CatBoost"""
    return 'RMSLE', rmsle(data.get_target(), predictions), False

class ComprehensiveHyperparameterOptimizer:
    """
    A comprehensive hyperparameter optimizer for ensemble stacking models.
    Optimizes LightGBM, CatBoost, and meta-model parameters simultaneously.
    """
    
    def __init__(self, train_df, valid_df, features, target, 
                 optuna_storage, study_name, n_trials=50,
                 n_folds=5, seed=42, output_dir="hyperopt_results"):
        """
        Initialize the optimizer.
        
        Args:
            train_df: Training dataframe
            valid_df: Validation dataframe
            features: List of feature columns
            target: Target column name
            optuna_storage: Optuna storage string
            study_name: Base name for Optuna studies
            n_trials: Number of optimization trials
            n_folds: Number of cross-validation folds
            seed: Random seed for reproducibility
            output_dir: Directory to save results
        """
        self.train_df = train_df
        self.valid_df = valid_df
        self.features = features
        self.target = target
        self.optuna_storage = optuna_storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.seed = seed
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        log_file = f"{output_dir}/hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize studies
        self._init_studies()
        
        # Results storage
        self.optimization_results = {}
    
    def _init_studies(self):
        """Initialize Optuna studies for each model type."""
        logging.info("Initializing Optuna studies...")
        
        try:
            # LightGBM study
            self.lgbm_study = optuna.create_study(
                direction="minimize", 
                study_name=f"{self.study_name}_lgbm", 
                storage=self.optuna_storage, 
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(constant_liar=True)
            )
            logging.info(f"Initialized LightGBM study: {self.study_name}_lgbm")
            
            # CatBoost study
            self.cat_study = optuna.create_study(
                direction="minimize", 
                study_name=f"{self.study_name}_cat", 
                storage=self.optuna_storage, 
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(constant_liar=True)
            )
            logging.info(f"Initialized CatBoost study: {self.study_name}_cat")
            
            # Meta-model study
            self.meta_study = optuna.create_study(
                direction="minimize", 
                study_name=f"{self.study_name}_meta", 
                storage=self.optuna_storage, 
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(constant_liar=True)
            )
            logging.info(f"Initialized meta-model study: {self.study_name}_meta")
            
        except Exception as e:
            logging.error(f"Error initializing studies: {e}")
            raise
    
    def objective_lgbm(self, trial):
        """Optuna objective function for LightGBM."""
        params = {
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 4, 1024),
            'max_depth': trial.suggest_int('lgbm_max_depth', 3, 50),
            'min_data_in_leaf': trial.suggest_int('lgbm_min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_float('lgbm_feature_fraction', 0.2, 1.0),
            'bagging_fraction': trial.suggest_float('lgbm_bagging_fraction', 0.2, 1.0),
            'bagging_freq': trial.suggest_int('lgbm_bagging_freq', 0, 10),
            'lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1e-8, 10.0, log=True),
        }
        
        # Add fixed params
        params.update({
            'objective': 'regression_l1',
            'boosting_type': 'gbdt',
            'n_estimators': 10000,  # Use early stopping instead
            'random_state': self.seed,
            'n_jobs': -1,
            'verbose': -1,
        })
        
        # Initialize cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_df)):
            X_train = self.train_df.iloc[train_idx][self.features]
            y_train = self.train_df.iloc[train_idx][self.target]
            X_val = self.train_df.iloc[val_idx][self.features]
            y_val = self.train_df.iloc[val_idx][self.target]
            
            # Train model
            model = LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgb_rmsle,
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Predict and calculate score
            preds = model.predict(X_val)
            score = rmsle(y_val, preds)
            cv_scores.append(score)
        
        # Average CV score
        mean_cv_score = np.mean(cv_scores)
        
        # Final validation score
        final_model = LGBMRegressor(**params)
        final_model.fit(
            self.train_df[self.features], self.train_df[self.target],
            eval_set=[(self.valid_df[self.features], self.valid_df[self.target])],
            eval_metric=lgb_rmsle,
            early_stopping_rounds=100,
            verbose=False
        )
        valid_preds = final_model.predict(self.valid_df[self.features])
        valid_score = rmsle(self.valid_df[self.target], valid_preds)
        
        # Log progress
        logging.info(f"LightGBM - CV: {mean_cv_score:.5f}, Valid: {valid_score:.5f}")
        
        return valid_score
    
    def objective_catboost(self, trial):
        """Optuna objective function for CatBoost."""
        params = {
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.001, 0.3, log=True),
            'depth': trial.suggest_int('cat_depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 0.1, 100.0, log=True),
            'random_strength': trial.suggest_float('cat_random_strength', 0.1, 10.0),
            'bagging_temperature': trial.suggest_float('cat_bagging_temperature', 0.0, 10.0),
            'border_count': trial.suggest_int('cat_border_count', 32, 255),
            'grow_policy': trial.suggest_categorical('cat_grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        }
        
        # Add fixed params
        params.update({
            'iterations': 10000,  # Use early stopping instead
            'random_seed': self.seed,
            'loss_function': 'MAE',  # Works well for RMSLE
            'verbose': 0,
            'allow_writing_files': False
        })
        
        # Initialize cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_df)):
            X_train = self.train_df.iloc[train_idx][self.features]
            y_train = self.train_df.iloc[train_idx][self.target]
            X_val = self.train_df.iloc[val_idx][self.features]
            y_val = self.train_df.iloc[val_idx][self.target]
            
            # Train model
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Predict and calculate score
            preds = model.predict(X_val)
            score = rmsle(y_val, preds)
            cv_scores.append(score)
        
        # Average CV score
        mean_cv_score = np.mean(cv_scores)
        
        # Final validation score
        final_model = CatBoostRegressor(**params)
        final_model.fit(
            self.train_df[self.features], self.train_df[self.target],
            eval_set=[(self.valid_df[self.features], self.valid_df[self.target])],
            early_stopping_rounds=100,
            verbose=False
        )
        valid_preds = final_model.predict(self.valid_df[self.features])
        valid_score = rmsle(self.valid_df[self.target], valid_preds)
        
        # Log progress
        logging.info(f"CatBoost - CV: {mean_cv_score:.5f}, Valid: {valid_score:.5f}")
        
        return valid_score
    
    def objective_meta(self, trial):
        """Optuna objective function for the meta-model."""
        # First, get best LightGBM and CatBoost models
        try:
            lgbm_best_params = self.lgbm_study.best_params
            cat_best_params = self.cat_study.best_params
        except Exception as e:
            logging.warning(f"Could not get best base model params: {e}. Using defaults.")
            lgbm_best_params = {}
            cat_best_params = {}
        
        # Train base models with cross-validation to get OOF predictions
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        # Initialize OOF predictions
        oof_lgb = np.zeros(len(self.train_df))
        oof_cat = np.zeros(len(self.train_df))
        valid_lgb = np.zeros(len(self.valid_df))
        valid_cat = np.zeros(len(self.valid_df))
        
        # Prepare LightGBM parameters
        lgbm_model_params = {
            'objective': 'regression_l1',
            'boosting_type': 'gbdt',
            'n_estimators': 10000,
            'learning_rate': 0.02,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_child_samples': 20,
            'random_state': self.seed,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Update with optimized parameters
        for key, value in lgbm_best_params.items():
            param_name = key.replace('lgbm_', '')
            lgbm_model_params[param_name] = value
        
        # Prepare CatBoost parameters
        cat_model_params = {
            'loss_function': 'MAE',
            'iterations': 10000,
            'learning_rate': 0.03,
            'depth': 6,
            'random_seed': self.seed,
            'verbose': 0,
            'allow_writing_files': False
        }
        
        # Update with optimized parameters
        for key, value in cat_best_params.items():
            param_name = key.replace('cat_', '')
            cat_model_params[param_name] = value
        
        # Train base models and get OOF predictions
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_df)):
            # Split data
            X_train = self.train_df.iloc[train_idx][self.features]
            y_train = self.train_df.iloc[train_idx][self.target]
            X_val = self.train_df.iloc[val_idx][self.features]
            y_val = self.train_df.iloc[val_idx][self.target]
            
            # Train LightGBM
            lgb_model = LGBMRegressor(**lgbm_model_params)
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgb_rmsle,
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Get OOF and validation predictions
            oof_lgb[val_idx] = lgb_model.predict(X_val)
            valid_lgb += lgb_model.predict(self.valid_df[self.features]) / self.n_folds
            
            # Train CatBoost
            cat_model = CatBoostRegressor(**cat_model_params)
            cat_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Get OOF and validation predictions
            oof_cat[val_idx] = cat_model.predict(X_val)
            valid_cat += cat_model.predict(self.valid_df[self.features]) / self.n_folds
        
        # Create meta-features
        meta_features_train = pd.DataFrame({
            'lgb_pred': oof_lgb,
            'cat_pred': oof_cat,
            'abs_lgb_error': np.abs(self.train_df[self.target].values - oof_lgb),
            'abs_cat_error': np.abs(self.train_df[self.target].values - oof_cat),
            'pred_diff': np.abs(oof_lgb - oof_cat),
            'pred_mean': (oof_lgb + oof_cat) / 2,
            'pred_product': oof_lgb * oof_cat,
            'checkout_price': self.train_df['checkout_price'].values if 'checkout_price' in self.train_df.columns else 0,
            'discount': self.train_df['discount'].values if 'discount' in self.train_df.columns else 0,
            'weekofyear': self.train_df['weekofyear'].values if 'weekofyear' in self.train_df.columns else 0,
        })
        
        meta_features_valid = pd.DataFrame({
            'lgb_pred': valid_lgb,
            'cat_pred': valid_cat,
            'abs_lgb_error': np.abs(self.valid_df[self.target].values - valid_lgb),
            'abs_cat_error': np.abs(self.valid_df[self.target].values - valid_cat),
            'pred_diff': np.abs(valid_lgb - valid_cat),
            'pred_mean': (valid_lgb + valid_cat) / 2,
            'pred_product': valid_lgb * valid_cat,
            'checkout_price': self.valid_df['checkout_price'].values if 'checkout_price' in self.valid_df.columns else 0,
            'discount': self.valid_df['discount'].values if 'discount' in self.valid_df.columns else 0,
            'weekofyear': self.valid_df['weekofyear'].values if 'weekofyear' in self.valid_df.columns else 0,
        })
        
        # Select meta-model type and parameters
        meta_model_type = trial.suggest_categorical('meta_model_type', ['ridge', 'linear', 'lgbm'])
        
        if meta_model_type == 'ridge':
            alpha = trial.suggest_float('meta_alpha', 0.01, 10.0, log=True)
            meta_model = Ridge(alpha=alpha, random_state=self.seed)
        elif meta_model_type == 'linear':
            meta_model = LinearRegression()
        else:  # lgbm
            meta_lgbm_params = {
                'objective': 'regression_l1',
                'boosting_type': 'gbdt',
                'n_estimators': 100,
                'learning_rate': trial.suggest_float('meta_learning_rate', 0.001, 0.1, log=True),
                'num_leaves': trial.suggest_int('meta_num_leaves', 4, 31),
                'max_depth': trial.suggest_int('meta_max_depth', 2, 7),
                'lambda_l1': trial.suggest_float('meta_lambda_l1', 1e-8, 1.0, log=True),
                'lambda_l2': trial.suggest_float('meta_lambda_l2', 1e-8, 1.0, log=True),
                'random_state': self.seed,
                'verbose': -1
            }
            meta_model = LGBMRegressor(**meta_lgbm_params)
        
        # Cross-validation for meta-model
        kf_meta = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        meta_cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf_meta.split(meta_features_train)):
            # Split meta-features
            X_train = meta_features_train.iloc[train_idx]
            y_train = self.train_df.iloc[train_idx][self.target]
            X_val = meta_features_train.iloc[val_idx]
            y_val = self.train_df.iloc[val_idx][self.target]
            
            # Train meta-model
            meta_model.fit(X_train, y_train)
            
            # Predict and calculate score
            preds = meta_model.predict(X_val)
            preds = np.clip(preds, 0, None)  # Clip negative predictions
            score = rmsle(y_val, preds)
            meta_cv_scores.append(score)
        
        # Average CV score
        mean_meta_cv_score = np.mean(meta_cv_scores)
        
        # Train final meta-model on all training data
        meta_model.fit(meta_features_train, self.train_df[self.target])
        
        # Make predictions on validation set
        valid_meta_preds = meta_model.predict(meta_features_valid)
        valid_meta_preds = np.clip(valid_meta_preds, 0, None)  # Clip negative predictions
        valid_meta_score = rmsle(self.valid_df[self.target], valid_meta_preds)
        
        # Calculate improvement over best base model
        base_scores = [
            rmsle(self.valid_df[self.target], valid_lgb),
            rmsle(self.valid_df[self.target], valid_cat)
        ]
        best_base_score = min(base_scores)
        improvement = (best_base_score - valid_meta_score) / best_base_score * 100
        
        # Log progress
        logging.info(f"Meta-model ({meta_model_type}) - CV: {mean_meta_cv_score:.5f}, Valid: {valid_meta_score:.5f}, Improvement: {improvement:.2f}%")
        
        return valid_meta_score
    
    def optimize(self):
        """Run hyperparameter optimization for all models."""
        logging.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Optimize LightGBM
        logging.info("Optimizing LightGBM model...")
        self.lgbm_study.optimize(self.objective_lgbm, n_trials=self.n_trials, timeout=None)
        logging.info(f"Best LightGBM score: {self.lgbm_study.best_value:.5f}")
        logging.info(f"Best LightGBM parameters: {self.lgbm_study.best_params}")
        
        # Optimize CatBoost
        logging.info("Optimizing CatBoost model...")
        self.cat_study.optimize(self.objective_catboost, n_trials=self.n_trials, timeout=None)
        logging.info(f"Best CatBoost score: {self.cat_study.best_value:.5f}")
        logging.info(f"Best CatBoost parameters: {self.cat_study.best_params}")
        
        # Optimize meta-model using the best LightGBM and CatBoost models
        logging.info("Optimizing meta-model...")
        self.meta_study.optimize(self.objective_meta, n_trials=self.n_trials, timeout=None)
        logging.info(f"Best meta-model score: {self.meta_study.best_value:.5f}")
        logging.info(f"Best meta-model parameters: {self.meta_study.best_params}")
        
        # Store results
        self.optimization_results = {
            'lgbm': {
                'best_score': self.lgbm_study.best_value,
                'best_params': self.lgbm_study.best_params,
                'n_trials': len(self.lgbm_study.trials),
            },
            'cat': {
                'best_score': self.cat_study.best_value,
                'best_params': self.cat_study.best_params,
                'n_trials': len(self.cat_study.trials),
            },
            'meta': {
                'best_score': self.meta_study.best_value,
                'best_params': self.meta_study.best_params,
                'n_trials': len(self.meta_study.trials),
            }
        }
        
        # Save optimization results
        self._save_results()
        
        return self.optimization_results
    
    def _save_results(self):
        """Save optimization results to files."""
        # Create directory for results
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save best parameters as JSON
        results_path = f"{self.output_dir}/best_parameters.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        logging.info(f"Saved best parameters to {results_path}")
        
        # Save visualization figures
        self._save_visualizations()
    
    def _save_visualizations(self):
        """Generate and save visualization plots."""
        # Create visualizations directory
        vis_dir = f"{self.output_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # LightGBM parameter importance
        try:
            fig = plot_param_importances(self.lgbm_study)
            fig.write_image(f"{vis_dir}/lgbm_param_importance.png")
            
            fig = plot_optimization_history(self.lgbm_study)
            fig.write_image(f"{vis_dir}/lgbm_optimization_history.png")
        except Exception as e:
            logging.warning(f"Could not generate LightGBM visualizations: {e}")
        
        # CatBoost parameter importance
        try:
            fig = plot_param_importances(self.cat_study)
            fig.write_image(f"{vis_dir}/cat_param_importance.png")
            
            fig = plot_optimization_history(self.cat_study)
            fig.write_image(f"{vis_dir}/cat_optimization_history.png")
        except Exception as e:
            logging.warning(f"Could not generate CatBoost visualizations: {e}")
        
        # Meta-model parameter importance
        try:
            fig = plot_param_importances(self.meta_study)
            fig.write_image(f"{vis_dir}/meta_param_importance.png")
            
            fig = plot_optimization_history(self.meta_study)
            fig.write_image(f"{vis_dir}/meta_optimization_history.png")
        except Exception as e:
            logging.warning(f"Could not generate meta-model visualizations: {e}")
        
        # Create models vs performance report
        try:
            model_names = ['LightGBM', 'CatBoost', 'Meta-Model']
            best_scores = [
                self.lgbm_study.best_value,
                self.cat_study.best_value,
                self.meta_study.best_value
            ]
            
            plt.figure(figsize=(10, 6))
            plt.bar(model_names, best_scores, color=['blue', 'green', 'red'])
            plt.xlabel('Model')
            plt.ylabel('RMSLE')
            plt.title('Best Model Performance Comparison')
            
            for i, score in enumerate(best_scores):
                plt.text(i, score, f"{score:.5f}", ha='center', va='bottom')
            
            plt.savefig(f"{vis_dir}/model_performance_comparison.png")
            plt.close()
        except Exception as e:
            logging.warning(f"Could not generate performance comparison: {e}")
    
    def train_best_models(self):
        """Train models using the best hyperparameters found."""
        logging.info("Training models with best hyperparameters...")
        
        # Get best parameters
        lgbm_params = self.optimization_results['lgbm']['best_params']
        cat_params = self.optimization_results['cat']['best_params']
        meta_params = self.optimization_results['meta']['best_params']
        
        # Convert parameter names by removing prefixes
        lgbm_model_params = {k.replace('lgbm_', ''): v for k, v in lgbm_params.items()}
        cat_model_params = {k.replace('cat_', ''): v for k, v in cat_params.items()}
        
        # Add default parameters
        lgbm_model_params.update({
            'objective': 'regression_l1',
            'boosting_type': 'gbdt',
            'n_estimators': 10000,
            'random_state': self.seed,
            'n_jobs': -1,
            'verbose': -1,
        })
        
        cat_model_params.update({
            'loss_function': 'MAE',
            'iterations': 10000,
            'random_seed': self.seed,
            'verbose': 0,
            'allow_writing_files': False
        })
        
        # Train base models
        lgbm_model = LGBMRegressor(**lgbm_model_params)
        lgbm_model.fit(
            self.train_df[self.features], self.train_df[self.target],
            eval_set=[(self.valid_df[self.features], self.valid_df[self.target])],
            eval_metric=lgb_rmsle,
            early_stopping_rounds=100,
            verbose=False
        )
        
        cat_model = CatBoostRegressor(**cat_model_params)
        cat_model.fit(
            self.train_df[self.features], self.train_df[self.target],
            eval_set=[(self.valid_df[self.features], self.valid_df[self.target])],
            early_stopping_rounds=100,
            verbose=False
        )
        
        # Make base model predictions
        lgbm_train_preds = lgbm_model.predict(self.train_df[self.features])
        cat_train_preds = cat_model.predict(self.train_df[self.features])
        lgbm_valid_preds = lgbm_model.predict(self.valid_df[self.features])
        cat_valid_preds = cat_model.predict(self.valid_df[self.features])
        
        # Create meta-features
        meta_features_train = pd.DataFrame({
            'lgb_pred': lgbm_train_preds,
            'cat_pred': cat_train_preds,
            'abs_lgb_error': np.abs(self.train_df[self.target].values - lgbm_train_preds),
            'abs_cat_error': np.abs(self.train_df[self.target].values - cat_train_preds),
            'pred_diff': np.abs(lgbm_train_preds - cat_train_preds),
            'pred_mean': (lgbm_train_preds + cat_train_preds) / 2,
            'pred_product': lgbm_train_preds * cat_train_preds,
            'checkout_price': self.train_df['checkout_price'].values if 'checkout_price' in self.train_df.columns else 0,
            'discount': self.train_df['discount'].values if 'discount' in self.train_df.columns else 0,
            'weekofyear': self.train_df['weekofyear'].values if 'weekofyear' in self.train_df.columns else 0,
        })
        
        meta_features_valid = pd.DataFrame({
            'lgb_pred': lgbm_valid_preds,
            'cat_pred': cat_valid_preds,
            'abs_lgb_error': np.abs(self.valid_df[self.target].values - lgbm_valid_preds),
            'abs_cat_error': np.abs(self.valid_df[self.target].values - cat_valid_preds),
            'pred_diff': np.abs(lgbm_valid_preds - cat_valid_preds),
            'pred_mean': (lgbm_valid_preds + cat_valid_preds) / 2,
            'pred_product': lgbm_valid_preds * cat_valid_preds,
            'checkout_price': self.valid_df['checkout_price'].values if 'checkout_price' in self.valid_df.columns else 0,
            'discount': self.valid_df['discount'].values if 'discount' in self.valid_df.columns else 0,
            'weekofyear': self.valid_df['weekofyear'].values if 'weekofyear' in self.valid_df.columns else 0,
        })
        
        # Train meta-model
        meta_model_type = meta_params.get('meta_model_type', 'ridge')
        
        if meta_model_type == 'ridge':
            alpha = meta_params.get('meta_alpha', 1.0)
            meta_model = Ridge(alpha=alpha, random_state=self.seed)
        elif meta_model_type == 'linear':
            meta_model = LinearRegression()
        else:  # lgbm
            meta_lgbm_params = {
                'objective': 'regression_l1',
                'boosting_type': 'gbdt',
                'n_estimators': 100,
                'learning_rate': meta_params.get('meta_learning_rate', 0.01),
                'num_leaves': meta_params.get('meta_num_leaves', 15),
                'max_depth': meta_params.get('meta_max_depth', 3),
                'lambda_l1': meta_params.get('meta_lambda_l1', 0.1),
                'lambda_l2': meta_params.get('meta_lambda_l2', 0.1),
                'random_state': self.seed,
                'verbose': -1
            }
            meta_model = LGBMRegressor(**meta_lgbm_params)
        
        # Train final meta-model
        meta_model.fit(meta_features_train, self.train_df[self.target])
        
        # Make predictions
        meta_valid_preds = meta_model.predict(meta_features_valid)
        meta_valid_preds = np.clip(meta_valid_preds, 0, None)  # Clip negative predictions
        
        # Calculate scores
        lgbm_score = rmsle(self.valid_df[self.target], lgbm_valid_preds)
        cat_score = rmsle(self.valid_df[self.target], cat_valid_preds)
        meta_score = rmsle(self.valid_df[self.target], meta_valid_preds)
        
        logging.info(f"Final LightGBM RMSLE: {lgbm_score:.5f}")
        logging.info(f"Final CatBoost RMSLE: {cat_score:.5f}")
        logging.info(f"Final Meta-model RMSLE: {meta_score:.5f}")
        
        # Save models
        models_dir = f"{self.output_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        
        with open(f"{models_dir}/lgbm_model.pkl", 'wb') as f:
            pickle.dump(lgbm_model, f)
        
        with open(f"{models_dir}/cat_model.pkl", 'wb') as f:
            pickle.dump(cat_model, f)
        
        with open(f"{models_dir}/meta_model.pkl", 'wb') as f:
            pickle.dump({
                'model': meta_model,
                'type': meta_model_type,
                'features': meta_features_train.columns.tolist()
            }, f)
        
        # Generate SHAP explanations
        try:
            self._generate_shap_explanations(lgbm_model, cat_model, meta_model, meta_model_type)
        except Exception as e:
            logging.warning(f"Could not generate SHAP explanations: {e}")
        
        # Return trained models and scores
        return {
            'models': {
                'lgbm': lgbm_model,
                'cat': cat_model,
                'meta': meta_model
            },
            'scores': {
                'lgbm': lgbm_score,
                'cat': cat_score,
                'meta': meta_score
            },
            'meta_model_type': meta_model_type
        }
    
    def _generate_shap_explanations(self, lgbm_model, cat_model, meta_model, meta_model_type):
        """Generate SHAP explanations for trained models."""
        logging.info("Generating SHAP explanations...")
        
        # Create directory for SHAP plots
        shap_dir = f"{self.output_dir}/shap"
        os.makedirs(shap_dir, exist_ok=True)
        
        # Sample data for SHAP (to keep computation reasonable)
        N_SAMPLES = 1000
        if len(self.train_df) > N_SAMPLES:
            sample_indices = np.random.choice(len(self.train_df), N_SAMPLES, replace=False)
            X_sample = self.train_df.iloc[sample_indices][self.features]
        else:
            X_sample = self.train_df[self.features]
        
        # LightGBM SHAP values
        try:
            lgbm_explainer = shap.TreeExplainer(lgbm_model)
            lgbm_shap_values = lgbm_explainer.shap_values(X_sample)
            
            # SHAP summary plot
            plt.figure(figsize=(10, 12))
            shap.summary_plot(lgbm_shap_values, X_sample, plot_type="bar", show=False)
            plt.title('LightGBM Feature Importance (SHAP)')
            plt.tight_layout()
            plt.savefig(f"{shap_dir}/lgbm_shap_importance.png")
            plt.close()
            
            # Save values for later use
            np.save(f"{shap_dir}/lgbm_shap_values.npy", lgbm_shap_values)
            X_sample.to_csv(f"{shap_dir}/shap_sample_data.csv", index=False)
            
            # SHAP summary dataframe
            lgbm_shap_df = pd.DataFrame({
                'feature': self.features,
                'importance': np.abs(lgbm_shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            lgbm_shap_df.to_csv(f"{shap_dir}/lgbm_shap_importance.csv", index=False)
            
            logging.info("Generated LightGBM SHAP explanations")
        except Exception as e:
            logging.warning(f"Could not generate LightGBM SHAP explanations: {e}")
        
        # CatBoost SHAP values
        try:
            cat_explainer = shap.TreeExplainer(cat_model)
            cat_shap_values = cat_explainer.shap_values(X_sample)
            
            # SHAP summary plot
            plt.figure(figsize=(10, 12))
            shap.summary_plot(cat_shap_values, X_sample, plot_type="bar", show=False)
            plt.title('CatBoost Feature Importance (SHAP)')
            plt.tight_layout()
            plt.savefig(f"{shap_dir}/cat_shap_importance.png")
            plt.close()
            
            # Save values for later use
            np.save(f"{shap_dir}/cat_shap_values.npy", cat_shap_values)
            
            # SHAP summary dataframe
            cat_shap_df = pd.DataFrame({
                'feature': self.features,
                'importance': np.abs(cat_shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            cat_shap_df.to_csv(f"{shap_dir}/cat_shap_importance.csv", index=False)
            
            logging.info("Generated CatBoost SHAP explanations")
        except Exception as e:
            logging.warning(f"Could not generate CatBoost SHAP explanations: {e}")
        
        # Meta-model SHAP values (for tree-based models only)
        if meta_model_type == 'lgbm':
            try:
                # Create meta-features for sample data
                lgbm_sample_preds = lgbm_model.predict(X_sample)
                cat_sample_preds = cat_model.predict(X_sample)
                
                sample_indices = X_sample.index
                meta_features_sample = pd.DataFrame({
                    'lgb_pred': lgbm_sample_preds,
                    'cat_pred': cat_sample_preds,
                    'abs_lgb_error': np.abs(self.train_df.loc[sample_indices, self.target].values - lgbm_sample_preds),
                    'abs_cat_error': np.abs(self.train_df.loc[sample_indices, self.target].values - cat_sample_preds),
                    'pred_diff': np.abs(lgbm_sample_preds - cat_sample_preds),
                    'pred_mean': (lgbm_sample_preds + cat_sample_preds) / 2,
                    'pred_product': lgbm_sample_preds * cat_sample_preds,
                    'checkout_price': X_sample['checkout_price'].values if 'checkout_price' in X_sample.columns else 0,
                    'discount': X_sample['discount'].values if 'discount' in X_sample.columns else 0,
                    'weekofyear': X_sample['weekofyear'].values if 'weekofyear' in X_sample.columns else 0,
                })
                
                # SHAP explanation
                meta_explainer = shap.TreeExplainer(meta_model)
                meta_shap_values = meta_explainer.shap_values(meta_features_sample)
                
                # SHAP summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(meta_shap_values, meta_features_sample, plot_type="bar", show=False)
                plt.title('Meta-Model Feature Importance (SHAP)')
                plt.tight_layout()
                plt.savefig(f"{shap_dir}/meta_shap_importance.png")
                plt.close()
                
                # Save values for later use
                np.save(f"{shap_dir}/meta_shap_values.npy", meta_shap_values)
                meta_features_sample.to_csv(f"{shap_dir}/meta_sample_data.csv", index=False)
                
                # SHAP summary dataframe
                meta_shap_df = pd.DataFrame({
                    'feature': meta_features_sample.columns,
                    'importance': np.abs(meta_shap_values).mean(axis=0)
                }).sort_values('importance', ascending=False)
                meta_shap_df.to_csv(f"{shap_dir}/meta_shap_importance.csv", index=False)
                
                logging.info("Generated Meta-model SHAP explanations")
            except Exception as e:
                logging.warning(f"Could not generate Meta-model SHAP explanations: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for stacking ensemble model')
    parser.add_argument('--train-file', type=str, default='train_processed.csv', help='Path to training data CSV')
    parser.add_argument('--valid-file', type=str, default=None, help='Path to validation data CSV (optional)')
    parser.add_argument('--target', type=str, default='num_orders', help='Target column name')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials per model')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='hyperopt_results', help='Output directory for results')
    parser.add_argument('--train-best', action='store_true', help='Train models with best parameters after optimization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load data
    logging.info(f"Loading data from {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    
    if args.valid_file:
        logging.info(f"Loading validation data from {args.valid_file}")
        valid_df = pd.read_csv(args.valid_file)
    else:
        # Use last 20% of data as validation
        logging.info("No validation file provided, using last 20% of train data")
        train_df = train_df.sort_values('week' if 'week' in train_df.columns else 'id')
        split_idx = int(len(train_df) * 0.8)
        valid_df = train_df.iloc[split_idx:].copy()
        train_df = train_df.iloc[:split_idx].copy()
    
    # Define features (exclude target and ID columns)
    exclude_cols = [args.target, 'id', 'week', 'center_id', 'meal_id']
    features = [col for col in train_df.columns if col not in exclude_cols]
    
    logging.info(f"Training with {len(features)} features")
    logging.info(f"Train shape: {train_df.shape}, Valid shape: {valid_df.shape}")
    
    # Define Optuna storage
    PG_USER = os.environ.get("POSTGRES_USER", "postgres")
    PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
    PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
    PG_DB = os.environ.get("POSTGRES_DB", "optuna")
    PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    
    # Create optimizer
    optimizer = ComprehensiveHyperparameterOptimizer(
        train_df=train_df,
        valid_df=valid_df,
        features=features,
        target=args.target,
        optuna_storage=OPTUNA_DB,
        study_name="comprehensive_food_delivery",
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Train models with best parameters if requested
    if args.train_best:
        logging.info("Training models with best parameters...")
        trained_models = optimizer.train_best_models()
        
        logging.info("Final performance:")
        for model_name, score in trained_models['scores'].items():
            logging.info(f"{model_name}: {score:.5f}")
    
    logging.info("Optimization completed")

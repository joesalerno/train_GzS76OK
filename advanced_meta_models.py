import numpy as np
import pandas as pd
import logging
import os
from sklearn.linear_model import Ridge, ElasticNet, SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)  # Ensure predictions are non-negative
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def train_advanced_meta_model(meta_features, target_values, model_type='ridge', params=None, cv=3, 
                             meta_feature_selection=False, output_dir=None):
    """
    Trains an advanced meta-model with extensive options and evaluation.
    
    Args:
        meta_features: DataFrame containing meta-features
        target_values: Target values to predict
        model_type: Type of meta-model ('ridge', 'elastic_net', 'sgd', 'linear', 'lightgbm', 
                   'catboost', 'rf', 'gbm', 'svr', 'knn', 'mlp')
        params: Dictionary of parameters for the selected model
        cv: Number of cross-validation folds
        meta_feature_selection: Whether to perform feature selection
        output_dir: Directory to save model outputs
        
    Returns:
        dict: Trained model and performance metrics
    """
    logging.info(f"Training advanced meta-model using {model_type}...")
    
    # Initialize default parameters dict if not provided
    if params is None:
        params = {}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Clone features to avoid modifying the original
    X = meta_features.copy()
    y = target_values.copy()
    
    # Feature selection if requested
    if meta_feature_selection:
        # Feature importance-based selection for tree-based models
        if model_type in ['lightgbm', 'catboost', 'rf', 'gbm']:
            # Train a quick model to get feature importances
            if model_type == 'lightgbm':
                selector = LGBMRegressor(n_estimators=100)
            elif model_type == 'catboost':
                selector = CatBoostRegressor(iterations=100, verbose=False)
            elif model_type == 'rf':
                selector = RandomForestRegressor(n_estimators=100)
            else:  # gbm
                selector = GradientBoostingRegressor(n_estimators=100)
            
            selector.fit(X, y)
            
            # Get feature importances
            if model_type == 'catboost':
                importances = selector.get_feature_importance()
            else:
                importances = selector.feature_importances_
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Keep top 80% of cumulative importance
            cum_importance = importance_df['importance'].cumsum() / importance_df['importance'].sum()
            keep_mask = cum_importance <= 0.8
            if keep_mask.sum() < 2:  # Ensure we keep at least 2 features
                keep_mask.iloc[0:2] = True
            
            selected_features = importance_df.loc[keep_mask, 'feature'].values
            X = X[selected_features]
            
            logging.info(f"Selected {len(selected_features)}/{len(meta_features.columns)} features for meta-model")
            
            # Save feature importances if output_dir provided
            if output_dir:
                importance_df.to_csv(f"{output_dir}/meta_feature_importance.csv", index=False)
        
        # Correlation-based selection for linear models
        elif model_type in ['ridge', 'elastic_net', 'sgd', 'linear', 'svr', 'knn', 'mlp']:
            # Calculate correlation with target
            corr = pd.DataFrame({
                'feature': X.columns,
                'correlation': [np.corrcoef(X[col], y)[0, 1] for col in X.columns]
            })
            corr['abs_correlation'] = np.abs(corr['correlation'])
            corr = corr.sort_values('abs_correlation', ascending=False)
            
            # Keep features with correlation above a threshold
            threshold = 0.1
            selected_features = corr.loc[corr['abs_correlation'] > threshold, 'feature'].values
            
            # Ensure we keep at least 2 features
            if len(selected_features) < 2:
                selected_features = corr.head(2)['feature'].values
                
            X = X[selected_features]
            
            logging.info(f"Selected {len(selected_features)}/{len(meta_features.columns)} features for meta-model")
            
            # Save feature correlations if output_dir provided
            if output_dir:
                corr.to_csv(f"{output_dir}/meta_feature_correlation.csv", index=False)
    
    # Initialize model based on model_type with parameters
    if model_type == 'ridge':
        model = Ridge(alpha=params.get('alpha', 1.0))
    elif model_type == 'elastic_net':
        model = ElasticNet(
            alpha=params.get('alpha', 1.0),
            l1_ratio=params.get('l1_ratio', 0.5),
            max_iter=params.get('max_iter', 1000)
        )
    elif model_type == 'sgd':
        model = SGDRegressor(
            alpha=params.get('alpha', 0.0001),
            penalty=params.get('penalty', 'l2'),
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42)
        )
    elif model_type == 'linear':
        model = LinearRegression(
            fit_intercept=params.get('fit_intercept', True),
            n_jobs=params.get('n_jobs', -1)
        )
    elif model_type == 'lightgbm':
        model = LGBMRegressor(
            objective=params.get('objective', 'regression_l1'),
            boosting_type=params.get('boosting_type', 'gbdt'),
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.01),
            num_leaves=params.get('num_leaves', 31),
            max_depth=params.get('max_depth', 5),
            random_state=params.get('random_state', 42)
        )
    elif model_type == 'catboost':
        model = CatBoostRegressor(
            loss_function=params.get('loss_function', 'MAE'),
            iterations=params.get('iterations', 100),
            learning_rate=params.get('learning_rate', 0.03),
            depth=params.get('depth', 5),
            random_seed=params.get('random_seed', 42),
            verbose=0
        )
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=params.get('random_state', 42)
        )
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            loss=params.get('loss', 'ls'),
            random_state=params.get('random_state', 42)
        )
    elif model_type == 'svr':
        model = SVR(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            epsilon=params.get('epsilon', 0.1),
            gamma=params.get('gamma', 'scale')
        )
    elif model_type == 'knn':
        model = KNeighborsRegressor(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'uniform'),
            p=params.get('p', 2)  # Minkowski distance parameter
        )
    elif model_type == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            learning_rate=params.get('learning_rate', 'constant'),
            max_iter=params.get('max_iter', 200),
            random_state=params.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validation metrics
    if cv > 1:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = {
            'rmse': [],
            'rmsle': [],
            'mae': []
        }
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train on this fold
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            preds = model.predict(X_val)
            preds = np.clip(preds, 0, None)  # Clip negative predictions
            
            # Calculate metrics
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
            cv_scores['rmsle'].append(rmsle(y_val, preds))
            cv_scores['mae'].append(mean_absolute_error(y_val, preds))
        
        # Average CV scores
        cv_metrics = {
            'cv_rmse': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse']),
            'cv_rmsle': np.mean(cv_scores['rmsle']),
            'cv_rmsle_std': np.std(cv_scores['rmsle']),
            'cv_mae': np.mean(cv_scores['mae']),
            'cv_mae_std': np.std(cv_scores['mae'])
        }
        
        logging.info(f"Cross-validated RMSLE: {cv_metrics['cv_rmsle']:.5f} ± {cv_metrics['cv_rmsle_std']:.5f}")
    else:
        cv_metrics = {}
    
    # Final model training on all data
    model.fit(X, y)
    
    # Final predictions
    final_preds = model.predict(X)
    final_preds = np.clip(final_preds, 0, None)  # Clip negative predictions
    
    # Calculate final metrics
    final_metrics = {
        'rmse': np.sqrt(mean_squared_error(y, final_preds)),
        'rmsle': rmsle(y, final_preds),
        'mae': mean_absolute_error(y, final_preds)
    }
    
    logging.info(f"Final model RMSLE: {final_metrics['rmsle']:.5f}")
    
    # Save model if output_dir provided
    if output_dir:
        with open(f"{output_dir}/meta_model_{model_type}.pkl", 'wb') as f:
            pickle.dump({
                'model': model,
                'selected_features': X.columns.tolist() if meta_feature_selection else None,
                'cv_metrics': cv_metrics,
                'final_metrics': final_metrics
            }, f)
    
    # Visualize residuals if output_dir provided
    if output_dir:
        plt.figure(figsize=(12, 10))
        
        # Predicted vs actual
        plt.subplot(2, 2, 1)
        plt.scatter(y, final_preds, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_type.upper()} Meta-Model: Actual vs Predicted')
        
        # Residual plot
        plt.subplot(2, 2, 2)
        residuals = y - final_preds
        plt.scatter(final_preds, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.title('Residual Plot')
        
        # Error distribution
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30)
        plt.xlabel('Residual')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        
        # Error vs order magnitude
        plt.subplot(2, 2, 4)
        plt.scatter(np.log1p(y), np.abs(residuals), alpha=0.5)
        plt.xlabel('Log(Actual Value + 1)')
        plt.ylabel('Absolute Error')
        plt.title('Error vs Order Magnitude')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/meta_model_{model_type}_diagnostics.png")
        plt.close()
    
    # Return the model and metrics
    return {
        'model': model,
        'selected_features': X.columns.tolist() if meta_feature_selection else None,
        'cv_metrics': cv_metrics,
        'final_metrics': final_metrics
    }

def train_meta_model_ensemble(meta_features, target_values, model_types=['ridge', 'lightgbm', 'catboost'], 
                             weights=None, output_dir=None):
    """
    Trains an ensemble of meta-models and combines their predictions.
    
    Args:
        meta_features: DataFrame containing meta-features
        target_values: Target values to predict
        model_types: List of model types to include in the ensemble
        weights: Optional weights for each model (defaults to equal weights)
        output_dir: Directory to save model outputs
        
    Returns:
        dict: Ensemble model and performance metrics
    """
    logging.info(f"Training meta-model ensemble with {len(model_types)} models...")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Train individual models
    models = {}
    for model_type in model_types:
        model_result = train_advanced_meta_model(
            meta_features=meta_features,
            target_values=target_values,
            model_type=model_type,
            output_dir=output_dir
        )
        models[model_type] = model_result
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = {model_type: 1/len(model_types) for model_type in model_types}
    else:
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    
    # Generate ensemble predictions
    predictions = np.zeros(len(target_values))
    for model_type, weight in weights.items():
        model = models[model_type]['model']
        
        # For feature-selected models, use only selected features
        if models[model_type]['selected_features']:
            pred = model.predict(meta_features[models[model_type]['selected_features']])
        else:
            pred = model.predict(meta_features)
        
        pred = np.clip(pred, 0, None)  # Clip negative predictions
        predictions += weight * pred
    
    # Calculate ensemble metrics
    ensemble_metrics = {
        'rmse': np.sqrt(mean_squared_error(target_values, predictions)),
        'rmsle': rmsle(target_values, predictions),
        'mae': mean_absolute_error(target_values, predictions)
    }
    
    logging.info(f"Ensemble meta-model RMSLE: {ensemble_metrics['rmsle']:.5f}")
    
    # Compare with individual models
    if output_dir:
        comparison = pd.DataFrame({
            'Model': list(models.keys()) + ['Ensemble'],
            'RMSLE': [models[m]['final_metrics']['rmsle'] for m in model_types] + [ensemble_metrics['rmsle']],
            'RMSE': [models[m]['final_metrics']['rmse'] for m in model_types] + [ensemble_metrics['rmse']],
            'MAE': [models[m]['final_metrics']['mae'] for m in model_types] + [ensemble_metrics['mae']]
        })
        comparison.to_csv(f"{output_dir}/meta_model_ensemble_comparison.csv", index=False)
        
        # Create comparison visualization
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(comparison))
        width = 0.25
        
        plt.bar(x - width, comparison['RMSLE'], width, label='RMSLE')
        plt.bar(x, comparison['RMSE'], width, label='RMSE')
        plt.bar(x + width, comparison['MAE'], width, label='MAE')
        
        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.title('Meta-Model Performance Comparison')
        plt.xticks(x, comparison['Model'])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/meta_model_ensemble_comparison.png")
        plt.close()
    
    # Save ensemble model info
    if output_dir:
        with open(f"{output_dir}/meta_model_ensemble.pkl", 'wb') as f:
            pickle.dump({
                'models': {m: models[m]['model'] for m in model_types},
                'selected_features': {m: models[m]['selected_features'] for m in model_types},
                'weights': weights,
                'metrics': ensemble_metrics
            }, f)
    
    # Return ensemble info
    return {
        'models': {m: models[m]['model'] for m in model_types},
        'selected_features': {m: models[m]['selected_features'] for m in model_types},
        'weights': weights,
        'metrics': ensemble_metrics
    }

def predict_with_meta_ensemble(meta_features, ensemble_model):
    """
    Makes predictions using a meta-model ensemble.
    
    Args:
        meta_features: DataFrame containing meta-features
        ensemble_model: Ensemble model information returned by train_meta_model_ensemble
        
    Returns:
        numpy.array: Ensemble predictions
    """
    predictions = np.zeros(len(meta_features))
    
    for model_type, weight in ensemble_model['weights'].items():
        model = ensemble_model['models'][model_type]
        
        # For feature-selected models, use only selected features
        selected_features = ensemble_model['selected_features'][model_type]
        if selected_features:
            pred = model.predict(meta_features[selected_features])
        else:
            pred = model.predict(meta_features)
        
        pred = np.clip(pred, 0, None)  # Clip negative predictions
        predictions += weight * pred
    
    return predictions

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # This would be replaced with actual data in production
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.abs(X.sum(axis=1) + np.random.randn(1000)))
    
    # Train individual advanced meta-model
    ridge_model = train_advanced_meta_model(
        meta_features=X,
        target_values=y,
        model_type='ridge',
        cv=5,
        meta_feature_selection=True,
        output_dir='meta_model_test'
    )
    
    # Train meta-model ensemble
    ensemble_model = train_meta_model_ensemble(
        meta_features=X,
        target_values=y,
        model_types=['ridge', 'lightgbm', 'catboost'],
        output_dir='meta_model_test'
    )
    
    # Make predictions with ensemble
    test_X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    ensemble_preds = predict_with_meta_ensemble(test_X, ensemble_model)
    
    logging.info(f"Generated {len(ensemble_preds)} ensemble predictions")

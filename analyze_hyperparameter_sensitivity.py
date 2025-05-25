import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_log_error
import optuna

def analyze_hyperparameter_sensitivity(study_name, storage, model_type='lgbm', top_n=20):
    """
    Analyzes the sensitivity of model performance to different hyperparameters.
    
    Args:
        study_name: Base name of the Optuna study
        storage: Optuna storage string
        model_type: Model type ('lgbm', 'cat', or 'meta')
        top_n: Number of top trials to analyze
    
    Returns:
        DataFrame: Hyperparameter importance analysis
    """
    logging.info(f"Analyzing hyperparameter sensitivity for {model_type} model...")
    
    # Load study
    full_study_name = f"{study_name}_{model_type}"
    try:
        study = optuna.load_study(study_name=full_study_name, storage=storage)
    except Exception as e:
        logging.error(f"Could not load study {full_study_name}: {e}")
        return None
    
    # Get trials dataframe
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        logging.warning(f"No trials found for study {full_study_name}")
        return None
    
    # Filter for completed trials and get top N
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    if completed_trials.empty:
        logging.warning(f"No completed trials found for study {full_study_name}")
        return None
    
    top_trials = completed_trials.nsmallest(top_n, 'value')
    
    # Extract parameter columns
    param_cols = [col for col in top_trials.columns if col.startswith('params_')]
    
    # Calculate parameter variance within top trials
    param_variance = {}
    param_range = {}
    for param in param_cols:
        param_name = param.replace('params_', '')
        values = top_trials[param].values
        
        try:
            if values.dtype == np.float64 or values.dtype == np.int64:
                param_variance[param_name] = np.var(values)
                param_range[param_name] = (np.min(values), np.max(values))
            else:
                # For categorical parameters, calculate entropy
                unique_vals, counts = np.unique(values, return_counts=True)
                if len(unique_vals) > 1:
                    probs = counts / len(values)
                    entropy = -np.sum(probs * np.log2(probs))
                    param_variance[param_name] = entropy
                    param_range[param_name] = list(unique_vals)
                else:
                    param_variance[param_name] = 0
                    param_range[param_name] = list(unique_vals)
        except Exception as e:
            logging.error(f"Error analyzing parameter {param_name}: {e}")
            param_variance[param_name] = float('nan')
            param_range[param_name] = "Error"
    
    # Create parameter importance dataframe
    param_importance = pd.DataFrame({
        'parameter': list(param_variance.keys()),
        'variance': list(param_variance.values()),
        'range': [param_range[p] for p in param_variance.keys()]
    })
    
    # Sort by variance (higher variance = less important parameter)
    param_importance = param_importance.sort_values('variance')
    
    # Calculate performance stability
    best_score = study.best_value
    worst_of_top = top_trials['value'].max()
    perf_range = (worst_of_top - best_score) / best_score * 100
    
    logging.info(f"Performance stability range: {perf_range:.2f}% within top {top_n} trials")
    logging.info(f"Best score: {best_score:.5f}")
    
    # Save results
    os.makedirs("hyperparameter_analysis", exist_ok=True)
    param_importance.to_csv(f"hyperparameter_analysis/{model_type}_parameter_importance.csv", index=False)
    
    # Create visualization
    try:
        plt.figure(figsize=(10, max(6, len(param_importance) * 0.4)))
        
        # Plot parameter importance (inverse of variance - smaller variance = more important)
        importance_values = 1 / (param_importance['variance'] + 1e-10)  # Add small epsilon to avoid division by zero
        importance_values = importance_values / importance_values.max() * 100  # Normalize to 0-100 scale
        
        plt.barh(param_importance['parameter'], importance_values, color='skyblue')
        plt.xlabel('Relative Importance (%)')
        plt.ylabel('Hyperparameter')
        plt.title(f'{model_type.upper()} Model Hyperparameter Importance')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"hyperparameter_analysis/{model_type}_parameter_importance.png")
        plt.close()
        
        # Create parameter distribution plots for top numerical parameters
        num_params = [p for p in param_importance['parameter'] 
                    if top_trials[f'params_{p}'].dtype in (np.float64, np.int64)][:5]
        
        if num_params:
            plt.figure(figsize=(15, 3*len(num_params)))
            
            for i, param in enumerate(num_params):
                plt.subplot(len(num_params), 1, i+1)
                
                # Plot histogram of all trials
                plt.hist(completed_trials[f'params_{param}'], bins=20, alpha=0.5, label='All trials')
                
                # Plot histogram of top trials
                plt.hist(top_trials[f'params_{param}'], bins=10, alpha=0.7, label=f'Top {top_n} trials')
                
                plt.xlabel(param)
                plt.ylabel('Count')
                plt.title(f'Distribution of {param} values')
                plt.legend()
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"hyperparameter_analysis/{model_type}_parameter_distributions.png")
            plt.close()
            
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
    
    return param_importance

def analyze_cross_model_interactions(study_name, storage, model_pairs=[('lgbm', 'cat'), ('lgbm', 'meta'), ('cat', 'meta')]):
    """
    Analyzes how hyperparameters in different models interact with each other.
    
    Args:
        study_name: Base name of the Optuna study
        storage: Optuna storage string
        model_pairs: List of model type pairs to analyze
        
    Returns:
        dict: Cross-model interaction analysis
    """
    logging.info("Analyzing cross-model hyperparameter interactions...")
    
    results = {}
    
    for model1, model2 in model_pairs:
        try:
            # Load studies
            study1 = optuna.load_study(study_name=f"{study_name}_{model1}", storage=storage)
            study2 = optuna.load_study(study_name=f"{study_name}_{model2}", storage=storage)
            
            # Get best parameters
            best_params1 = study1.best_params
            best_params2 = study2.best_params
            
            # Get top trials
            trials1_df = study1.trials_dataframe()
            trials2_df = study2.trials_dataframe()
            
            # Filter completed trials
            completed1 = trials1_df[trials1_df['state'] == 'COMPLETE']
            completed2 = trials2_df[trials2_df['state'] == 'COMPLETE']
            
            # Get performance correlation
            model1_best_trial_number = study1.best_trial.number
            model2_best_trial_number = study2.best_trial.number
            
            # Extract creation dates of best trials
            if 'datetime_start' in completed1.columns and 'datetime_start' in completed2.columns:
                model1_best_date = completed1[completed1['number'] == model1_best_trial_number]['datetime_start'].iloc[0]
                model2_best_date = completed2[completed2['number'] == model2_best_trial_number]['datetime_start'].iloc[0]
                
                # Check if one best model was found after the other
                time_difference = abs((model2_best_date - model1_best_date).total_seconds())
                sequential_discovery = time_difference > 60  # More than 60 seconds apart
                
                results[f"{model1}-{model2}"] = {
                    'model1_best_value': study1.best_value,
                    'model2_best_value': study2.best_value,
                    'model1_best_params': best_params1,
                    'model2_best_params': best_params2,
                    'time_difference_seconds': time_difference,
                    'sequential_discovery': sequential_discovery
                }
                
                logging.info(f"Models {model1}-{model2} were optimized {'sequentially' if sequential_discovery else 'in parallel'}")
            else:
                logging.warning(f"Datetime information not available for {model1}-{model2} pair")
            
        except Exception as e:
            logging.error(f"Error analyzing {model1}-{model2} interaction: {e}")
    
    return results

def analyze_ensemble_optimization_process(study_name, storage):
    """
    Analyzes the optimization process of the entire ensemble, showing how
    base models and meta-model performance evolved during the optimization.
    
    Args:
        study_name: Base name of the Optuna study
        storage: Optuna storage string
        
    Returns:
        DataFrame: Timeline of optimization events
    """
    logging.info("Analyzing ensemble optimization process...")
    
    # Load studies
    try:
        lgbm_study = optuna.load_study(study_name=f"{study_name}_lgbm", storage=storage)
        cat_study = optuna.load_study(study_name=f"{study_name}_cat", storage=storage)
        meta_study = optuna.load_study(study_name=f"{study_name}_meta", storage=storage)
    except Exception as e:
        logging.error(f"Error loading studies: {e}")
        return None
    
    # Get trials for each model
    lgbm_trials = lgbm_study.trials_dataframe()
    cat_trials = cat_study.trials_dataframe()
    meta_trials = meta_study.trials_dataframe()
    
    # Add model type column
    if not lgbm_trials.empty:
        lgbm_trials['model_type'] = 'LightGBM'
    if not cat_trials.empty:
        cat_trials['model_type'] = 'CatBoost'
    if not meta_trials.empty:
        meta_trials['model_type'] = 'Meta-Model'
    
    # Combine all trials
    all_trials = pd.concat([lgbm_trials, cat_trials, meta_trials], ignore_index=True)
    
    # Keep only completed trials
    all_trials = all_trials[all_trials['state'] == 'COMPLETE']
    
    # Sort by datetime_start
    if 'datetime_start' in all_trials.columns:
        all_trials = all_trials.sort_values('datetime_start')
        
        # Add elapsed minutes from first trial
        all_trials['elapsed_minutes'] = (all_trials['datetime_start'] - all_trials['datetime_start'].min()).dt.total_seconds() / 60
        
        # Get only relevant columns
        timeline_df = all_trials[['number', 'model_type', 'value', 'elapsed_minutes', 'datetime_start', 'datetime_complete']]
        
        # Save timeline data
        os.makedirs("hyperparameter_analysis", exist_ok=True)
        timeline_df.to_csv("hyperparameter_analysis/optimization_timeline.csv", index=False)
        
        # Create visualization of optimization timeline
        try:
            plt.figure(figsize=(15, 8))
            
            # Plot performance over time for each model type
            for model_type in ['LightGBM', 'CatBoost', 'Meta-Model']:
                model_data = timeline_df[timeline_df['model_type'] == model_type]
                if not model_data.empty:
                    plt.plot(model_data['elapsed_minutes'], model_data['value'], '-o', 
                             label=model_type, alpha=0.7)
            
            # Calculate and plot best seen so far for each model
            for model_type in ['LightGBM', 'CatBoost', 'Meta-Model']:
                model_data = timeline_df[timeline_df['model_type'] == model_type]
                if not model_data.empty:
                    # Sort by time
                    model_data = model_data.sort_values('elapsed_minutes')
                    # Calculate cumulative minimum (best seen so far)
                    model_data['best_so_far'] = model_data['value'].cummin()
                    plt.plot(model_data['elapsed_minutes'], model_data['best_so_far'], '--', 
                             label=f"{model_type} (Best)", alpha=0.5)
            
            plt.xlabel('Elapsed Time (minutes)')
            plt.ylabel('RMSLE')
            plt.title('Optimization Performance over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add annotations for best results
            for model_type in ['LightGBM', 'CatBoost', 'Meta-Model']:
                model_data = timeline_df[timeline_df['model_type'] == model_type]
                if not model_data.empty:
                    best_idx = model_data['value'].idxmin()
                    best_time = model_data.loc[best_idx, 'elapsed_minutes']
                    best_value = model_data.loc[best_idx, 'value']
                    plt.annotate(f'{best_value:.5f}', 
                                 xy=(best_time, best_value),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig("hyperparameter_analysis/optimization_timeline.png")
            plt.close()
            
            # Create a second plot showing cumulative improvement
            plt.figure(figsize=(15, 8))
            
            # Start with all models at baseline (worst performance)
            baseline = max(timeline_df['value'].max(), 0.3)  # Use reasonable baseline
            
            # Calculate improvement over baseline
            improvement_data = {}
            
            for model_type in ['LightGBM', 'CatBoost', 'Meta-Model']:
                model_data = timeline_df[timeline_df['model_type'] == model_type]
                if not model_data.empty:
                    model_data = model_data.sort_values('elapsed_minutes')
                    model_data['best_so_far'] = model_data['value'].cummin()
                    model_data['improvement'] = (baseline - model_data['best_so_far']) / baseline * 100
                    improvement_data[model_type] = model_data
            
            # Plot improvement curves
            for model_type, model_data in improvement_data.items():
                plt.plot(model_data['elapsed_minutes'], model_data['improvement'], '-o', 
                         label=model_type, alpha=0.7)
            
            plt.xlabel('Elapsed Time (minutes)')
            plt.ylabel('Improvement over Baseline (%)')
            plt.title('Cumulative Optimization Improvement')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add final improvement annotations
            for model_type, model_data in improvement_data.items():
                last_idx = model_data['improvement'].idxmax()
                last_time = model_data.loc[last_idx, 'elapsed_minutes']
                last_improvement = model_data.loc[last_idx, 'improvement']
                plt.annotate(f'{last_improvement:.2f}%', 
                             xy=(last_time, last_improvement),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig("hyperparameter_analysis/optimization_improvement.png")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating optimization timeline visualization: {e}")
        
        return timeline_df
    else:
        logging.warning("Datetime information not available in trials")
        return None

def run_comprehensive_analysis(study_name, storage):
    """
    Runs a comprehensive analysis of all optimization studies.
    
    Args:
        study_name: Base name of the Optuna study
        storage: Optuna storage string
    """
    logging.info("Running comprehensive hyperparameter optimization analysis...")
    
    # Create output directory
    os.makedirs("hyperparameter_analysis", exist_ok=True)
    
    # 1. Analyze individual model sensitivity
    lgbm_sensitivity = analyze_hyperparameter_sensitivity(study_name, storage, 'lgbm')
    cat_sensitivity = analyze_hyperparameter_sensitivity(study_name, storage, 'cat')
    meta_sensitivity = analyze_hyperparameter_sensitivity(study_name, storage, 'meta')
    
    # 2. Analyze cross-model interactions
    interactions = analyze_cross_model_interactions(study_name, storage)
    
    # Save cross-model analysis
    with open("hyperparameter_analysis/cross_model_interactions.txt", "w") as f:
        for pair, data in interactions.items():
            f.write(f"=== {pair} Interaction Analysis ===\n\n")
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\n")
    
    # 3. Analyze optimization process
    timeline = analyze_ensemble_optimization_process(study_name, storage)
    
    # 4. Create consolidated analysis report
    with open("hyperparameter_analysis/optimization_summary.txt", "w") as f:
        f.write("=== Food Delivery Demand Forecasting Hyperparameter Optimization Analysis ===\n\n")
        
        # Add model performance summary
        f.write("== Model Performance Summary ==\n")
        try:
            lgbm_study = optuna.load_study(study_name=f"{study_name}_lgbm", storage=storage)
            cat_study = optuna.load_study(study_name=f"{study_name}_cat", storage=storage)
            meta_study = optuna.load_study(study_name=f"{study_name}_meta", storage=storage)
            
            best_lgbm = lgbm_study.best_value
            best_cat = cat_study.best_value
            best_meta = meta_study.best_value
            
            f.write(f"Best LightGBM RMSLE: {best_lgbm:.5f}\n")
            f.write(f"Best CatBoost RMSLE: {best_cat:.5f}\n")
            f.write(f"Best Meta-Model RMSLE: {best_meta:.5f}\n\n")
            
            # Calculate improvements
            best_base = min(best_lgbm, best_cat)
            improvement = (best_base - best_meta) / best_base * 100
            f.write(f"Meta-Model Improvement over Best Base Model: {improvement:.2f}%\n\n")
            
            # Add number of trials
            lgbm_trials = len(lgbm_study.trials)
            cat_trials = len(cat_study.trials)
            meta_trials = len(meta_study.trials)
            
            f.write(f"LightGBM Trials: {lgbm_trials}\n")
            f.write(f"CatBoost Trials: {cat_trials}\n")
            f.write(f"Meta-Model Trials: {meta_trials}\n\n")
            
        except Exception as e:
            f.write(f"Error retrieving performance summary: {e}\n\n")
        
        # Add key hyperparameters for each model
        f.write("== Best Hyperparameters ==\n")
        try:
            f.write("LightGBM:\n")
            for param, value in lgbm_study.best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nCatBoost:\n")
            for param, value in cat_study.best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nMeta-Model:\n")
            for param, value in meta_study.best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        except Exception as e:
            f.write(f"Error retrieving best hyperparameters: {e}\n\n")
        
        # Add sensitivity summary
        f.write("== Hyperparameter Sensitivity Summary ==\n")
        f.write("Most Important Parameters (lowest variance in top trials):\n")
        
        if lgbm_sensitivity is not None:
            f.write("\nLightGBM Important Parameters:\n")
            for _, row in lgbm_sensitivity.head(3).iterrows():
                f.write(f"  {row['parameter']}: range={row['range']}\n")
        
        if cat_sensitivity is not None:
            f.write("\nCatBoost Important Parameters:\n")
            for _, row in cat_sensitivity.head(3).iterrows():
                f.write(f"  {row['parameter']}: range={row['range']}\n")
        
        if meta_sensitivity is not None:
            f.write("\nMeta-Model Important Parameters:\n")
            for _, row in meta_sensitivity.head(3).iterrows():
                f.write(f"  {row['parameter']}: range={row['range']}\n")
        
        f.write("\n\n")
        
        # Add optimization process summary
        if timeline is not None:
            f.write("== Optimization Timeline Summary ==\n")
            total_time_minutes = timeline['elapsed_minutes'].max()
            f.write(f"Total optimization time: {total_time_minutes:.1f} minutes\n")
            
            # Time to best solution for each model
            for model_type in ['LightGBM', 'CatBoost', 'Meta-Model']:
                model_data = timeline[timeline['model_type'] == model_type]
                if not model_data.empty:
                    best_idx = model_data['value'].idxmin()
                    best_time = model_data.loc[best_idx, 'elapsed_minutes']
                    f.write(f"Time to best {model_type} solution: {best_time:.1f} minutes\n")
        
        f.write("\n=== End of Summary ===\n")
    
    logging.info("Comprehensive hyperparameter analysis completed")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define Optuna storage and study name
    OPTUNA_STUDY_NAME = "experiment_2"
    PG_USER = os.environ.get("POSTGRES_USER", "postgres")
    PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
    PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
    PG_DB = os.environ.get("POSTGRES_DB", "optuna")
    PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    
    # Run comprehensive analysis
    run_comprehensive_analysis(OPTUNA_STUDY_NAME, OPTUNA_DB)

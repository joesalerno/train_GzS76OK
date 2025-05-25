"""
Copy trials from one Optuna study to another.
This script transfers trial data from a source Optuna database to a target database.
"""
try:
    import optuna
except ImportError:
    print("Optuna not found. Please install with: pip install optuna psycopg2-binary tqdm")
    exit(1)
    
import logging
import sys
import os
import argparse
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_trials(source_storage, target_storage, study_name, n_trials=None, offset=0, copy_if_exists=False):
    """
    Copy trials from one study to another.
    
    Args:
        source_storage (str): Source storage URL
        target_storage (str): Target storage URL
        study_name (str): Study name to copy
        n_trials (int, optional): Number of trials to copy (None = all)
        offset (int): Skip first n trials
        copy_if_exists (bool): Whether to copy trials if study already exists
    """
    # Load source study
    try:
        source_study = optuna.load_study(study_name=study_name, storage=source_storage)
        logging.info(f"Source study '{study_name}' loaded with {len(source_study.trials)} trials")
    except Exception as e:
        logging.error(f"Error loading source study: {e}")
        return
    
    # Check if trials exist
    if len(source_study.trials) == 0:
        logging.warning("Source study has no trials. Nothing to copy.")
        return
    
    # Create or load target study
    try:
        try:
            # First try to create a new study
            if source_study.directions:
                # Multi-objective study
                directions = source_study.directions
                target_study = optuna.create_study(
                    study_name=study_name,
                    storage=target_storage,
                    directions=[d.name for d in directions],
                    load_if_exists=copy_if_exists
                )
            else:
                # Single-objective study
                target_study = optuna.create_study(
                    study_name=study_name,
                    storage=target_storage,
                    direction=source_study.direction.name,
                    load_if_exists=copy_if_exists
                )
            logging.info(f"Created new target study '{study_name}'")
        except optuna.exceptions.DuplicatedStudyError:
            if not copy_if_exists:
                logging.error(f"Target study '{study_name}' already exists. Use --copy-if-exists to copy anyway.")
                return
            # Load existing study if allowed
            target_study = optuna.load_study(study_name=study_name, storage=target_storage)
            logging.info(f"Target study '{study_name}' already exists with {len(target_study.trials)} trials")
    except Exception as e:
        logging.error(f"Error creating/loading target study: {e}")
        return
    
    # Calculate trials to copy
    trials = source_study.trials
    if offset > 0:
        trials = trials[offset:]
    if n_trials is not None:
        trials = trials[:n_trials]
    
    total_to_copy = len(trials)
    if total_to_copy == 0:
        logging.warning("No trials to copy after applying offset/limit.")
        return
        
    logging.info(f"Copying {total_to_copy} trials")
    
    # Copy trials
    copied_count = 0
    for trial in tqdm(trials):
        try:
            # Create a new trial in the target study
            if trial.state.is_finished():
                if len(source_study.directions) > 1:
                    # Multi-objective study
                    target_trial = target_study.add_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        values=trial.values,
                        user_attrs=trial.user_attrs
                    )
                else:
                    # Single-objective study
                    target_trial = target_study.add_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        value=trial.value,
                        user_attrs=trial.user_attrs
                    )
                copied_count += 1
        except Exception as e:
            logging.error(f"Error copying trial {trial.number}: {e}")
    
    logging.info(f"Successfully copied {copied_count} of {total_to_copy} trials")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy Optuna trials between databases")
    parser.add_argument("--source", required=True, help="Source database URL")
    parser.add_argument("--target", required=True, help="Target database URL")
    parser.add_argument("--study-name", required=True, help="Study name to copy")
    parser.add_argument("--n-trials", type=int, help="Number of trials to copy (default: all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first n trials")
    parser.add_argument("--copy-if-exists", action="store_true", help="Copy even if study exists in target")
    
    args = parser.parse_args()
    
    copy_trials(
        source_storage=args.source,
        target_storage=args.target,
        study_name=args.study_name,
        n_trials=args.n_trials,
        offset=args.offset,
        copy_if_exists=args.copy_if_exists
    )

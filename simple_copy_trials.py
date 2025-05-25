"""
Simple script to copy trials from SQLite database to PostgreSQL
"""
try:
    import optuna
except ImportError:
    print("Error: Optuna not found. Installing required packages...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "optuna", "psycopg2-binary", "tqdm"])
        import optuna
        print("Successfully installed packages")
    except Exception as e:
        print(f"Failed to install packages: {e}")
        exit(1)

import logging
import os
from tqdm import tqdm

# Study configuration
SOURCE_DB = "sqlite:///optuna_study_multi_objective_lgbm_tuning.db"

# Get PostgreSQL credentials from environment variables or use defaults
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")

# Construct the target database URL
TARGET_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
STUDY_NAME = "multi_objective_lgbm_tuning"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_trials():
    """Copy all trials from source to target database"""
    # Load source study
    try:
        source_study = optuna.load_study(study_name=STUDY_NAME, storage=SOURCE_DB)
        logging.info(f"Source study '{STUDY_NAME}' loaded with {len(source_study.trials)} trials")
    except Exception as e:
        logging.error(f"Error loading source study: {e}")
        return
    
    # Check if trials exist
    if len(source_study.trials) == 0:
        logging.warning("Source study has no trials. Nothing to copy.")
        return
    
    # Create or load target study
    try:
        # Check if study exists in target
        try:
            target_study = optuna.load_study(study_name=STUDY_NAME, storage=TARGET_DB)
            logging.info(f"Target study already exists with {len(target_study.trials)} trials")
            should_continue = input("Target study already exists. Continue copying? (y/n): ").lower() == 'y'
            if not should_continue:
                logging.info("Operation cancelled by user.")
                return
        except:
            # Create new study
            if hasattr(source_study, 'directions') and source_study.directions:
                # Multi-objective study
                target_study = optuna.create_study(
                    study_name=STUDY_NAME,
                    storage=TARGET_DB,
                    directions=[d.name for d in source_study.directions]
                )
            else:
                # Single-objective study
                target_study = optuna.create_study(
                    study_name=STUDY_NAME,
                    storage=TARGET_DB,
                    direction=source_study.direction.name
                )
            logging.info(f"Created new target study '{STUDY_NAME}'")
    except Exception as e:
        logging.error(f"Error creating/loading target study: {e}")
        return
    
    # Calculate trials to copy
    trials = source_study.trials
    total_to_copy = len(trials)
    
    logging.info(f"Copying {total_to_copy} trials")
    
    # Copy trials
    copied_count = 0
    for trial in tqdm(trials):
        try:
            # Skip already finished trials in target
            existing_trial_numbers = {t.number for t in target_study.trials}
            if trial.number in existing_trial_numbers:
                logging.info(f"Trial {trial.number} already exists in target, skipping")
                continue
                
            # Create a new trial in the target study
            if trial.state.is_finished():
                if hasattr(source_study, 'directions') and len(source_study.directions) > 1:
                    # Multi-objective study
                    target_study.add_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        values=trial.values,
                        user_attrs=trial.user_attrs
                    )
                else:
                    # Single-objective study
                    target_study.add_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        value=trial.value,
                        user_attrs=trial.user_attrs
                    )
                copied_count += 1
                if copied_count % 10 == 0:
                    logging.info(f"Copied {copied_count}/{total_to_copy} trials")
        except Exception as e:
            logging.error(f"Error copying trial {trial.number}: {e}")
    
    logging.info(f"Successfully copied {copied_count} of {total_to_copy} trials")

if __name__ == "__main__":
    print("Starting trial migration from SQLite to PostgreSQL")
    print(f"Source: {SOURCE_DB}")
    print(f"Target: {TARGET_DB}")
    print(f"Study: {STUDY_NAME}")
    
    user_confirm = input("Continue with migration? (y/n): ").lower()
    if user_confirm == 'y':
        copy_trials()
    else:
        print("Migration cancelled")

import optuna
import pandas as pd
import os

# Update these if your study name or DB path is different
OPTUNA_STUDY_NAME = "recursive_lgbm_tuning"

# Get PostgreSQL credentials from environment variables or use defaults
PG_USER = os.environ.get("POSTGRES_USER", "neondb_owner")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "npg_b9Jo7RhUgpSd")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "neondb")

# Construct the database URL with SSL mode
OPTUNA_DB = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=require"

# Load the study
study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)

# Get the trials dataframe
df_trials = study.trials_dataframe()
value_cols = [col for col in df_trials.columns if col.startswith('values_')]

if value_cols:
    print("Objective statistics:")
    for i, col in enumerate(value_cols):
        vals = df_trials[col]
        print(f"Objective {i} ({col}): min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}")
else:
    print("No objective values found in the study.")
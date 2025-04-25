import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
SEED = 42
LAG_WEEKS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10]

# --- Load data ---
df = pd.read_csv(DATA_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
test = pd.read_csv(TEST_PATH).sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
meal_info = pd.read_csv(MEAL_INFO_PATH)
df = df.merge(meal_info, on="meal_id", how="left")
test = test.merge(meal_info, on="meal_id", how="left")
center_info = pd.read_csv(CENTER_INFO_PATH)
df = df.merge(center_info, on="center_id", how="left")
test = test.merge(center_info, on="center_id", how="left")

# --- Feature engineering (old + interactions) ---
def create_features(df):
    df = df.copy()
    group = ["center_id", "meal_id"]
    for lag in LAG_WEEKS:
        df[f"num_orders_lag_{lag}"] = df.groupby(group)["num_orders"].shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = df.groupby(group)["num_orders"].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean().reset_index(0, drop=True)
        df[f"rolling_std_{window}"] = shifted.rolling(window).std().reset_index(0, drop=True)
    df["discount"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = df["discount"] / df["base_price"]
    df["price_diff"] = df.groupby(group)["checkout_price"].diff()
    for col in ["emailer_for_promotion", "homepage_featured"]:
        shifted = df.groupby(group)[col].shift(1)
        df[f"{col}_rolling_sum_3"] = shifted.rolling(3).sum().reset_index(0, drop=True)
    df["weekofyear"] = df["week"] % 52
    # --- Interaction features ---
    df["price_diff_x_emailer"] = df["price_diff"] * df["emailer_for_promotion"]
    df["lag1_x_emailer"] = df["num_orders_lag_1"] * df["emailer_for_promotion"]
    df["price_diff_x_home"] = df["price_diff"] * df["homepage_featured"]
    df["lag1_x_home"] = df["num_orders_lag_1"] * df["homepage_featured"]
    # Fill NaNs only for columns that exist
    lag_roll_cols = [col for col in df.columns if any(x in col for x in ["lag", "rolling", "diff", "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"])]
    lag_roll_cols = [col for col in lag_roll_cols if col in df.columns]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    return df

df = create_features(df)
df = df[df["num_orders"].notna()].reset_index(drop=True)

# --- One-hot encoding ---
cat_cols = []
if "category" in df.columns:
    cat_cols.append("category")
if "cuisine" in df.columns:
    cat_cols.append("cuisine")
if "center_type" in df.columns:
    cat_cols.append("center_type")
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols)
    test = pd.get_dummies(test, columns=cat_cols)
df, test = df.align(test, join="left", axis=1, fill_value=0)

# --- Feature list (old + interactions) ---
FEATURES = [
    "center_id", "meal_id", "checkout_price", "base_price",
    "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff", "weekofyear"
]
FEATURES += [f"num_orders_lag_{lag}" for lag in LAG_WEEKS]
FEATURES += [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
FEATURES += [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
FEATURES += [f"{col}_rolling_sum_3" for col in ["emailer_for_promotion", "homepage_featured"]]
FEATURES += ["price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"]
FEATURES += [col for col in df.columns if any(col.startswith(prefix) for prefix in ["category_", "cuisine_", "center_type_"])]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET = "num_orders"

# --- Train/validation split (last 8 weeks for validation) ---
max_week = df["week"].max()
valid_df = df[df["week"] > max_week - 8].copy()
train_df = df[df["week"] <= max_week - 8].copy()

# --- RMSLE metric ---
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred.clip(0)) - np.log1p(y_true.clip(0)))))

def lgb_rmsle(y_true, y_pred):
    rmsle_score = rmsle(y_true, y_pred)
    return 'rmsle', rmsle_score, False

# --- Model training function ---
def get_lgbm(params=None):
    default_params = dict(
        objective="regression",
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_child_samples=20,
        lambda_l1=0.1,
        lambda_l2=0.1,
        max_depth=5,
        random_state=SEED,
        verbose=-1
    )
    if params:
        default_params.update(params)
    return LGBMRegressor(**default_params)

# --- Optimized cross-validation (GroupKFold) ---
def cross_val_score_groupkfold(df, features, target, group_col, get_lgbm, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, valid_idx in gkf.split(df, groups=df[group_col]):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        model = get_lgbm()
        model.fit(train_df[features], train_df[target], eval_set=[(valid_df[features], valid_df[target])], eval_metric=lgb_rmsle)
        preds = model.predict(valid_df[features])
        score = rmsle(valid_df[target], preds)
        scores.append(score)
    print(f"GroupKFold CV RMSLE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return scores

# --- Helper functions: place all before main pipeline ---
def greedy_feature_selection(train_df, valid_df, FEATURES, TARGET, get_lgbm):
    best_features = FEATURES.copy()
    best_rmsle = float('inf')
    improved = True
    while improved and len(best_features) > 1:
        improved = False
        scores = []
        for f in best_features:
            trial_features = [x for x in best_features if x != f]
            model = get_lgbm()
            model.fit(
                train_df[trial_features], train_df[TARGET],
                eval_set=[(valid_df[trial_features], valid_df[TARGET])],
                eval_metric=lgb_rmsle
            )
            preds = model.predict(valid_df[trial_features])
            score = rmsle(valid_df[TARGET], preds)
            scores.append((score, f))
        scores.sort()
        if scores[0][0] < best_rmsle:
            best_rmsle = scores[0][0]
            best_features.remove(scores[0][1])
            improved = True
            print(f"Removed {scores[0][1]}, new best RMSLE: {best_rmsle:.5f}")
    return best_features, best_rmsle

def make_test_features(test, train_hist):
    test_feat = test.copy()
    for lag in LAG_WEEKS:
        test_feat[f"num_orders_lag_{lag}"] = np.nan
    for window in ROLLING_WINDOWS:
        test_feat[f"rolling_mean_{window}"] = np.nan
        test_feat[f"rolling_std_{window}"] = np.nan
    test_feat["price_diff"] = np.nan
    for col in ["emailer_for_promotion", "homepage_featured"]:
        test_feat[f"{col}_rolling_sum_3"] = np.nan
    test_feat["weekofyear"] = test_feat["week"] % 52
    test_feat["price_diff_x_emailer"] = np.nan
    test_feat["lag1_x_emailer"] = np.nan
    test_feat["price_diff_x_home"] = np.nan
    test_feat["lag1_x_home"] = np.nan
    for idx, row in test_feat.iterrows():
        cid, mid, week = row["center_id"], row["meal_id"], row["week"]
        hist = train_hist[(train_hist["center_id"] == cid) & (train_hist["meal_id"] == mid) & (train_hist["week"] < week)].sort_values("week")
        for lag in LAG_WEEKS:
            test_feat.at[idx, f"num_orders_lag_{lag}"] = hist["num_orders"].iloc[-lag] if len(hist) >= lag else 0
        for window in ROLLING_WINDOWS:
            vals = hist["num_orders"].iloc[-window:] if len(hist) >= window else hist["num_orders"]
            test_feat.at[idx, f"rolling_mean_{window}"] = vals.mean() if len(vals) > 0 else 0
            test_feat.at[idx, f"rolling_std_{window}"] = vals.std() if len(vals) > 1 else 0
        test_feat.at[idx, "price_diff"] = row["checkout_price"] - hist["checkout_price"].iloc[-1] if len(hist) >= 1 else 0
        for col in ["emailer_for_promotion", "homepage_featured"]:
            vals = hist[col].iloc[-3:] if len(hist) >= 3 else hist[col]
            test_feat.at[idx, f"{col}_rolling_sum_3"] = vals.sum() if len(vals) > 0 else 0
        test_feat.at[idx, "price_diff_x_emailer"] = test_feat.at[idx, "price_diff"] * row["emailer_for_promotion"]
        test_feat.at[idx, "lag1_x_emailer"] = test_feat.at[idx, "num_orders_lag_1"] * row["emailer_for_promotion"]
        test_feat.at[idx, "price_diff_x_home"] = test_feat.at[idx, "price_diff"] * row["homepage_featured"]
        test_feat.at[idx, "lag1_x_home"] = test_feat.at[idx, "num_orders_lag_1"] * row["homepage_featured"]
    test_feat["discount"] = test_feat["base_price"] - test_feat["checkout_price"]
    test_feat["discount_pct"] = test_feat["discount"] / test_feat["base_price"]
    cat_cols = []
    for col in ["category", "cuisine", "center_type"]:
        if col in test_feat.columns:
            cat_cols.append(col)
    if cat_cols:
        test_feat = pd.get_dummies(test_feat, columns=cat_cols)
    test_feat, _ = test_feat.align(train_hist, join="right", axis=1, fill_value=0)
    lag_roll_cols = [col for col in test_feat.columns if any(x in col for x in ["lag", "rolling", "diff", "price_diff_x_emailer", "lag1_x_emailer", "price_diff_x_home", "lag1_x_home"])]
    test_feat[lag_roll_cols] = test_feat[lag_roll_cols].fillna(0)
    return test_feat

def cross_val_feature_selection_and_offsets(df, FEATURES, TARGET, group_cols, get_lgbm, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    from collections import defaultdict
    all_val_idx = []
    oof_preds = np.zeros(len(df))
    offsets_dict = defaultdict(list)
    best_features = FEATURES.copy()
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(df, groups=df[group_cols[0]])):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        fold_features, _ = greedy_feature_selection(train_df, valid_df, best_features, TARGET, get_lgbm)
        best_features = list(set(best_features) & set(fold_features))
    print(f"CV-selected features: {best_features}")
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(df, groups=df[group_cols[0]])):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        model = get_lgbm()
        model.fit(train_df[best_features], train_df[TARGET], eval_set=[(valid_df[best_features], valid_df[TARGET])], eval_metric=lgb_rmsle)
        preds = model.predict(valid_df[best_features])
        oof_preds[valid_idx] = preds
        val_df = valid_df.copy()
        val_df['preds'] = preds
        for group, group_df in val_df.groupby(group_cols):
            y_true = group_df[TARGET]
            y_pred = group_df['preds']
            offsets = np.linspace(-20, 20, 201)
            rmsle_scores = [rmsle(y_true, y_pred + o) for o in offsets]
            best_offset = offsets[np.argmin(rmsle_scores)]
            offsets_dict[group].append(best_offset)
    group_offsets = {k: np.mean(v) for k, v in offsets_dict.items()}
    global_offset = np.mean(list(group_offsets.values())) if group_offsets else 0
    return best_features, group_offsets, global_offset

def apply_group_offsets_with_fallback(model, test_feat, features, group_cols, group_offsets, global_offset):
    preds = model.predict(test_feat[features])
    offsets = np.zeros(len(test_feat))
    group_keys = test_feat[group_cols].apply(tuple, axis=1)
    for i, key in enumerate(group_keys):
        offsets[i] = group_offsets.get(key, global_offset)
    return np.clip(preds + offsets, 0, None).round().astype(int)

# --- Main pipeline ---
if __name__ == "__main__":
    best_features, group_offsets, global_offset = cross_val_feature_selection_and_offsets(
        df, FEATURES, TARGET, ["meal_id", "center_id"], get_lgbm, n_splits=5)
    full_df = df.copy()
    final_model = get_lgbm()
    final_model.fit(full_df[best_features], full_df[TARGET], eval_metric=lgb_rmsle)
    test_feat = make_test_features(test, full_df)
    test_feat["num_orders"] = apply_group_offsets_with_fallback(final_model, test_feat, best_features, ["meal_id", "center_id"], group_offsets, global_offset)
    submission = test_feat[["id", "num_orders"]].copy()
    submission["id"] = submission["id"].astype(int)
    submission.to_csv("submission_hybrid_groupoffset.csv", index=False)
    print("submission_hybrid_groupoffset.csv saved.")

    # --- SHAP analysis and Optuna tuning (use new final_model, best_features, test_feat) ---
    print("Calculating SHAP values for hybrid model...")
    shap_sample = full_df[best_features].sample(n=min(2000, len(full_df)), random_state=SEED)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample)
    shap_df = pd.DataFrame(shap_values, columns=best_features)
    shap_df.to_csv("shap_hybrid_values.csv", index=False)
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': best_features,
        'mean_abs_shap': shap_importance
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv("shap_hybrid_feature_importances.csv", index=False)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_sample, feature_names=best_features, show=False)
    plt.tight_layout()
    plt.savefig("shap_hybrid_summary.png")
    plt.close()
    plt.figure(figsize=(10, 6))
    shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
    plt.title('Top 20 SHAP Feature Importances (Hybrid Model)')
    plt.ylabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig('shap_hybrid_top20.png')
    plt.close()
    print("SHAP analysis for hybrid model saved.")

    # --- Optuna tuning for further improvement (with feature selection, resume support) ---
    import lightgbm as lgb
    import os

    OPTUNA_DB = "sqlite:///optuna_hybrid_optuna.db"
    OPTUNA_STUDY_NAME = "hybrid_optuna_feature_selection"
    all_candidate_features = best_features.copy()

    def objective(trial):
        # Feature subset selection
        selected = [f for f in all_candidate_features if trial.suggest_categorical(f'use_{f}', [True, False])]
        if not selected:
            return 1e6  # Avoid empty feature set
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
        }
        model = get_lgbm(params)
        model.fit(
            train_df[selected], train_df[TARGET],
            eval_set=[(valid_df[selected], valid_df[TARGET])],
            eval_metric=lgb_rmsle
        )
        preds = model.predict(valid_df[selected])
        score = rmsle(valid_df[TARGET], preds)
        return score

    # Resume or create Optuna study
    try:
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
        print(f"Resuming Optuna study '{OPTUNA_STUDY_NAME}'...")
    except Exception:
        study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB)
        print(f"Created new Optuna study '{OPTUNA_STUDY_NAME}'...")

    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    # Extract best feature subset from best trial
    best_trial = study.best_trial
    best_selected_features = [f for f in all_candidate_features if best_trial.params.get(f'use_{f}', True)]
    print("Best Optuna params:", best_params)
    print("Best Optuna feature subset:", best_selected_features)
    # Save Optuna trials to CSV for review
    study.trials_dataframe().to_csv("optuna_hybrid_optuna_trials.csv", index=False)

    # Retrain with best params and best feature subset on full data
    final_model = get_lgbm(best_params)
    final_model.fit(full_df[best_selected_features], full_df[TARGET], eval_metric=lgb_rmsle)
    test_feat["num_orders"] = apply_group_offsets_with_fallback(final_model, test_feat, best_selected_features, ["meal_id", "center_id"], group_offsets, global_offset)
    submission = test_feat[["id", "num_orders"]].copy()
    submission["id"] = submission["id"].astype(int)
    submission.to_csv("submission_hybrid_optuna.csv", index=False)
    print("submission_hybrid_optuna.csv saved.")
    # SHAP for Optuna-tuned model
    shap_sample = full_df[best_selected_features].sample(n=min(2000, len(full_df)), random_state=SEED)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample)
    shap_df = pd.DataFrame(shap_values, columns=best_selected_features)
    shap_df.to_csv("shap_hybrid_optuna_values.csv", index=False)
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': best_selected_features,
        'mean_abs_shap': shap_importance
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv("shap_hybrid_optuna_feature_importances.csv", index=False)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_sample, feature_names=best_selected_features, show=False)
    plt.tight_layout()
    plt.savefig("shap_hybrid_optuna_summary.png")
    plt.close()
    plt.figure(figsize=(10, 6))
    shap_importance_df.head(20).plot.bar(x='feature', y='mean_abs_shap', legend=False)
    plt.title('Top 20 SHAP Feature Importances (Hybrid Optuna Model)')
    plt.ylabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig('shap_hybrid_optuna_top20.png')
    plt.close()
    print("SHAP analysis for Optuna-tuned hybrid model saved.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, ShuffleSplit, StratifiedShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, LeaveOneOut, PredefinedSplit, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_log_error
import warnings
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load data and features from test.py context ---
DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"
OUTPUT_DIRECTORY = "output"
SEED = 45
TARGET = "num_orders"
# --- CV strategies ---
N_SPLITS = 5
VAL_WINDOW = 10
MIN_TRAIN_WEEKS = 18

# Load data
try:
    df = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    meal_info = pd.read_csv(MEAL_INFO_PATH)
    center_info = pd.read_csv(CENTER_INFO_PATH)
except FileNotFoundError as e:
    logging.error(f"Error loading data file: {e}.")
    raise

def preprocess_data(df, meal_info, center_info):
    df = df.merge(meal_info, on="meal_id", how="left")
    df = df.merge(center_info, on="center_id", how="left")
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    return df

df = preprocess_data(df, meal_info, center_info)

# Use a simple feature set for demonstration (can be replaced with your FEATURES list)
FEATURES = [
    "checkout_price", "base_price", "homepage_featured", "emailer_for_promotion",
    "discount", "discount_pct", "price_diff",
    "center_id", "meal_id"
]
for f in FEATURES:
    if f not in df.columns:
        FEATURES.remove(f)


# Add discount features if not present
if "discount" not in df.columns:
    df["discount"] = df["base_price"] - df["checkout_price"]
if "discount_pct" not in df.columns:
    df["discount_pct"] = df["discount"] / df["base_price"].replace(0, np.nan)
if "price_diff" not in df.columns:
    df["price_diff"] = df.groupby(["center_id", "meal_id"])["checkout_price"].diff().fillna(0)

# Fill missing values
for f in FEATURES:
    if f in df.columns:
        df[f] = df[f].fillna(0)

# Remove rows with missing target
train_df = df[df[TARGET].notnull()].copy()

# Helper metric
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# --- ExpandingGroupTimeSeriesSplit implementation ---
class ExpandingGroupTimeSeriesSplit:
    """
    Expanding window cross-validator for time series data with group awareness.
    For each split, the training set starts at the beginning and expands,
    the validation set is the next val_window unique weeks.
    """
    def __init__(self, n_splits=3, min_train_window=20, val_window=4, week_col='week'):
        self.n_splits = n_splits
        self.min_train_window = min_train_window
        self.val_window = val_window
        self.week_col = week_col

    def split(self, X, y=None, groups=None):
        weeks = np.sort(X[self.week_col].unique())
        total_weeks = len(weeks)
        max_start = total_weeks - self.min_train_window - self.val_window + 1
        if self.n_splits > max_start:
            raise ValueError(f"Not enough weeks for {self.n_splits} splits with min_train_window={self.min_train_window} and val_window={self.val_window}.")
        for i in range(self.n_splits):
            train_end = self.min_train_window + i * (max_start // self.n_splits)
            val_start = train_end
            val_end = val_start + self.val_window
            train_weeks = weeks[:train_end]
            val_weeks = weeks[val_start:val_end]
            train_mask = X[self.week_col].isin(train_weeks)
            val_mask = X[self.week_col].isin(val_weeks)
            train_indices = np.where(train_mask & pd.notnull(groups))[0]
            val_indices = np.where(val_mask & pd.notnull(groups))[0]
            yield train_indices, val_indices


# --- RollingGroupTimeSeriesSplit implementation ---
class RollingGroupTimeSeriesSplit:
    """
    Rolling window cross-validator for time series data with group awareness.
    For each split, the training set is a rolling window of train_window unique weeks,
    and the validation set is the next val_window unique weeks.
    Groups are respected (e.g., center_id, meal_id).
    No gap is used between train and validation.
    """
    def __init__(self, n_splits=3, train_window=80, val_window=10, week_col='week'):
        self.n_splits = n_splits
        self.train_window = train_window
        self.val_window = val_window
        self.week_col = week_col

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Group labels must be provided for RollingGroupTimeSeriesSplit.")
        weeks = np.sort(X[self.week_col].unique())
        total_weeks = len(weeks)
        max_start = total_weeks - self.train_window - self.val_window + 1
        if self.n_splits > max_start:
            raise ValueError(f"Not enough weeks for {self.n_splits} splits with train_window={self.train_window} and val_window={self.val_window}.")
        for i in range(self.n_splits):
            train_start = i * (max_start // self.n_splits)
            train_end = train_start + self.train_window
            val_start = train_end
            val_end = val_start + self.val_window
            train_weeks = weeks[train_start:train_end]
            val_weeks = weeks[val_start:val_end]
            train_mask = X[self.week_col].isin(train_weeks)
            val_mask = X[self.week_col].isin(val_weeks)
            train_indices = np.where(train_mask & pd.notnull(groups))[0]
            val_indices = np.where(val_mask & pd.notnull(groups))[0]
            yield train_indices, val_indices

# --- RecursiveGroupTimeSeriesSplit implementation ---



class ShrinkingGroupTimeSeriesSplit:
    """
    Runs group-aware time series CV: for n in 1..max_splits, splits the data into n contiguous time segments and yields expanding window splits for each segment.
    """
    def __init__(self, min_weeks=12, max_splits=3, val_window=10, week_col='week', group_col='center_id'):
        self.min_weeks = min_weeks
        self.max_splits = max_splits
        self.val_window = val_window
        self.week_col = week_col
        self.group_col = group_col

    def split(self, X, y=None, groups=None):
        weeks = np.sort(X[self.week_col].unique())
        n_weeks = len(weeks)
        for n_splits in range(1, self.max_splits + 1):
            split_indices = np.linspace(0, n_weeks, n_splits + 1, dtype=int)
            for i in range(n_splits):
                seg_weeks = weeks[split_indices[i]:split_indices[i+1]]
                if len(seg_weeks) < self.min_weeks:
                    continue
                seg_X = X[X[self.week_col].isin(seg_weeks)]
                # Always use all but the last val_window weeks for training, last val_window for validation
                if len(seg_weeks) < self.min_weeks + self.val_window:
                    continue
                train_weeks = seg_weeks[:-self.val_window]
                val_weeks = seg_weeks[-self.val_window:]
                train_mask = seg_X[self.week_col].isin(train_weeks)
                val_mask = seg_X[self.week_col].isin(val_weeks)
                train_indices = np.where(train_mask & pd.notnull(seg_X[self.group_col]))[0]
                val_indices = np.where(val_mask & pd.notnull(seg_X[self.group_col]))[0]
                yield seg_X.index[train_indices], seg_X.index[val_indices]


try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False


cv_strategies = {
    "ExpandingGroupTimeSeriesSplit": ExpandingGroupTimeSeriesSplit(n_splits=N_SPLITS, min_train_window=MIN_TRAIN_WEEKS, val_window=VAL_WINDOW, week_col='week'),
    "RollingGroupTimeSeriesSplit": RollingGroupTimeSeriesSplit(n_splits=N_SPLITS, train_window=MIN_TRAIN_WEEKS, val_window=VAL_WINDOW, week_col='week'),
    "ShrinkingGroupTimeSeriesSplit": ShrinkingGroupTimeSeriesSplit(min_weeks=MIN_TRAIN_WEEKS, max_splits=N_SPLITS, val_window=VAL_WINDOW, week_col='week', group_col='center_id'),
    # "TimeSeriesSplit": TimeSeriesSplit(n_splits=N_SPLITS),
    #
    # "KFold": KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
    # "StratifiedKFold": StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
    # "GroupKFold": GroupKFold(n_splits=N_SPLITS),
    # "ShuffleSplit": ShuffleSplit(n_splits=N_SPLITS, random_state=SEED),
    # "StratifiedShuffleSplit": StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=SEED),
    # "LeaveOneGroupOut": LeaveOneGroupOut(),
    # "RepeatedKFold": RepeatedKFold(n_splits=N_SPLITS, n_repeats=2, random_state=SEED),
    # "RepeatedStratifiedKFold": RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=2, random_state=SEED),
}
# if HAS_ITERSTRAT:
    # cv_strategies["MultilabelStratifiedKFold"] = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    # cv_strategies["MultilabelStratifiedShuffleSplit"] = MultilabelStratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=SEED)

# For stratification, use binned target
train_df["target_bin"] = pd.qcut(train_df[TARGET], q=3, labels=False, duplicates="drop")

results = []


for name, cv in cv_strategies.items():
    logging.info(f"Evaluating {name}...")
    try:
        # Ensure 'week' column is present for time series splits
        if name in ["ExpandingGroupTimeSeriesSplit", "RollingGroupTimeSeriesSplit", "ShrinkingGroupTimeSeriesSplit"]:
            if 'week' not in train_df.columns:
                raise ValueError("'week' column is missing from train_df, required for time series splits.")
        if name == "StratifiedKFold":
            split = cv.split(train_df[FEATURES], train_df["target_bin"])
        elif name == "GroupKFold":
            split = cv.split(train_df[FEATURES], train_df[TARGET], groups=train_df["center_id"])
        elif name == "StratifiedShuffleSplit":
            split = cv.split(train_df[FEATURES], train_df["target_bin"])
        elif name == "LeaveOneGroupOut":
            split = cv.split(train_df[FEATURES], train_df[TARGET], groups=train_df["center_id"])
        elif name == "RollingGroupTimeSeriesSplit":
            split = cv.split(train_df, groups=train_df["center_id"])
        elif name == "ExpandingGroupTimeSeriesSplit":
            split = cv.split(train_df, groups=train_df["center_id"])
        elif name == "ShrinkingGroupTimeSeriesSplit":
            split = cv.split(train_df, groups=train_df["center_id"])
        elif name == "RepeatedStratifiedKFold":
            split = cv.split(train_df[FEATURES], train_df["target_bin"])
        elif name == "RecursiveGroupTimeSeriesSplit":
            split = cv.split(train_df, groups=train_df["center_id"])
        elif name == "MultilabelStratifiedKFold" and HAS_ITERSTRAT:
            # For demo, use binary encoding of target_bin as multilabel
            multilabel = np.zeros((len(train_df), 3))
            for i in range(3):
                multilabel[:, i] = (train_df["target_bin"] == i).astype(int)
            split = cv.split(train_df[FEATURES], multilabel)
        elif name == "MultilabelStratifiedShuffleSplit" and HAS_ITERSTRAT:
            multilabel = np.zeros((len(train_df), 3))
            for i in range(3):
                multilabel[:, i] = (train_df["target_bin"] == i).astype(int)
            split = cv.split(train_df[FEATURES], multilabel)
        else:
            split = cv.split(train_df[FEATURES], train_df[TARGET])
        fold = 0
        for train_idx, valid_idx in split:
            fold += 1
            X_train, X_valid = train_df.iloc[train_idx][FEATURES], train_df.iloc[valid_idx][FEATURES]
            y_train, y_valid = train_df.iloc[train_idx][TARGET], train_df.iloc[valid_idx][TARGET]
            model = LGBMRegressor(n_estimators=600, random_state=SEED, force_row_wise=True)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)
            train_rmsle = rmsle(y_train, y_train_pred)
            valid_rmsle = rmsle(y_valid, y_valid_pred)
            results.append({
                "strategy": name,
                "fold": fold,
                "train_rmsle": train_rmsle,
                "valid_rmsle": valid_rmsle,
            })
    except Exception as e:
        logging.warning(f"Skipping {name} due to error: {e}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_csv = os.path.join(OUTPUT_DIRECTORY, "cv_strategy_empirical_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"Results saved to {results_csv}")

# Plot
plt.figure(figsize=(10, 6))
import seaborn as sns
sns.boxplot(x="strategy", y="valid_rmsle", data=results_df)
plt.title("Validation RMSLE by Cross-Validation Strategy")
plt.ylabel("Validation RMSLE")
plt.xlabel("CV Strategy")
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIRECTORY, "cv_strategy_empirical_plot.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

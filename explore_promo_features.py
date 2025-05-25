import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('train.csv')

# 1. Persistence/Duration: Lagged effect of promotions on num_orders
print("\n1. Persistence/Duration (lagged effect):")
for feature in ['emailer_for_promotion', 'homepage_featured']:
    for lag in [1, 2, 3]:
        lagged = df.groupby(['center_id', 'meal_id'])[feature].shift(lag)
        corr = df['num_orders'].corr(lagged.fillna(0))
        print(f"  Correlation of num_orders with {feature} lagged by {lag} week(s): {corr:.4f}")

# 2. Promotion Overlap
print("\n2. Promotion Overlap:")
df['promo_overlap'] = df['emailer_for_promotion'] & df['homepage_featured']
overlap_rate = df['promo_overlap'].mean()
print(f"  Fraction of rows with both promotions active: {overlap_rate:.4f}")

# 3. Frequency/Recency
print("\n3. Frequency/Recency:")
for feature in ['emailer_for_promotion', 'homepage_featured']:
    # Rolling sum: align index with original DataFrame
    df[f'{feature}_last_4w'] = (
        df.groupby(['center_id', 'meal_id'])[feature]
        .rolling(4, min_periods=1).sum()
        .reset_index(level=[0,1], drop=True)
    )
    # Weeks since last promotion: use a robust function
    def weeks_since_last(arr):
        arr = np.asarray(arr)
        out = np.full_like(arr, np.nan, dtype=float)
        last = -1
        for i in range(len(arr)):
            if arr[i] == 1:
                last = i
                out[i] = 0
            elif last == -1:
                out[i] = np.nan
            else:
                out[i] = i - last
        return out
    df[f'{feature}_weeks_since'] = (
        df.groupby(['center_id', 'meal_id'])[feature]
        .transform(weeks_since_last)
    )
    print(f"  Mean {feature} frequency in last 4 weeks: {df[f'{feature}_last_4w'].mean():.2f}")
    print(f"  Mean weeks since last {feature}: {pd.Series(df[f'{feature}_weeks_since']).mean():.2f}")

# 4. Seasonality
print("\n4. Seasonality:")
df['weekofyear'] = df['week'] % 52
df['month'] = ((df['week'] - 1) // 4 + 1).astype(int)
weekly_avg = df.groupby('weekofyear')['num_orders'].mean()
monthly_avg = df.groupby('month')['num_orders'].mean()
print(f"  Weekly num_orders mean (first 5):\n{weekly_avg.head()}")
print(f"  Monthly num_orders mean (first 5):\n{monthly_avg.head()}")
weekly_avg.to_csv('seasonality_weekly_avg.csv')
monthly_avg.to_csv('seasonality_monthly_avg.csv')

# 5. Rolling Statistics
print("\n5. Rolling Statistics:")
for feature in ['num_orders', 'emailer_for_promotion', 'homepage_featured']:
    for window in [3, 5]:
        df[f'{feature}_rolling_mean_{window}'] = (
            df.groupby(['center_id', 'meal_id'])[feature]
            .rolling(window, min_periods=1).mean()
            .reset_index(level=[0,1], drop=True)
        )
        df[f'{feature}_rolling_std_{window}'] = (
            df.groupby(['center_id', 'meal_id'])[feature]
            .rolling(window, min_periods=2).std()
            .reset_index(level=[0,1], drop=True)
        )
        print(f"  {feature} rolling mean {window}w mean: {df[f'{feature}_rolling_mean_{window}'].mean():.2f}")

# 6. Target Encoding
print("\n6. Target Encoding:")
center_mean = df.groupby('center_id')['num_orders'].mean()
meal_mean = df.groupby('meal_id')['num_orders'].mean()
combo_mean = df.groupby(['center_id', 'meal_id'])['num_orders'].mean()
print(f"  Center mean (first 5):\n{center_mean.head()}")
print(f"  Meal mean (first 5):\n{meal_mean.head()}")
center_mean.to_csv('center_num_orders_mean.csv')
meal_mean.to_csv('meal_num_orders_mean.csv')
combo_mean.to_csv('center_meal_num_orders_mean.csv')

# 7. Price Sensitivity
print("\n7. Price Sensitivity:")
df['price_diff'] = df['checkout_price'] - df['base_price']
for feature in ['emailer_for_promotion', 'homepage_featured']:
    interaction = df['price_diff'] * df[feature]
    corr = df['num_orders'].corr(interaction)
    print(f"  Correlation of num_orders with price_diff x {feature}: {corr:.4f}")

print("\nAnalysis complete. See CSV files for seasonality and target encoding summaries.")

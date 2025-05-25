import pandas as pd

# Load the data
df = pd.read_csv('train.csv')

features = ['emailer_for_promotion', 'homepage_featured']
results = {}

for feature in features:
    intervals = []
    # Group by center_id and meal_id
    for _, group in df.groupby(['center_id', 'meal_id']):
        # Find weeks where the feature is 1
        weeks = group.loc[group[feature] == 1, 'week'].sort_values().values
        if len(weeks) > 1:
            # Calculate differences between consecutive weeks
            diffs = weeks[1:] - weeks[:-1]
            intervals.extend(diffs)
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        results[feature] = avg_interval
    else:
        results[feature] = None

for feature, avg in results.items():
    if avg is not None:
        print(f"Average interval between appearances for {feature}: {avg:.2f} weeks")
    else:
        print(f"No repeated appearances found for {feature}.")

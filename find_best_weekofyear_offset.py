import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
best_offset = None
best_score = float('inf')
for offset in range(0, 52):
    df['weekofyear_offset'] = (df['week'] + offset) % 52
    # Use only weeks within a single year (1-52)
    mask = (df['weekofyear_offset'] >= 0) & (df['weekofyear_offset'] < 52)
    score = df[mask].groupby('weekofyear_offset')['num_orders'].mean().std()
    print(f'Offset {offset}: std={score:.2f}')
    if score < best_score:
        best_score = score
        best_offset = offset
print(f'Best offset for weekofyear: {best_offset} (std={best_score:.2f})')

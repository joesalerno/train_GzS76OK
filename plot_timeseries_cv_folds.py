import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Load the data
df = pd.read_csv('output/timeseries_cv_comparison_folds_multiseed.csv')

# Set up colors and markers for clarity
method_palette = sns.color_palette("colorblind", n_colors=3)
method_order = ['TimeSeriesSplit', 'GroupTimeSeriesSplit', 'RollingGroupTimeSeriesSplit']
method_map = {m: method_palette[i] for i, m in enumerate(method_order)}
seed_markers = ['o', 's', 'D', '^', 'v']

plt.figure(figsize=(10, 6))

# Plot each (method, seed) as a separate line
for method_idx, method in enumerate(method_order):
    method_df = df[df['cv'] == method]
    for i, seed in enumerate(sorted(method_df['seed'].unique())):
        sub = method_df[method_df['seed'] == seed]
        plt.plot(
            sub['fold'],
            sub['rmsle'],
            marker=seed_markers[i % len(seed_markers)],
            color=method_map[method],
            label=f"{method} (seed={seed})",
            alpha=0.85,
            linewidth=2
        )

plt.xlabel('Fold')
plt.ylabel('RMSLE')
plt.title('Time Series CV Comparison: Fold RMSLEs (Multiple Seeds)')
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0, fontsize=9)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig('output/timeseries_cv_comparison_multiseed_replot.png', bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Approximated data from the image
folds = np.arange(5)
data = {
    'TimeSeriesSplit': {
        13:   [0.535, 0.525, 0.532, 0.505, 0.515],
        123:  [0.540, 0.530, 0.535, 0.510, 0.520],
        1999: [0.533, 0.523, 0.530, 0.500, 0.510],
        2025: [0.537, 0.527, 0.534, 0.507, 0.517],
        9001: [0.534, 0.524, 0.531, 0.502, 0.512],
    },
    'GroupTimeSeriesSplit': {
        13:   [0.520, 0.515, 0.530, 0.510, 0.520],
        123:  [0.522, 0.517, 0.532, 0.512, 0.522],
        1999: [0.519, 0.514, 0.529, 0.509, 0.519],
        2025: [0.521, 0.516, 0.531, 0.511, 0.521],
        9001: [0.518, 0.513, 0.528, 0.508, 0.518],
    },
    'RollingGroupTimeSeriesSplit': {
        13:   [0.510, 0.520, 0.525, 0.515, 0.525],
        123:  [0.512, 0.522, 0.527, 0.517, 0.527],
        1999: [0.509, 0.519, 0.524, 0.514, 0.524],
        2025: [0.511, 0.521, 0.526, 0.516, 0.526],
        9001: [0.508, 0.518, 0.523, 0.513, 0.523],
    }
}

colors = {
    'TimeSeriesSplit': 'tab:blue',
    'GroupTimeSeriesSplit': 'tab:orange',
    'RollingGroupTimeSeriesSplit': 'tab:green'
}
markers = ['o', 's', 'D', '^', 'v']

plt.figure(figsize=(8, 6))
for method_idx, (method, seeds) in enumerate(data.items()):
    for i, (seed, rmsle) in enumerate(seeds.items()):
        plt.plot(folds, rmsle, marker=markers[i], color=colors[method], label=f"{method} (seed={seed})", alpha=0.8)

plt.xlabel('Fold')
plt.ylabel('RMSLE')
plt.title('Time Series CV Comparison: Fold RMSLEs (Multiple Seeds)')
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0, fontsize=9)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig("timeseries_cv_comparison_multiseed_replot.png", bbox_inches='tight')
plt.show()
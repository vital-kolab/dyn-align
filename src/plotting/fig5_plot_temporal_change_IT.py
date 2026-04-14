"""
Temporal Change Analysis for IT (Inferotemporal) Neurons

This script analyzes and visualizes temporal changes in neural correlation patterns
during visual stimulus processing. It processes pre-computed correlation metrics and
their reliability estimates to calculate how pairwise correlations change across time.

The analysis includes:
1. Loading temporal change scores from HDF5 file
2. Computing reliability-corrected correlations across neural populations
3. Identifying adaptive temporal start points for each neuron
4. Computing temporal change as the difference between correlations at different time lags
5. Generating two publication-quality plots:
    - Time series plot of median temporal change with error bands
    - Histogram comparison of early vs. late temporal changes across neurons

Key outputs:
- IT_temporal_change.pdf: Time series of temporal change across stimulus onset
- IT_temporal_change_early-late-histograms.pdf: Distribution of early vs. late temporal changes
"""

import sys
sys.path.append('../')

import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils.correlation_metrics import spearmanbrown_correction
from utils.plot_utils import journal_figure_pdf

import scienceplots

# Set matplotlib style for publication-quality figures
plt.style.use(['nature'])

# Configuration parameters
adaptive_t_start = True
scores_path = '[path to your folder containing temporal change scores]'

# Load temporal change scores from HDF5 file
neurons_h5_file = h5py.File(scores_path, 'r')

corrs = np.array(neurons_h5_file['raw_correlations'])
shcs_nt = np.array(neurons_h5_file['shcs_nt'])
shcs_nt_next = np.array(neurons_h5_file['shcs_nt_next'])

# Calculate per-neuron median SHC if using adaptive time start

# Trim data to 390ms (13 time bins)
corrs = corrs[:, :13, :13, :]
shcs_nt = shcs_nt[:, :13, :13, :]
shcs_nt_next = shcs_nt_next[:, :13, :13, :]


# Define time bins and array dimensions
t_idxs = [3, 4, 5, 6, 7, 8, 9, 10]
n_time_bins = corrs.shape[1]
n_neurons = corrs.shape[-1]

# Compute median across repetitions and correct for reliability
corrs = np.nanmedian(corrs, 0)
shcs_nt = np.nanmedian(shcs_nt, 0)
shcs_nt_next = np.nanmedian(shcs_nt_next, 0)

corrs = corrs / np.sqrt(spearmanbrown_correction(shcs_nt) * spearmanbrown_correction(shcs_nt_next))

# Initialize array to store temporal changes
delta_corrs = np.zeros((n_time_bins, n_neurons)) + np.nan

# Find adaptive time start for each neuron (first time bin with SHC > 0.4)
if adaptive_t_start:
    shc_per_neuron = shcs_nt
    t_idxs_per_neuron = np.zeros(n_neurons) + np.nan
    for n in range(n_neurons):
        for t in t_idxs:
            shc = shc_per_neuron[t, :, n]
            if shc > 0.4:
                t_idxs_per_neuron[n] = t
                break

# Calculate temporal change: difference between 1-bin and bin_range-bin ahead correlations
t_start = t_idxs[0]
end_time_bin_idx = 12
bin_range = 2

for n in range(n_neurons):
    if np.isnan(t_idxs_per_neuron[n]):
        continue
    if adaptive_t_start:
        t_start = int(t_idxs_per_neuron[n])
    for t in range(t_start, end_time_bin_idx + 1):
        if t + bin_range < corrs.shape[0]:
            delta_corrs[t, n] = corrs[t, t + 1, n] - corrs[t, t + bin_range, n]

# Plot 1: Temporal change across time with error bands
plt.figure(figsize=(5, 4))
plt.errorbar(np.arange(delta_corrs[t_start:end_time_bin_idx+1].shape[0]-bin_range), 
             np.nanmedian(delta_corrs[t_start:end_time_bin_idx+1-bin_range], -1), 
             c='tab:purple', capsize=0, linestyle='-')
plt.fill_between(np.arange(delta_corrs[t_start:end_time_bin_idx+1].shape[0]-bin_range), 
                 np.nanmedian(delta_corrs[t_start:end_time_bin_idx+1-bin_range], -1) - (np.nanstd(delta_corrs[t_start:end_time_bin_idx+1-bin_range], -1) / np.sqrt(n_neurons)),
                 np.nanmedian(delta_corrs[t_start:end_time_bin_idx+1-bin_range], -1) + (np.nanstd(delta_corrs[t_start:end_time_bin_idx+1-bin_range], -1) / np.sqrt(n_neurons)),
                 color='tab:purple', alpha=0.5)

# Format x-axis with time labels in milliseconds
start_time = (t_start - 1) * 30
bin_duration = 30
labels = [start_time + ((i + 1) * bin_duration) for i in range(end_time_bin_idx + 1 - bin_range - t_start)]
range_ = list(range(len(labels)))

plt.xticks(range_, labels, fontsize=6)
plt.xlabel("time after stimulus onset (ms)", fontsize=12)
plt.ylabel("temporal change", fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

journal_figure_pdf(do_save=True, filename=f'../../plots/IT_temporal_change.pdf')

# Extract early and late temporal change values for each neuron
curve_params = np.zeros((n_neurons, 2)) + np.nan
for n in range(n_neurons):
    y = delta_corrs[t_start:end_time_bin_idx+1-bin_range, n]
    curve_params[n, 0] = y[0]      # Early temporal change
    curve_params[n, 1] = y[-1]     # Late temporal change

# Plot 2: Histogram comparing early vs. late temporal change distributions
plt.figure(figsize=(4, 5))
plt.hist(curve_params[:, 0], bins=20, color='tab:purple', edgecolor='black', linewidth=2, alpha=1.0, zorder=1)
plt.hist(curve_params[:, -1], bins=20, color='tab:purple', edgecolor='black', linewidth=2, alpha=0.5, zorder=3)

# Add median lines for early temporal change
mean_val = np.nanmedian(curve_params[:, 0])
plt.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'median={np.round(mean_val, 3)}')
plt.plot(mean_val, plt.ylim()[1] * 1.0, marker='v', color='black', markersize=10)

# Add median lines for late temporal change
mean_val = np.nanmedian(curve_params[:, -1])
plt.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'median={np.round(mean_val, 3)}')
plt.plot(mean_val, plt.ylim()[1] * 1.0, marker='v', color='black', markersize=10)

plt.yticks(fontsize=12)
plt.xlabel(f"temporal change", fontsize=12)
plt.ylabel("number of neurons", fontsize=12)

journal_figure_pdf(do_save=True, filename=f'../../plots/IT_temporal_change_early-late-histograms.pdf')

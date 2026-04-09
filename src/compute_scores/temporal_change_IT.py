"""
Temporal Change Analysis - Neural Data Correlation Computation
This script computes temporal correlations between neural responses across time bins,
along with split-half reliability measures for neural data. It performs repetition 
resampling to estimate the stability of correlations between consecutive time periods
in neural recordings.
The analysis:
1. Loads neural rate data from an HDF5 file containing [time bins x videos x sites x repetitions]
2. For each pair of time bins and neural site, computes:
    - Temporal correlation between neural responses at two different time points
    - Split-half reliability (Spearman-Brown corrected) for each time bin
3. Performs repetition resampling (default: 100 iterations) for robust estimation
4. Saves all results (raw correlations and reliability measures) to output HDF5 file
Key parameters:
- num_repetitions: Number of repetition iterations (default: 100)
- correlation_metric: Type of correlation metric ('pearson' or 'spearman')
- Input: Neural rate data with shape [time bins x videos x sites x repetitions]
- Output: Arrays of temporal correlations and split-half correlations for each repetition sample
"""
import sys
sys.path.append('../')

from utils.correlation_metrics import get_splithalf_corr, spearmanbrown_correction
import h5py
import numpy as np
from scipy import stats
import os

# Load neural data from HDF5 file
neurons_file_path = '[your file containing neural recordings for videos]'
neurons_h5_file = h5py.File(neurons_file_path, 'r')
print("Neurons Keys:", list(neurons_h5_file.keys()))
neuron_file_name = neurons_file_path.split('/')[-1].split('.')[0]

# Extract neural rate data: [time bins x videos x sites x repetitions]
neural_rates = np.array(neurons_h5_file['rate_it'])
correlation_metric = 'pearson'  # 'pearson' or 'spearman'
print(f'Neurons: {neural_rates.shape} [time bins x videos x sites x repetitions]')

# Initialize repetition parameters
num_repetitions = 100
np.random.seed(42)
scores_save_path = '../../scores'

# Initialize result arrays
temporal_correlations = np.zeros((num_repetitions, neural_rates.shape[0], neural_rates.shape[0], neural_rates.shape[2])) + np.nan
split_half_corr_t1 = np.zeros((num_repetitions, neural_rates.shape[0], neural_rates.shape[0], neural_rates.shape[2])) + np.nan
split_half_corr_t2 = np.zeros((num_repetitions, neural_rates.shape[0], neural_rates.shape[0], neural_rates.shape[2])) + np.nan

# Compute correlations across time bins and neural sites
for time_idx1 in range(neural_rates.shape[0]):
    for time_idx2 in range(time_idx1 + 1, time_idx1 + 3):
        for site_idx in range(neural_rates.shape[2]):
            for repetition_idx in range(num_repetitions):
                repetition_seed = np.random.randint(0, 10000)
                
                # Time bin 1: compute split-half reliability
                rates_t1 = np.nanmean(neural_rates[time_idx1, :, site_idx], axis=-1)
                shc_t1 = spearmanbrown_correction(get_splithalf_corr(neural_rates[time_idx1, :, site_idx], seed=repetition_seed, type=correlation_metric)['split_half_corr'])
                split_half_corr_t1[repetition_idx, time_idx1, time_idx2, site_idx] = shc_t1

                # Time bin 2: compute split-half reliability
                rates_t2 = np.nanmean(neural_rates[time_idx2, :, site_idx], axis=-1)
                shc_t2 = spearmanbrown_correction(get_splithalf_corr(neural_rates[time_idx2, :, site_idx], seed=repetition_seed, type=correlation_metric)['split_half_corr'])
                split_half_corr_t2[repetition_idx, time_idx1, time_idx2, site_idx] = shc_t2

                # Compute temporal correlation between time bins
                if correlation_metric == 'pearson':
                    corr_value = stats.pearsonr(rates_t1, rates_t2)[0]
                else:
                    corr_value = stats.spearmanr(rates_t1, rates_t2)[0]
                
                temporal_correlations[repetition_idx, time_idx1, time_idx2, site_idx] = corr_value

# Create output directory if needed
output_dir = os.path.join(scores_save_path, neuron_file_name)
os.makedirs(output_dir, exist_ok=True)

# Save results to HDF5 file
output_h5_path = os.path.join(output_dir, f"temporal_change_{correlation_metric}.h5")
with h5py.File(output_h5_path, 'w') as h5_file:
    # Output shape: [repetitions x time bins x time bins x sites]
    h5_file.create_dataset("raw_correlations", data=temporal_correlations)
    h5_file.create_dataset("shcs_nt", data=split_half_corr_t1)
    h5_file.create_dataset("shcs_nt_next", data=split_half_corr_t2)
    print(f"Saved scores to {output_h5_path}, shape: {temporal_correlations.shape}")

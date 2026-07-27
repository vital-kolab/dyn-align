"""
This script analyzes neural responses from IT (Inferior Temporal) cortex neurons
by computing correlations between different visual stimulus conditions (coherent,
incoherent, appearance-free).
The analysis involves:
- Loading neural firing rates from HDF5 files for three conditions:
    * Coherent motion stimulus
    * Incoherent motion stimulus
    * appearance-free stimulus (AFV)
- Computing split-half reliability correlations for each neural response
- Computing cross-condition correlations (coherent vs AFV, coherent vs incoherent, etc.)
- Performing multiple repetitions with random seeds for robustness
- Saving all correlation results to an output HDF5 file
This allows quantification of how different visual conditions
drive neural activity, useful for understanding static vs dynamic factor encoding
in the ventral system.
Output:
    HDF5 file containing:
    - correlations_coherent: Split-half corrected coherent condition correlations
    - correlations_afv: Coherent-AFV cross-condition correlations
    - correlations_incoherent: Coherent-incoherent cross-condition correlations
    - correlations_incoherent_afv: Incoherent-AFV cross-condition correlations
    - shc_coherent/afv/incoherent: Raw split-half reliability scores
"""
import sys
sys.path.append('../')
from utils.correlation_metrics import get_splithalf_corr
import h5py
import numpy as np
import os
from scipy import stats


# Configuration
scores_save_path = '../../scores'
correlation_method = 'spearman'  # 'pearson' or 'spearman'

# Load coherent condition neural data
coherent_file_path = '[path to your coherent condition neural recordings HDF5 file]'
with h5py.File(coherent_file_path, 'r') as coherent_file:
    coherent_rates = np.array(coherent_file['rates'])
coherent_file_name = coherent_file_path.split('/')[-1].split('.')[0]

# Load incoherent condition neural data
incoherent_file_path = '[path to your incoherent condition neural recordings HDF5 file]'
with h5py.File(incoherent_file_path, 'r') as incoherent_file:
    incoherent_rates = np.array(incoherent_file['rates'])

# Load AFV neural data, coherent motion, different appearance
afv_file_path = '[path to your AFV condition neural recordings HDF5 file]'
with h5py.File(afv_file_path, 'r') as afv_file:
    afv_rates = np.array(afv_file['rates'])

# Initialize arrays for storing correlation results
num_repetitions = 1000
n_time_bins = coherent_rates.shape[0]
num_neurons = coherent_rates.shape[2]

# Correlation arrays for different comparisons
correlations_afv = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan
correlations_incoherent = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan

# Split-half correlation arrays
shc_coherent = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan
shc_afv = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan
shc_incoherent = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan

# Compute correlations across repetitions
for rep in range(num_repetitions):
    for time_bin in range(n_time_bins):
        coherent_trial = coherent_rates[time_bin, :, :]
        incoherent_trial = incoherent_rates[time_bin, :, :]
        afv_trial = afv_rates[time_bin, :, :]

        for neuron_idx in range(num_neurons):
            # Compute split-half reliability for coherent condition
            coherent_shc = get_splithalf_corr(coherent_trial[:, neuron_idx], 
                                              seed=np.random.randint(10000), 
                                              type=correlation_method)['split_half_corr']
            shc_coherent[rep, time_bin, neuron_idx] = coherent_shc

            # Compute correlation between coherent and AFV responses
            coherent_mean = np.nanmean(coherent_trial[:, neuron_idx], -1)
            afv_mean = np.nanmean(afv_trial[:, neuron_idx], -1)
            if correlation_method == 'spearman':
                afv_correlation = stats.spearmanr(coherent_mean, afv_mean)[0]
            else:
                afv_correlation = stats.pearsonr(coherent_mean, afv_mean)[0]
            
            afv_shc = get_splithalf_corr(afv_trial[:, neuron_idx], 
                                         seed=np.random.randint(10000), 
                                         type=correlation_method)['split_half_corr']
            correlations_afv[rep, time_bin, neuron_idx] = afv_correlation
            shc_afv[rep, time_bin, neuron_idx] = afv_shc
            
            # Compute correlation between coherent and incoherent responses
            incoherent_mean = np.nanmean(incoherent_trial[:, neuron_idx], -1)
            if correlation_method == 'spearman':
                incoherent_correlation = stats.spearmanr(coherent_mean, incoherent_mean)[0]
            else:
                incoherent_correlation = stats.pearsonr(coherent_mean, incoherent_mean)[0]
            
            incoherent_shc = get_splithalf_corr(incoherent_trial[:, neuron_idx], 
                                                seed=np.random.randint(10000), 
                                                type=correlation_method)['split_half_corr']
            correlations_incoherent[rep, time_bin, neuron_idx] = incoherent_correlation
            shc_incoherent[rep, time_bin, neuron_idx] = incoherent_shc

# Handle infinite values
correlations_afv[np.isinf(correlations_afv)] = np.nan
correlations_incoherent[np.isinf(correlations_incoherent)] = np.nan

# Save results to HDF5 file
os.makedirs(os.path.join(scores_save_path, coherent_file_name), exist_ok=True)

output_h5_path = os.path.join(scores_save_path, coherent_file_name, 
                              f"correlations_static_dynamic_{correlation_method}.h5")
with h5py.File(output_h5_path, 'w') as h5_file:
    h5_file.create_dataset("correlations_afv", data=correlations_afv)
    h5_file.create_dataset("correlations_incoherent", data=correlations_incoherent)
    h5_file.create_dataset("shc_coherent", data=shc_coherent)
    h5_file.create_dataset("shc_afv", data=shc_afv)
    h5_file.create_dataset("shc_incoherent", data=shc_incoherent)
    print(f"Saved scores to {output_h5_path}, shape: {correlations_afv.shape}")

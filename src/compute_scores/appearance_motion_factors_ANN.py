"""
This script analyzes neural responses from IT (Inferior Temporal) cortex neurons
by computing correlations between different visual stimulus conditions.

The analysis involves:
- Loading neural features from HDF5 files for three conditions:
    * Coherent motion stimulus
    * Incoherent motion stimulus
    * Appearance-free stimulus (AFV)
- Computing cross-condition correlations:
    * Coherent vs AFV features
    * Coherent vs Incoherent features
- Processing multiple repetitions across time bins and neurons
- Saving correlation results to an output HDF5 file

This enables quantification of how different visual conditions drive model features,
useful for understanding static vs dynamic factor encoding in ANN models.

Output:
    HDF5 file containing:
    - correlations_afv: Coherent-AFV cross-condition correlations
    - correlations_incoherent: Coherent-incoherent cross-condition correlations
    
    Shape: (num_repetitions, n_time_bins, num_neurons)
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

# Load coherent condition neural features
coherent_file_path = '[path to your coherent condition features HDF5 file]'
with h5py.File(coherent_file_path, 'r') as coherent_file:
    coherent_features = np.array(coherent_file['features'])
coherent_file_name = coherent_file_path.split('/')[-1].split('.')[0]

# Load incoherent condition neural features
incoherent_file_path = '[path to your incoherent condition features HDF5 file]'
with h5py.File(incoherent_file_path, 'r') as incoherent_file:
    incoherent_features = np.array(incoherent_file['features'])

# Load AFV neural features, coherent motion, different appearance
afv_file_path = '[path to your AFV condition features HDF5 file]'
with h5py.File(afv_file_path, 'r') as afv_file:
    afv_features = np.array(afv_file['features'])

# Initialize arrays for storing correlation results
num_repetitions = coherent_features.shape[3]
n_time_bins = coherent_features.shape[0]
num_neurons = coherent_features.shape[2]

# Correlation arrays for different comparisons
correlations_afv = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan
correlations_incoherent = np.zeros((num_repetitions, n_time_bins, num_neurons)) + np.nan

# Compute correlations across repetitions
for rep in range(num_repetitions):
    for time_bin in range(n_time_bins):
        coherent_trial = coherent_features[time_bin, :, :, rep]
        incoherent_trial = incoherent_features[time_bin, :, :, rep]
        afv_trial = afv_features[time_bin, :, :, rep]

        for neuron_idx in range(num_neurons):
            
            # Compute correlation between coherent and AFV features
            coherent_ = coherent_trial[:, neuron_idx]
            afv_ = afv_trial[:, neuron_idx]
            if correlation_method == 'spearman':
                afv_correlation = stats.spearmanr(coherent_, afv_)[0]
            else:
                afv_correlation = stats.pearsonr(coherent_, afv_)[0]
            
            correlations_afv[rep, time_bin, neuron_idx] = afv_correlation
            
            # Compute correlation between coherent and incoherent features
            incoherent_ = incoherent_trial[:, neuron_idx]
            if correlation_method == 'spearman':
                incoherent_correlation = stats.spearmanr(coherent_, incoherent_)[0]
            else:
                incoherent_correlation = stats.pearsonr(coherent_, incoherent_)[0]
            
            correlations_incoherent[rep, time_bin, neuron_idx] = incoherent_correlation

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
    print(f"Saved scores to {output_h5_path}, shape: {correlations_afv.shape}")

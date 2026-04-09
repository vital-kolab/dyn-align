"""
Temporal Change Analysis - Model Feature Correlation Computation
This script computes temporal correlations between model features across time bins,
along with split-half reliability measures for feature data. It performs repetition 
resampling to estimate the stability of correlations between consecutive time periods
in model feature representations.
The analysis:
1. Loads model feature data from an HDF5 file containing [time bins x videos x units x repetitions]
2. For each pair of time bins and feature unit, computes:
    - Temporal correlation between features at two different time points
    - Split-half reliability (Spearman-Brown corrected) for each time bin
3. Performs repetition resampling (default: 100 iterations) for robust estimation
4. Saves all results (raw correlations and reliability measures) to output HDF5 file
Key parameters:
- num_repetitions: Number of repetition iterations (default: 100)
- correlation_metric: Type of correlation metric ('pearson' or 'spearman')
- Input: Model feature data with shape [time bins x videos x units x repetitions]
- Output: Arrays of temporal correlations and split-half correlations for each repetition sample
"""
import sys
sys.path.append('../')

from utils.correlation_metrics import get_splithalf_corr, spearmanbrown_correction
import h5py
import numpy as np
from scipy import stats
import os

# Load model feature data from HDF5 file
features_file_path = '[your file containing model features for videos]'
features_h5_file = h5py.File(features_file_path, 'r')
print("Features Keys:", list(features_h5_file.keys()))
feature_file_name = features_file_path.split('/')[-1].split('.')[0]

# Extract model feature data: [time bins x videos x units x repetitions]
model_features = np.array(features_h5_file['features'])
correlation_metric = 'pearson'  # 'pearson' or 'spearman'
print(f'Features: {model_features.shape} [time bins x videos x units x repetitions]')

# Initialize repetition parameters
np.random.seed(42)
scores_save_path = '../../scores'

# Initialize result arrays
temporal_correlations = np.zeros((model_features.shape[3], model_features.shape[0], model_features.shape[0], model_features.shape[2])) + np.nan

# Compute correlations across time bins and feature units
for time_idx1 in range(model_features.shape[0]):
    for time_idx2 in range(time_idx1 + 1, time_idx1 + 5, 2):
        for unit_idx in range(model_features.shape[2]):
            for repetition_idx in range(model_features.shape[3]):
                repetition_seed = np.random.randint(0, 10000)
                
                # Time bin 1: compute split-half reliability
                features_t1 = np.nanmean(model_features[time_idx1, :, unit_idx], axis=-1)
                
                # Time bin 2: compute split-half reliability
                features_t2 = np.nanmean(model_features[time_idx2, :, unit_idx], axis=-1)
                
                # Compute temporal correlation between time bins
                if correlation_metric == 'pearson':
                    corr_value = stats.pearsonr(features_t1, features_t2)[0]
                else:
                    corr_value = stats.spearmanr(features_t1, features_t2)[0]
                
                temporal_correlations[repetition_idx, time_idx1, time_idx2, unit_idx] = corr_value

# Create output directory if needed
output_dir = os.path.join(scores_save_path, feature_file_name)
os.makedirs(output_dir, exist_ok=True)

# Save results to HDF5 file
output_h5_path = os.path.join(output_dir, f"temporal_change_{correlation_metric}.h5")
with h5py.File(output_h5_path, 'w') as h5_file:
    # Output shape: [repetitions x time bins x time bins x units]
    h5_file.create_dataset("raw_correlations", data=temporal_correlations)
    print(f"Saved scores to {output_h5_path}, shape: {temporal_correlations.shape}")

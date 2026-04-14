"""
Script: fig6_plot_temporal_change_ANN.py
This script analyzes and visualizes temporal changes in neural network activations
across different video understanding architectures. It compares static models
(using repeated frame features) with dynamic models (using temporal features)
by computing correlation decreases over time and comparing them to biological
neuron statistics.
Key functionalities:
- Loads temporal correlation data from HDF5 files for multiple neural network models
- Computes temporal changes in correlations (delta_corrs) across spatial and temporal dimensions
- Extracts early and late decline curve parameters for each model
- Generates temporal change plots for both static and dynamic model groups
- Creates comparative histograms showing model curve parameters vs. biological neuron statistics
- Overlays median values and biological neuron statistics (median absolute deviation) on visualizations
Output:
- Temporal change curves (PDF plots)
- Histogram comparisons of early and late decline parameters between models and biological neurons
"""

import sys
sys.path.append('../')

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.plot_utils import journal_figure_pdf
import scienceplots

plt.style.use(['nature'])

# Define static models (with repeated frame features)
static_models = {
    "images_moving1_c2d_r50-blocks.4.res_blocks.5.activation_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'C2D'], 
    "images_moving1_i3d_r50-blocks.5.res_blocks.2.activation_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'I3D'], 
    "images_moving1_x3d_xs-blocks.4.res_blocks.6.activation_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'X3D-XS'], 
    "images_moving1_sam2.1_hiera_b+-256-memory_attention-2_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SAM2'], 
    "images_moving1_slow_r50_ssv2-blocks.4.res_blocks.2.activation_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SlowR50-ssv2'], 
    "images_moving1_slow_r50-blocks.4.res_blocks.2.activation_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SlowR50'], 
    "images_moving1_timesformer_ssv2-timesformer.encoder.layer.8_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TimesFormer-ssv2'], 
    "images_moving1_timesformer-timesformer.encoder.layer.11_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TimesFormer'], 
    "images_moving1_videomamba_ssv2-layers.10.mixer_repeated-frame_features_srp_1000_10.h5"  : ['d', 'tab:blue', 'VideoMamba-ssv2'], 
    "images_moving1_videomamba-layers.10.mixer_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'VideoMamba'], 
    "images_moving1_videomae-videomae.encoder.layer.2_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'VideoMAE'], 
    "images_moving1_matnet_fusion_type_gated-layer4,sensor_fusion_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'MatNet'],
    "images_moving1_twostream_deeplabv3plus_resnet101_davis-layer4,sensor_fusion_repeated-frame_features_srp_1000_10.h5" : ['d', 'tab:blue', 'FusionSeg'],
}

# Define dynamic models (with temporal features)
dynamic_models = {
    "images_moving1_c2d_r50-blocks.4.res_blocks.5.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'C2D'], 
    "images_moving1_i3d_r50-blocks.5.res_blocks.2.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'I3D'], 
    "images_moving1_x3d_xs-blocks.4.res_blocks.6.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'X3D-XS'], 
    "images_moving1_sam2.1_hiera_b+-256-memory_attention-2_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SAM2'], 
    "images_moving1_slow_r50_ssv2-blocks.4.res_blocks.2.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SlowR50-ssv2'], 
    "images_moving1_slow_r50-blocks.4.res_blocks.2.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SlowR50'], 
    "images_moving1_timesformer_ssv2-timesformer.encoder.layer.8_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TimesFormer-ssv2'], 
    "images_moving1_timesformer-timesformer.encoder.layer.11_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TimesFormer'], 
    "images_moving1_videomamba_ssv2-layers.10.mixer_features_srp_1000_10.h5"  : ['d', 'tab:blue', 'VideoMamba-ssv2'], 
    "images_moving1_videomamba-layers.10.mixer_features_srp_1000_10.h5" : ['d', 'tab:blue', 'VideoMamba'], 
    "images_moving1_videomae-videomae.encoder.layer.2_features_srp_1000_10.h5" : ['d', 'tab:blue', 'VideoMAE'], 
    "images_moving1_matnet_fusion_type_gated-layer4,sensor_fusion_moving-object1_features_srp_1000_10.h5" : ['d', 'tab:blue', 'MatNet-FUS'],
    "images_moving1_twostream_deeplabv3plus_resnet101_davis-layer4,sensor_fusion_moving-object1_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TSDLv3-FUS'],
}

# Neural network neuron statistics (median and median absolute deviation)
neurons_params_median = [0.2291011918000626, 0.6303214252388362, 0.1078761380826867]
neurons_params_mad = [0.061590730368238034, 0.5024047980560771, 0.04412893983073607]

# Initialize arrays to store results
repetitions = 10
curve_params_all = np.zeros((len(static_models) + len(dynamic_models), 1000, 3)) + np.nan
delta_corrs_all = np.zeros((len(static_models) + len(dynamic_models), 7, 1000)) + np.nan

# Process both static and dynamic models
for j, model_group in enumerate(['static_models', 'dynamic_models']):
    if model_group == 'static_models':
        model_color = 'tab:red'
        models = static_models
    else:
        model_color = 'tab:blue'
        models = dynamic_models

    # Load and process each model
    for m, model in enumerate(models.keys()):
        neurons_file_path = f'../../scores/{model}/temporal_change_pearson.h5'
        neurons_h5_file = h5py.File(neurons_file_path, 'r')
        
        # Load correlations and compute median across repetitions
        corrs = np.array(neurons_h5_file['raw_correlations'])
        corrs = np.nanmedian(corrs, axis=0)

        # Handle models with smaller spatial dimensions
        if 'matnet' in model or 'two' in model:
            empty = np.zeros((18, 18, corrs.shape[2])) + np.nan
            empty[:corrs.shape[0], :corrs.shape[0]] = corrs
            corrs = empty

        # Downsample spatial dimensions by factor of 2
        corrs = corrs[0:corrs.shape[0]:2, 0:corrs.shape[1]:2, :]

        # Extract dimensions
        t_idxs = list(range(0, corrs.shape[1]-2))
        n_time_bins = corrs.shape[1]
        n_neurons = corrs.shape[-1]

        # Compute correlation statistics across neurons
        corrs_mean = np.nanmedian(corrs, axis=2)
        corrs_std = np.nanstd(corrs, axis=2)

        # Calculate temporal change in correlations (delta_corrs)
        delta_corrs = np.zeros((n_time_bins, n_neurons)) + np.nan

        t_start = t_idxs[0]
        end_time_bin_idx = t_idxs[-1]
        
        for n in range(n_neurons):
            for t in range(t_start, end_time_bin_idx+1):
                if t+2 < corrs.shape[0]:
                    delta_corrs[t, n] = corrs[t, t+1, n] - corrs[t, t+2, n]

        # Extract curve parameters (early and late decline)
        curve_params = np.zeros((repetitions, n_neurons, 2)) + np.nan

        for r in range(repetitions):
            for n in range(n_neurons):
                if np.isnan(delta_corrs[t_start:end_time_bin_idx+1, n]).any():
                    continue
                if np.isinf(delta_corrs[t_start:end_time_bin_idx+1, n]).any():
                    continue
                
                y = delta_corrs[t_start:end_time_bin_idx+1, n]
                curve_params[r, n, 0] = y[0]
                curve_params[r, n, 2] = y[-1]

        # Store aggregated results
        curve_params = np.nanmedian(curve_params, axis=0)
        delta_corrs_all[len(static_models) * j + m] = delta_corrs[t_start:end_time_bin_idx+1]
        curve_params_all[len(static_models) * j + m] = curve_params

    # Plot temporal change for current model group
    delta_corrs_all_ = np.nanmean(delta_corrs_all, axis=0)
    plt.figure(figsize=(5, 4))

    plt.errorbar(np.arange(delta_corrs_all_[t_start:end_time_bin_idx+1].shape[0]), 
                 np.nanmedian(delta_corrs_all_[t_start:end_time_bin_idx+1], -1), 
                 c=model_color, capsize=0, linestyle='-')
    
    plt.fill_between(np.arange(delta_corrs_all_[t_start:end_time_bin_idx+1].shape[0]), 
                     np.nanmedian(delta_corrs_all_[t_start:end_time_bin_idx+1], -1) - np.nanstd(delta_corrs_all_[t_start:end_time_bin_idx+1], -1),
                     np.nanmedian(delta_corrs_all_[t_start:end_time_bin_idx+1], -1) + np.nanstd(delta_corrs_all_[t_start:end_time_bin_idx+1], -1),
                     color=model_color, alpha=0.5)

    # Format x-axis labels
    labels = [i+1 for i in range(end_time_bin_idx+1 - t_start)]
    range_ = list(range(len(labels)))
    labels = labels[::3]
    range_ = range_[::3]

    plt.xticks(range_, labels, fontsize=6)
    plt.ylim([0., 0.25])
    plt.xlabel("feature frame", fontsize=12)
    plt.ylabel("correlation decrease", fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    journal_figure_pdf(do_save=True, filename=f'../../plots/temporal_change_ANN-{model_group}.pdf')

# Load pre-computed neuron curve parameters
neurons_curve_params = np.load('neurons_curve_params.npy')

# Compare model curve parameters with biological neurons
file_name = 'static_dynamic_models'
for i, phase in enumerate(['early', 'late']):
    # Extract curve parameters for each model type
    static = np.nanmedian(curve_params_all[:len(static_models), :, i], 1)
    dynamic = np.nanmedian(curve_params_all[len(static_models):, :, i], 1)

    plt.figure(figsize=(4, 5))

    # Plot histograms 
    plt.hist(static, bins=5, color='tab:red', edgecolor='black', density=False, linewidth=2, alpha=0.6, zorder=2)
    plt.hist(dynamic, bins=5, color='tab:blue', edgecolor='black', density=False, linewidth=2, alpha=0.6, zorder=1)
    plt.xlim([0.0, 0.36])

    # Overlay medians and biological neuron statistics
    plt.axvline(np.nanmedian(static), color='tab:red', linestyle='--', linewidth=2, label=f'median={np.round(np.nanmedian(static), 3)}', zorder=2)
    plt.plot(np.nanmedian(static), plt.ylim()[1] * 1.0, marker='v', color='tab:red', markersize=10, zorder=2)
    plt.axvline(np.nanmedian(dynamic), color='tab:blue', linestyle='--', linewidth=2, label=f'median={np.round(np.nanmedian(dynamic), 3)}', zorder=1)
    plt.plot(np.nanmedian(dynamic), plt.ylim()[1] * 1.0, marker='v', color='tab:blue', markersize=10, zorder=1)

    plt.axvline(neurons_params_median[i], color='tab:green', linestyle='--', linewidth=1.5, label=f'median={np.round(neurons_params_median[i], 3)}', zorder=3)
    plt.axvspan(neurons_params_median[i] - neurons_params_mad[i], neurons_params_median[i] + neurons_params_mad[i], color='tab:green', alpha=0.15, zorder=3)
    plt.plot(neurons_params_median[i], plt.ylim()[1] * 1.0, marker='v', color='tab:green', markersize=10, zorder=3)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(f"{phase} temporal change", fontsize=12) 
    plt.ylabel("number of models", fontsize=12)

    # Save figure
    journal_figure_pdf(do_save=True, filename=f'../../plots/temporal_change_ANN_{phase}_model_hist.pdf')

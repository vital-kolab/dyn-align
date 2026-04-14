"""
Behavioral Decoding Analysis of Object Motion Direction for ANN models

This script visualizes histograms of models based on their behavioral performance. It processes pre-computed accuracy scores.

The analysis includes:
1. Loading accuracy scores from HDF5 file
2. Computing the highest accuracy across time bins for each model
3. Generating a histogram comparing static (image-based) vs dynamic (video-based) ANN models, with vertical lines indicating mean performance for each group and chance-level performance.

Key outputs:
- decode_obj_dir_linear_model_histogram.pdf: Histogram comparing model performance on object motion direction decoding task, with mean performance indicated for static vs dynamic models.
"""

import sys
sys.path.append('../')

from utils.plot_utils import journal_figure_pdf

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import scienceplots

plt.style.use(['nature'])

static_models = {
    "images_moving2_alexnet_features_srp_1000_10" : ['f', 'tab:red', 'AlexNet'], 
    "images_moving2_convnext_features_srp_1000_10" : ['f', 'tab:red', 'ConvNeXt'], 
    "images_moving2_densenet121_features_srp_1000_10" : ['f', 'tab:red', 'DenseNet-121'], 
    "images_moving2_efficientnet_b0_features_srp_1000_10" : ['f', 'tab:red', 'EfficientNet-B0'], 
    "images_moving2_hiera_features_srp_1000_10.h5" : ['f', 'tab:red', 'Hiera'], 
    "images_moving2_inception_v3_features_srp_1000_10" : ['f', 'tab:red', 'Inception-V3'], 
    "images_moving2_pnasnet_features_srp_1000_10" : ['f', 'tab:red', 'PNASNet'], 
    "images_moving2_resnet50_features_srp_1000_10" : ['f', 'tab:red', 'ResNet-50'], 
    "images_moving2_resnet50_ssl_features_srp_1000_10" : ['f', 'tab:red', 'ResNet-50-SSL'], 
    "images_moving2_vit_features_srp_1000_10" : ['f', 'tab:red', 'ViT'], 
    "images_moving2_vit_ssl_features_srp_1000_10" : ['f', 'tab:red', 'ViT-SSL'], 
    "images_moving2_nasnet_features_srp_1000_10" : ['f', 'tab:red', 'NASNet'], 
    "images_moving2_rgc_intermediate_features_ff_srp_1000_10.h5" : ['ri', 'tab:red', 'ConvRNN-F'],
    "images_moving2_CORnet-S_IT_features_srp_1000_10" : ['ri', 'tab:red', 'CORnet-S'],
}

dynamic_models = {
    "images_moving2_c2d_r50-blocks.4.res_blocks.5.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'C2D'], 
    "images_moving2_i3d_r50-blocks.5.res_blocks.2.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'I3D'], 
    "images_moving2_x3d_xs-blocks.4.res_blocks.6.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'X3D-XS'], 
    "images_moving2_sam2.1_hiera_b+-256-memory_attention-2_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SAM2'], 
    "images_moving2_slow_r50_ssv2-blocks.4.res_blocks.2.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SlowR50-ssv2'], 
    "images_moving2_slow_r50-blocks.4.res_blocks.2.activation_features_srp_1000_10.h5" : ['d', 'tab:blue', 'SlowR50'], 
    "images_moving2_timesformer_ssv2-timesformer.encoder.layer.8_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TimesFormer-ssv2'], 
    "images_moving2_timesformer-timesformer.encoder.layer.11_features_srp_1000_10.h5" : ['d', 'tab:blue', 'TimesFormer'], 
    "images_moving2_videomamba_ssv2-layers.10.mixer_features_srp_1000_10.h5"  : ['d', 'tab:blue', 'VideoMamba-ssv2'], 
    "images_moving2_videomamba-layers.10.mixer_features_srp_1000_10.h5" : ['d', 'tab:blue', 'VideoMamba'], 
    "images_moving2_videomae-videomae.encoder.layer.2_features_srp_1000_10.h5" : ['d', 'tab:blue', 'VideoMAE'], 
    "images_moving2_matnet_fusion_type_gated-layer4,sensor_fusion_features_srp_1000_10.h5" : ['d', 'tab:blue', 'MatNet'],
    "images_moving2_twostream_deeplabv3plus_resnet101_davis-layer4,sensor_fusion_features_srp_1000_10.h5" : ['d', 'tab:blue', 'FusionSeg'],
    "images_moving2_rgc_intermediate_features_srp_1000_10" : ['rv', 'tab:blue', 'ConvRNN-V']
}

models = {**static_models, **dynamic_models}

scores_path = '[path to your fodler containing model classification scores]'

values = []
errors = []
methods_names = list(models.keys())
for name in methods_names:
    file_rgb = h5py.File(f'{scores_path}/{name}/decode_ann_obj_dir_linear.h5', 'r')
    i1_per_time = np.array(file_rgb[f'i1_all'])

    i1_per_time = np.nanmean(i1_per_time, axis=-1)
    i1_per_time = np.nanmean(i1_per_time, axis=0)

    i1_per_time_m = np.nanmean(i1_per_time, axis=-1)
    max_id = np.argmax(i1_per_time_m)
    i1_per_time = i1_per_time[max_id]

    i1_per_time_mean = np.nanmean(i1_per_time, axis=0)
    i1_per_time_se = np.nanstd(i1_per_time, axis=0) / np.sqrt(i1_per_time.shape[0])

    values.append(i1_per_time_mean)
    errors.append(i1_per_time_se)


values_ff = np.array(values[:len(static_models)])
values_dd = np.array(values[len(static_models):])

plt.figure(figsize=(4, 5))
plt.hist(values_ff, bins=5, color='tab:red', alpha=.6, label='image-based ANNs', edgecolor='black')
plt.hist(values_dd, bins=5, color='tab:blue', alpha=0.6, label='video-based ANNs', edgecolor='black')

plt.axvline(np.nanmean(values_ff), color='tab:red', linestyle='--', linewidth=2, label=f'image-based ANNs: {np.nanmean(values_ff):.3f}')
plt.plot(np.nanmean(values_ff), plt.ylim()[1] * 1.0, marker='v', color='tab:red', markersize=10, zorder=3)

plt.axvline(np.nanmean(values_dd), color='tab:blue', linestyle='--', linewidth=2, label=f'video-based ANNs: {np.nanmean(values_dd):.3f}')
plt.plot(np.nanmean(values_dd), plt.ylim()[1] * 1.0, marker='v', color='tab:blue', markersize=10, zorder=3)

plt.axvline(0.5, color='black', linestyle='--', linewidth=1, label=f'chance-level')

plt.xlabel('accuracy (percentage correct)', fontsize=12)
plt.ylabel('number of models', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()

journal_figure_pdf(do_save=True, filename=f'../../plots/decode_obj_dir_linear_model_histogram.pdf')

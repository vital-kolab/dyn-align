"""
Decode Object Direction from Neural Recordings using RNN Classification
This script processes neural recordings to classify object direction using LSTM
with 10-fold cross-validation, 
training on AFV neural data and testing on both AFV and RGB neural data.
The script:
1. Loads pre-computed neural recordings for RGB and AFV videos from HDF5 files with shape [n_time_bins, videos, sites]
2. Loads object direction labels corresponding to each video
3. Distributes classification tasks across multiple CPU cores using multiprocessing
4. For each combination of (split_repetition, time_bin):
    - Extracts neural recordings up to the specific time bin for both RGB and AFV data
    - Performs RNN classification with 10-fold cross-validation training on AFV neural data and testing on both AFV and RGB neural data
    - Computes per-class I1 (information integration) scores for RGB and AFV predictions
    - Stores results in shared numpy arrays
5. Saves the per-class classification scores for both modalities to an HDF5 output file
The script supports both local and HPC (SLURM) execution environments, automatically
detecting available CPU cores for parallel processing.
Input:
     - RGB neural recordings HDF5 file: [n_time_bins, videos, sites]
     - AFV neural recordings HDF5 file: [n_time_bins, videos, sites]
     - Object direction labels: 1D array of class labels indexed by video
Output:
     - HDF5 file containing I1 classification scores for both RGB and AFV with shape [split_repetitions, n_time_bins, videos]
"""
import sys
sys.path.append('../')

import os
import h5py
import numpy as np
from utils.classification import compute_i1, get_rnn_classifications_rgb_afv
import multiprocessing as mp 
from multiprocessing.managers import BaseManager

np.random.seed(42)

# Manager class to handle shared numpy arrays across processes
class MyManager(BaseManager):
    pass
    
MyManager.register('np_empty', np.empty, mp.managers.ArrayProxy)


def compute_scores(process_id, parallelizations, classification_scores_rgb, classification_scores_afv, neural_recordings_rgb, neural_recordings_afv, object_labels, split_repetitions, start, end):
    # Process a chunk of parallelizations assigned to this worker
    for split_rep, time_bin in parallelizations[start:end]:
        print(f'Process #{process_id}: Computing scores for time bin {time_bin}')

        # Extract neural data up to specific time bin for both RGB and AFV
        current_data_rgb = neural_recordings_rgb[:time_bin+1, :, :]  # shape: [time_bins, videos, sites]
        current_data_afv = neural_recordings_afv[:time_bin+1, :, :]  # shape: [time_bins, videos, sites]

        assert current_data_rgb.shape[1] == object_labels.shape[0]

        # Classify using RNN with 10-fold cross-validation
        preds_afv, probs_afv, preds_rgb, probs_rgb = get_rnn_classifications_rgb_afv(current_data_afv, current_data_rgb, object_labels, model_config=model_config, nrfolds=10, seed=np.random.randint(1000), standardize=True)

        i1_rgb, i1_rgb_std, i1_rgb_all = compute_i1(probs_rgb, object_labels, return_scores=True)
        i1_afv, i1_afv_std, i1_afv_all = compute_i1(probs_afv, object_labels, return_scores=True)
    
        # Store per-class scores
        classification_scores_rgb[split_rep, time_bin, :] = i1_rgb_all
        classification_scores_afv[split_rep, time_bin, :] = i1_afv_all

        print(f'Process #{process_id}: split: {split_rep+1}/{split_repetitions} - time_bin: {time_bin}')


if __name__ == "__main__":
    
    neural_recordings_rgb_path = '[your file containing neural recordings for RGB videos]'
    neural_recordings_afv_path = '[your file containing neural recordings for AFV videos]'
    split_repetitions = 10
    hpc = True
    scores_save_path = '../../scores'

    recordings_rgb_filename = neural_recordings_rgb_path.split('/')[-1]
    recordings_rgb_file = h5py.File(neural_recordings_rgb_path, 'r')
    print("Keys:", list(recordings_rgb_file.keys()))
    # INPUT FILE SHAPE: [n_time_bins, videos, sites]
    neural_recordings_rgb = np.array(recordings_rgb_file['neural_data'])

    recordings_afv_filename = neural_recordings_afv_path.split('/')[-1]
    recordings_afv_file = h5py.File(neural_recordings_afv_path, 'r')
    print("Keys:", list(recordings_afv_file.keys()))
    # INPUT FILE SHAPE: [n_time_bins, videos, sites]
    neural_recordings_afv = np.array(recordings_afv_file['neural_data'])
   
    n_time_bins = neural_recordings_rgb.shape[0]

    # Create list of all (split_rep, time_bin) tuples to process
    parallelizations = []
    for split_rep in range(split_repetitions):
        for time_bin in range(n_time_bins):
            parallelizations.append((split_rep, time_bin))

    n_parallelizations = len(parallelizations)

    # Load object direction labels indexed by video
    object_labels = np.genfromtxt('[your label file]', delimiter='\n', dtype=np.int64)
    assert object_labels.shape[0] == neural_recordings_rgb.shape[1]

    # Configuration for RNN model decoder
    model_config = dict(
        hidden_dim=200,
        output_dim=8, 
        model='lstm',
        num_layers=1,
        patience=100, 
        max_epochs=200,
        learning_rate=1e-2, 
        verbose=False
    )

    # Create shared array for results across processes
    m = MyManager()
    m.start()

    # OUTPUT ARRAY SHAPE: [split_repetitions, n_frames, n_videos]
    classification_scores_rgb = m.np_empty((split_repetitions, neural_recordings_rgb.shape[0], neural_recordings_rgb.shape[1]))
    classification_scores_afv = m.np_empty((split_repetitions, neural_recordings_afv.shape[0], neural_recordings_afv.shape[1]))
    classification_scores_rgb[:] = np.nan
    classification_scores_afv[:] = np.nan

    # Distribute work across available CPU cores
    num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1)) if hpc else mp.cpu_count()
    chunk_size = n_parallelizations // num_processes
    
    # Launch worker processes
    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = n_parallelizations if i == num_processes - 1 else (i + 1) * chunk_size
        p = mp.Process(target=compute_scores, args=(i, parallelizations, classification_scores_rgb, classification_scores_afv,
                                                     neural_recordings_rgb, neural_recordings_afv, object_labels, model_config,
                                                     split_repetitions, start, end))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

    classification_scores_rgb = np.array(classification_scores_rgb)
    classification_scores_afv = np.array(classification_scores_afv)

    # Save results
    os.makedirs(os.path.join(scores_save_path, recordings_rgb_filename), exist_ok=True)

    save_file_name = f"decode_IT_obj_dir_rnn_rgb2afv.h5" 
    output_h5_path = os.path.join(scores_save_path, recordings_rgb_filename, save_file_name)
    with h5py.File(output_h5_path, 'w') as h5_file:
        h5_file.create_dataset("i1_all_rgb", data=classification_scores_rgb)
        h5_file.create_dataset("i1_all_afv", data=classification_scores_afv)
        print(f"Saved scores to {output_h5_path}, shape: {classification_scores_rgb.shape}")

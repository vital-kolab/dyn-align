
"""
Decode Object speed from ANN Model Features using Linear Regression
This script processes ANN (Artificial Neural Network) model features to classify object
speed using Linear Regression with 10-fold cross-validation.
The script:
1. Loads pre-computed ANN model features from an HDF5 file with shape [n_frames, videos, units, model_repetitions]
2. Loads object speed labels corresponding to each video
3. Distributes classification tasks across multiple CPU cores using multiprocessing
4. For each combination of (split_repetition, time_bin, model_repetition):
    - Extracts model features for the specific time bin and model repetition
    - Performs LDA classification with 10-fold cross-validation
    - Computes per-class I1 (information integration) scores
    - Stores results in a shared numpy array
5. Saves the per-class classification scores to an HDF5 output file
The script supports both local and HPC (SLURM) execution environments, automatically
detecting available CPU cores for parallel processing.
Input:
     - Model features HDF5 file: [n_frames, videos, units, model_repetitions]
     - Object direction labels: 1D array of class labels indexed by video
Output:
     - HDF5 file containing classification scores with shape [split_repetitions, n_frames, videos, model_repetitions]
"""
import sys
sys.path.append('../')

import os
import h5py
import numpy as np
from utils.regression import get_regressions, speed_accuracy
import multiprocessing as mp 
from multiprocessing.managers import BaseManager

np.random.seed(42)

# Manager class to handle shared numpy arrays across processes
class MyManager(BaseManager):
    pass
    
MyManager.register('np_empty', np.empty, mp.managers.ArrayProxy)


def compute_preds(process_id, parallelizations, model_predictions, model_features, object_labels, split_repetitions, start, end):
    # Process a chunk of parallelizations assigned to this worker
    for split_rep, time_bin, model_rep in parallelizations[start:end]:
        print(f'Process #{process_id}: Computing scores for time bin {time_bin}')

        # Extract data for specific model repetition across all time bins and videos
        current_data = model_features[:, :, :, model_rep]  # shape: [time_bins, videos, sites]
        current_data = current_data[time_bin]  # shape: [videos, sites]

        assert current_data.shape[0] == object_labels.shape[0]

        # regress using linear regression with 10-fold cross-validation
        preds = get_regressions(current_data, object_labels, nrfolds=10, seed=np.random.randint(1000), standardize=False)

        # Store predictions for
        model_predictions[split_rep, time_bin, :, model_rep] = preds

        print(f'Process #{process_id}: rep: {model_rep+1} - split: {split_rep+1}/{split_repetitions}')


if __name__ == "__main__":
    
    model_features_path = '[your file containing ANN model features]'
    split_repetitions = 10
    hpc = True
    scores_save_path = '../../scores'

    features_filename = model_features_path.split('/')[-1]
    features_file = h5py.File(model_features_path, 'r')
    print("Keys:", list(features_file.keys()))

    # INPUT FILE SHAPE: [n_frames, videos, units, model repetitions]
    model_features = np.array(features_file['features'])
   
    n_time_bins = model_features.shape[0]

    # Create list of all (split_rep, time_bin, model_rep) tuples to process
    parallelizations = []
    for split_rep in range(split_repetitions):
        for time_bin in range(n_time_bins):
            for model_rep in range(model_features.shape[3]):
                parallelizations.append((split_rep, time_bin, model_rep))

    n_parallelizations = len(parallelizations)

    # Load object direction labels indexed by video
    object_labels = np.genfromtxt('[your label file]', delimiter='\n', dtype=np.int64)
    assert object_labels.shape[0] == model_features.shape[1]

    # Create shared array for results across processes
    m = MyManager()
    m.start()

    # OUTPUT ARRAY SHAPE: [split_repetitions, n_frames, n_videos, model repetitions]
    classification_scores = m.np_empty((split_repetitions, model_features.shape[0], model_features.shape[1], model_features.shape[3]))
    classification_scores[:] = np.nan

    model_preds = np.zeros((split_repetitions, model_features.shape[0], model_features.shape[1], model_features.shape[3]))
    model_preds[:] = np.nan

    # Distribute work across available CPU cores
    num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1)) if hpc else mp.cpu_count()
    chunk_size = n_parallelizations // num_processes
    
    # Launch worker processes
    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = n_parallelizations if i == num_processes - 1 else (i + 1) * chunk_size
        p = mp.Process(target=compute_preds, args=(i, parallelizations, model_preds, 
                                                     model_features, object_labels, 
                                                     split_repetitions, start, end))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

    classification_scores = np.array(classification_scores)

    for sr in range(split_repetitions):
        for r in range(model_features.shape[3]):
            preds = np.transpose(model_preds[sr, :, :, r], (1, 0))
            i1_all = speed_accuracy(model_preds, object_labels)
            classification_scores[sr, :, :, r] = np.transpose(i1_all, (1, 0))


    # Save results
    os.makedirs(os.path.join(scores_save_path, features_filename), exist_ok=True)

    save_file_name = f"decode_ann_obj_speed_linear.h5" 
    output_h5_path = os.path.join(scores_save_path, features_filename, save_file_name)
    with h5py.File(output_h5_path, 'w') as h5_file:
        h5_file.create_dataset("i1_all", data=classification_scores)
        print(f"Saved scores to {output_h5_path}, shape: {classification_scores.shape}")

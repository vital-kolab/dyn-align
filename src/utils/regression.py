"""
Utility functions for regression analysis on video data.
"""
import numpy as np
from utils.regression_metrics import pls_regress, get_train_test_indices


def speed_accuracy(predictions, vel):
    """
    Calculate accuracy of predictions based on relative speed ordering.
    
    Args:
        predictions: Array of shape (n_videos, n_time_bins) with predicted values
        vel: Array of video velocities
        
    Returns:
        performance: Array of shape (n_videos, n_time_bins) with accuracy scores
    """
    n_videos, n_time_bins = predictions.shape
    # Create array to store per-comparison accuracy
    perf = np.full((n_videos, n_videos, n_time_bins), np.nan)

    # Compare predictions between every pair of videos
    for i in range(n_videos):
        for j in range(n_videos):
            if i != j:
                # For each time bin, check if predictions preserve velocity ordering
                for k in range(n_time_bins):
                    if vel[i] > vel[j]:
                        # If video i is faster, prediction[i] should be greater
                        perf[i, j, k] = 1 if predictions[i, k] > predictions[j, k] else 0
                    else:
                        # Otherwise, prediction[i] should be less than or equal
                        perf[i, j, k] = 1 if predictions[i, k] <= predictions[j, k] else 0

    # Average accuracy across all video pairs
    performance = np.nanmean(perf, axis=1)
    return performance


def get_regressions(rates, targets, ncomp=10, nrfolds=10, seed=0, standardize=False):
    """
    Perform k-fold cross-validated PLS regression.
    
    Args:
        rates: Input feature array
        targets: Target values to predict
        ncomp: Number of PLS components
        nrfolds: Number of cross-validation folds
        seed: Random seed for reproducibility
        standardize: Whether to standardize features
        
    Returns:
        ypred: Full predictions array with NaNs replaced by fold predictions
    """
    nrImages = rates.shape[0]
    # Initialize prediction array
    ypred = np.arange(nrImages, dtype=float)
    ypred[:] = np.nan
    
    # Iterate through each fold
    for i in range(nrfolds):
        # Get train/test split for current fold
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed)
        x_train, y_train, x_test = rates[train], targets[train], rates[test]
        
        # Standardize features if requested
        if standardize:
            x_train_mean, x_train_std = np.nanmean(x_train, 0), np.nanstd(x_train, 0)
            x_train = (x_train - x_train_mean[np.newaxis, :]) / x_train_std[np.newaxis, :]
            x_test = (x_test - x_train_mean[np.newaxis, :]) / x_train_std[np.newaxis, :]

        # Train PLS model and predict on test set
        pred = pls_regress(x_train, y_train, x_test, ncomp=ncomp)
        # Store predictions in corresponding test indices
        np.put(ypred, test, pred)
     
    return ypred

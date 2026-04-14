"""
Classification utilities for training and evaluating various classifiers.
Supports Linear Discriminant Analysis (LDA), and RNN-based models with cross-validation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.regression_metrics import get_train_test_indices
from utils.rnn_classification import ShallowRNN

def compute_i1(probs, targets, return_scores=False):
    """
    Compute I1 accuracy metric: average ratio of correct class probability 
    to sum of correct and incorrect class probabilities.
    
    Parameters
    ----------
    probs : ndarray
        Probability predictions of shape (n_samples, n_classes)
    targets : ndarray
        Ground truth labels of shape (n_samples,)
    return_scores : bool
        If True, return per-sample scores
        
    Returns
    -------
    i1_score_mean : float
        Mean I1 score across all samples
    i1_score_std : float
        Standard error of the mean
    i1_scores : ndarray, optional
        Per-sample I1 scores if return_scores=True
    """
    class_labels = np.unique(targets)
    i1_scores = np.zeros(targets.shape[0]) + np.nan

    for i in range(targets.shape[0]):
        img_gt_class = int(targets[i])
        img_score = probs[i, img_gt_class]
        acc_dist = np.zeros(len(class_labels)) + np.nan 
        
        # Compute ratio of correct class probability to sum with each other class
        for c in class_labels:
            if c == img_gt_class:
                continue
            acc_dist[int(c)] = img_score / (img_score + probs[i, int(c)])

        i1_scores[i] = np.nanmean(acc_dist)

    i1_score_mean = np.nanmean(i1_scores)
    i1_score_std = np.nanstd(i1_scores) / np.sqrt(i1_scores.shape[0])
    print(f"I1 accuracy: {i1_score_mean:.4f}+-{i1_score_std:.4f}")
    
    if return_scores:
        return i1_score_mean, i1_score_std, i1_scores
    else:
        return i1_score_mean, i1_score_std

def classify_lda(X_train, Y_train, X_test, ncomp=20, return_weights=False):
    """
    Linear Discriminant Analysis classifier using One-vs-Rest strategy.
    
    Parameters
    ----------
    X_train : ndarray of shape (n_train_samples, n_features)
        Training features
    Y_train : ndarray of shape (n_train_samples,)
        Training labels
    X_test : ndarray of shape (n_test_samples, n_features)
        Test features
    ncomp : int, optional
        Number of components for LDA. Default is 20.
    return_weights : bool, optional
        If True, return classifier weights. Default is False.

    Returns
    -------
    Y_test_pred : ndarray
        Predicted class labels
    Y_test_prob : ndarray of shape (n_test_samples, n_classes)
        Probability estimates for each class
    weights : ndarray, optional
        Classifier weights if return_weights=True
    """
    lda = OneVsRestClassifier(LinearDiscriminantAnalysis(n_components=ncomp, solver='lsqr', shrinkage='auto'))
    lda.fit(X_train, Y_train)

    Y_test_pred = lda.predict(X_test)
    Y_test_prob = lda.predict_proba(X_test)

    if return_weights:
        weights = [est.coef_ for est in lda.estimators_]
        weights = np.concatenate(weights)
        return Y_test_pred, Y_test_prob, weights
    else:   
        return Y_test_pred, Y_test_prob


def get_classifications(rates, targets, model='logreg', nclasses=10, return_weights=False, ncomp=10, nrfolds=10, seed=0, standardize=False):
    """
    Perform k-fold cross-validation classification with logistic regression or LDA.
    
    Parameters
    ----------
    rates : ndarray of shape (n_samples, n_features)
        Input features
    targets : ndarray of shape (n_samples,)
        Target labels
    model : str, optional
        Model type: 'logreg' or 'lda'. Default is 'logreg'
    nclasses : int, optional
        Number of output classes. Default is 10
    return_weights : bool, optional
        Not used. Default is False
    ncomp : int, optional
        Number of components for dimensionality reduction. Default is 10
    nrfolds : int, optional
        Number of cross-validation folds. Default is 10
    seed : int, optional
        Random seed for reproducibility. Default is 0
    standardize : bool, optional
        If True, standardize features using training set statistics. Default is False
        
    Returns
    -------
    ypred : ndarray
        Predicted labels for all samples
    yprob : ndarray of shape (n_samples, n_classes)
        Probability predictions for all samples
    """
    nrImages = rates.shape[0]
    ypred = np.arange(nrImages, dtype=float)
    ypred[:] = np.nan
    yprob = np.zeros((nrImages, nclasses), dtype=float)
    yprob[:,:] = np.nan
    
    for i in range(nrfolds):
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed)
        
        x_train, y_train, x_test = rates[train], targets[train], rates[test]
        
        # Standardize features based on training set statistics
        if standardize:
            x_train_mean, x_train_std = np.nanmean(x_train, 0), np.nanstd(x_train, 0)
            x_train = (x_train - x_train_mean[np.newaxis, :]) / x_train_std[np.newaxis, :]
            x_test = (x_test - x_train_mean[np.newaxis, :]) / x_train_std[np.newaxis, :]

        pred, prob = classify_lda(x_train, y_train, x_test, ncomp=ncomp)
        
        np.put(ypred, test, pred)
        yprob[test,:] = prob
     
    return ypred, yprob

def rnn_classify(X_train, Y_train, X_test, model_config=None):
    """
    RNN-based classification using ShallowRNN model.
    
    Parameters
    ----------
    X_train : ndarray of shape (n_train_samples, seq_length, n_features)
        Training sequences
    Y_train : ndarray of shape (n_train_samples,)
        Training labels
    X_test : ndarray of shape (n_test_samples, seq_length, n_features)
        Test sequences
    model_config : dict, optional
        Configuration dictionary with keys: hidden_dim, output_dim, model, 
        num_layers, patience, max_epochs, learning_rate, verbose

    Returns
    -------
    Y_test_pred : ndarray
        Predicted class labels
    Y_test_probs : ndarray
        Probability estimates for each class
    """
    model = ShallowRNN(input_dim=X_train.shape[2], 
                        hidden_dim=100 if model_config is None else model_config['hidden_dim'], 
                        output_dim=8 if model_config is None else model_config['output_dim'], 
                        model='gru' if model_config is None else model_config['model'],
                        num_layers=1 if model_config is None else model_config['num_layers'], 
                        patience=10 if model_config is None else model_config['patience'], 
                        max_epochs=300 if model_config is None else model_config['max_epochs'],
                        batch_size=X_train.shape[0],
                        learning_rate=1e-2 if model_config is None else model_config['learning_rate'],
                        verbose=True if model_config is None else model_config['verbose'])
    model.fit(X_train, Y_train)

    Y_test_pred, Y_test_probs = model.predict(X_test, return_scores=True)
    
    return Y_test_pred, Y_test_probs

def rnn_classify_rgb_afv(X_train, Y_train, X_test, X_test_afv, model_config=None):
    """
    RNN-based classification on original and AFV test sets.
    
    Parameters
    ----------
    X_train : ndarray
        Training sequences
    Y_train : ndarray
        Training labels
    X_test : ndarray
        Original test sequences
    X_test_afv : ndarray
        AFV augmented test sequences
    model_config : dict, optional
        RNN configuration dictionary

    Returns
    -------
    Y_test_pred : ndarray
        Predictions on original test set
    Y_test_scores : ndarray
        Probabilities on original test set
    Y_test_pred_afv : ndarray
        Predictions on AFV test set
    Y_test_pred_afv_scores : ndarray
        Probabilities on AFV test set
    """
    model = ShallowRNN(input_dim=X_train.shape[2], 
                        hidden_dim=100 if model_config is None else model_config['hidden_dim'], 
                        output_dim=8 if model_config is None else model_config['output_dim'], 
                        model='gru' if model_config is None else model_config['model'],
                        num_layers=1 if model_config is None else model_config['num_layers'], 
                        patience=10 if model_config is None else model_config['patience'], 
                        max_epochs=300 if model_config is None else model_config['max_epochs'],
                        batch_size=X_train.shape[0],
                        learning_rate=1e-2 if model_config is None else model_config['learning_rate'],
                        verbose=True if model_config is None else model_config['verbose'])
    model.fit(X_train, Y_train)

    Y_test_pred, Y_test_scores = model.predict(X_test, return_scores=True)
    Y_test_pred_afv, Y_test_pred_afv_scores = model.predict(X_test_afv, return_scores=True)
    
    return Y_test_pred, Y_test_scores, Y_test_pred_afv, Y_test_pred_afv_scores


def get_rnn_classifications_rgb_afv(rates, rates_afv, targets, n_classes=8, ncomp=10, nrfolds=10, seed=0, model_config=None, standardize=False):
    """
    k-fold cross-validation with RNN on original and AFV sequences.
    
    Parameters
    ----------
    rates : ndarray
        Original sequences
    rates_afv : ndarray
        AFV augmented sequences
    targets : ndarray
        Target labels
    n_classes : int, optional
        Number of classes. Default is 8
    ncomp : int, optional
        Not used. Default is 10
    nrfolds : int, optional
        Number of folds. Default is 10
    seed : int, optional
        Random seed. Default is 0
    model_config : dict, optional
        RNN configuration. Default is None
    standardize : bool, optional
        Whether to standardize features. Default is False
        
    Returns
    -------
    ypred : ndarray
        Predictions on original data
    ypred_scores : ndarray
        Probabilities on original data
    ypred_afv : ndarray
        Predictions on AFV data
    ypred_afv_scores : ndarray
        Probabilities on AFV data
    """
    nrImages = rates.shape[0]
    ypred = np.arange(nrImages, dtype=float)
    ypred[:] = np.nan
    ypred_scores = np.zeros((nrImages, n_classes), dtype=float) + np.nan

    ypred_afv = np.arange(nrImages, dtype=float)
    ypred_afv[:] = np.nan
    ypred_afv_scores = np.zeros((nrImages, n_classes), dtype=float) + np.nan
    
    for i in range(nrfolds):
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed)

        x_train, x_test, x_test_afv = rates[train], rates[test], rates_afv[test]
        
        if standardize:
            x_train_mean, x_train_std = np.nanmean(np.reshape(x_train, (-1, x_train.shape[-1])), 0), np.nanstd(np.reshape(x_train, (-1, x_train.shape[-1])), 0)
            x_train = (x_train - x_train_mean[np.newaxis, np.newaxis, :]) / x_train_std[np.newaxis, np.newaxis, :]
            x_test = (x_test - x_train_mean[np.newaxis, np.newaxis, :]) / x_train_std[np.newaxis, np.newaxis, :]
            x_test_afv = (x_test_afv - x_train_mean[np.newaxis, np.newaxis, :]) / x_train_std[np.newaxis, np.newaxis, :]

        pred_rgb, pred_scores, pred_avf, pred_afv_scores = rnn_classify_rgb_afv(x_train, targets[train], x_test, x_test_afv, model_config=model_config)
        
        np.put(ypred, test, pred_rgb)
        ypred_scores[test] = pred_scores

        np.put(ypred_afv, test, pred_avf)
        ypred_afv_scores[test] = pred_afv_scores

    return ypred, ypred_scores, ypred_afv, ypred_afv_scores

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:30:00 2020

@author: kohitij
"""

import numpy as np
import torch
from scipy import stats
from utils.regression_metrics import pls_regress, recurrent_regress, get_train_test_indices
from utils.correlation_metrics import get_splithalves, spearmanbrown_correction, get_splithalf_corr

def get_modelpredictions(rates, model_features, ncomp=10, nrfolds=10, seed=0, standardize=False):
    
    nrImages = rates.shape[0]
    ypred = np.arange(nrImages, dtype=float)
    ypred[:] = np.nan
    
    for i in range(nrfolds): # at each fold, we predict disjoint test images and store them
        # print('fold number is: ' + str(i))
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed) # return indices of train and test samples

        x_train, x_test = model_features[train,:], model_features[test,:]
        if standardize:
            x_train_mean, x_train_std = np.nanmean(x_train, 0), np.nanstd(x_train, 0)
            x_train = (x_train - x_train_mean[np.newaxis, :]) / x_train_std[np.newaxis, :]
            x_test = (x_test - x_train_mean[np.newaxis, :]) / x_train_std[np.newaxis, :]

        # np.nanmean(rates[train,:],axis=1) -> Compute the arithmetic mean along the specified axis, ignoring NaNs.
        pred = pls_regress(x_train, np.nanmean(rates[train,:], 1), x_test, ncomp=ncomp)
        
        np.put(ypred, test, pred)

    # at the end of the for loop, ypred contains the predicted neural recordings for all the images in the dataset
     
    return ypred


def get_modelpredictions_recurrent(rates, model_features, ncomp=10, nrfolds=10, seed=0, model_config=None, standardize=False):
    
    nrImages = rates.shape[0]
    ypred = np.arange(nrImages, dtype=float)
    ypred[:] = np.nan
    
    for i in range(nrfolds): # at each fold, we predict disjoint test images and store them
        # print('fold number is: ' + str(i))
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed) # return indices of train and test samples

        # np.nanmean(rates[train,:],axis=1) -> Compute the arithmetic mean along the specified axis, ignoring NaNs.
        x_train, x_test = model_features[train,:], model_features[test,:]
        if standardize:
            x_train_mean, x_train_std = np.nanmean(np.reshape(x_train, (-1, x_train.shape[-1])), 0), np.nanstd(np.reshape(x_train, (-1, x_train.shape[-1])), 0)
            x_train = (x_train - x_train_mean[np.newaxis, np.newaxis, :]) / x_train_std[np.newaxis, np.newaxis, :]
            x_test = (x_test - x_train_mean[np.newaxis, np.newaxis, :]) / x_train_std[np.newaxis, np.newaxis, :]

        pred = recurrent_regress(x_train, np.nanmean(rates[train,:],axis=1), x_test, ncomp=ncomp, model_config=model_config)

        #print('TEST LOSS', torch.nn.functional.mse_loss(torch.tensor(pred), torch.tensor(np.nanmean(rates[test,:],axis=1))))
        
        np.put(ypred, test, pred)

    # at the end of the for loop, ypred contains the predicted neural recordings for all the images in the dataset
     
    return ypred


def get_model_neural_splithalfcorr(rates,model_features,ncomp=10,nrfolds=10,seed=0, standardize=False):
    sp1, sp2, _, _ = get_splithalves(rates,ax=1)
    shc = get_splithalf_corr(rates,ax=1)
     # model  predictions split half 1 -- 
    p1 = get_modelpredictions(sp1,model_features, nrfolds=nrfolds, ncomp = ncomp, seed=seed, standardize=standardize)
     # model  predictions split half 1 -- 
    p2 = get_modelpredictions(sp2,model_features, nrfolds=nrfolds, ncomp = ncomp, seed=seed, standardize=standardize)
    #print(stats.pearsonr(p1.T,p2.T)[0])
    model_shc = spearmanbrown_correction(stats.pearsonr(p1.T,p2.T)[0])
    neural_shc = spearmanbrown_correction(shc['split_half_corr'])
    return model_shc, neural_shc

def get_model_neural_splithalfcorr_recurrent(rates, model_features, ncomp=10, nrfolds=10, seed=0, model_config=None, standardize=False):
    sp1, sp2, _, _ = get_splithalves(rates,ax=1)
    shc = get_splithalf_corr(rates,ax=1)
     # model  predictions split half 1 -- 
    p1 = get_modelpredictions_recurrent(sp1, model_features, nrfolds=nrfolds, ncomp = ncomp, seed=seed, model_config=model_config, standardize=standardize)
     # model  predictions split half 1 -- 
    p2 = get_modelpredictions_recurrent(sp2, model_features, nrfolds=nrfolds, ncomp = ncomp, seed=seed, model_config=model_config, standardize=standardize)
    #print(stats.pearsonr(p1.T,p2.T)[0])
    #print(stats.pearsonr(p1.T,p2.T)[0])
    #print(p1,p2)
    model_shc = spearmanbrown_correction(stats.pearsonr(p1.T,p2.T)[0])
    neural_shc = spearmanbrown_correction(shc['split_half_corr'])
    return model_shc, neural_shc
    
def predictivity(x,y,rho_xx, rho_yy):
    """
    

    Parameters
    ----------
    x : float np array ,
        e.g. measured firing rates  [images x trials]
    y : float np array 
        ,e.g. model predictions for [images x 1]
    rho_xx : float64 scalar
        internal reliablity of x
    rho_yy : float64 scalar
        internal reliablity of y

    Returns
    -------
    ev : float64
        % EV
    raw_corr : float64
        % raw Pearson correlated
    corrected_raw_corr : float64
        % noise corrected Pearson Correlation
    """
    numerator = stats.pearsonr(x, y)[0]
    denominator = np.sqrt(np.multiply(rho_xx, rho_yy)) # denominstor < 0.4-0-5 return nan
    raw_corr = numerator
    corrected_raw_corr = numerator/denominator
    ev = ((corrected_raw_corr)**2)*100
    return ev, raw_corr, corrected_raw_corr

def predictivity_new(x,y,rho_xx, rho_yy):
    """
    

    Parameters
    ----------
    x : float np array ,
        e.g. measured firing rates  [images x trials]
    y : float np array 
        ,e.g. model predictions for [images x 1]
    rho_xx : float64 scalar, neural correlation
        internal reliablity of x
    rho_yy : float64 scalar, model correlation
        internal reliablity of y

    Returns
    -------
    ev : float64
        % EV
    raw_corr : float64
        % raw Pearson correlated
    corrected_raw_corr : float64
        % noise corrected Pearson Correlation
    """
    pearson_corr = stats.pearsonr(x, y)
    numerator = pearson_corr[0]
    ci_low, ci_high = pearson_corr.confidence_interval()
    denominator = np.sqrt(np.multiply(rho_xx, rho_yy)) # denominstor < 0.4-0-5 return nan
    # nan appaens when denominator is negative, which happens when rho_xx or rho_yy is negative
    # rho_xx is the internal reliability of x (neural data) and rho_yy is the internal reliability of y (model predictions)
    # if rho_xx is negative, it means that the neural data is not reliable, and if rho_yy is negative, it means that the model predictions are not reliable
    #print(rho_xx, rho_yy, denominator) # ev is high > 1.0 when rho_yy (i.e. model split-half correlation) is low 
    #if denominator < 0.4:
    #    denominator = np.nan
    #    ci_low, ci_high = np.nan, np.nan
    raw_corr = numerator ** 2
    corrected_raw_corr = numerator / denominator
    ev = ((corrected_raw_corr)**2) #*100

    corrected_ci_low = ci_low / denominator
    corrected_ci_high = ci_high / denominator
    corrected_ci_low = corrected_ci_low ** 2
    corrected_ci_high = corrected_ci_high ** 2

    return ev, raw_corr, corrected_raw_corr, corrected_ci_low, corrected_ci_high 
     
  
    
    
    
    

import numpy as np
from scipy import stats
import random
import math

def get_splithalf_corr(var,ax=1,type='spearman',seed=0): 
    """
    specify the variable (var) for which splits are required, 
    along which axis (ax)?
    which correlation method do you want (type)?
    """
    _,_, split_mean1, split_mean2 = get_splithalves(var, ax=ax, seed=seed)
    if(type=='spearman'):
        split_half_correlation = stats.spearmanr(split_mean1,split_mean2) #get the Spearman Correlation
        
        r = split_half_correlation[0]
        num = split_mean1.shape[0]
        stderr = 1.0 / math.sqrt(num - 3)
        delta = 1.96 * stderr
        lower = math.tanh(math.atanh(r) - delta)
        upper = math.tanh(math.atanh(r) + delta)
    
    else:
        split_half_correlation = stats.pearsonr(split_mean1, split_mean2) #get the Pearson Correlation
        lower, upper = None, None
    return {'split_half_corr':split_half_correlation[0],
            'p-value':split_half_correlation[1],
            'type':type,
            'ci_lower' : lower,
            'ci_upper' : upper
            }

def get_splithalves(var, ax=1, seed=0):
    """
    Parameters
    ----------
    var : TYPE
        DESCRIPTION.
    ax : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    split_mean1 : TYPE
        DESCRIPTION.
    split_mean2 : TYPE
        DESCRIPTION.

    """
    np.random.seed(seed)
    shuffled = var.copy()
    np.apply_along_axis(np.random.shuffle, ax, var)   # randomly shuffle the array along the specific axis (ax)    
     
    split1, split2 = np.array_split(shuffled, 2, axis=ax) # split the aray into 2 halves
    split_mean1 = np.nanmean(split1,axis=ax) # split half 1
    split_mean2 = np.nanmean(split2,axis=ax) # split half 2
    return split1, split2, split_mean1, split_mean2

def get_splithalves__(var, ax=1, seed=0):
    """
    Randomly split the array along the specified axis and return the two halves and their means.

    Parameters
    ----------
    var : ndarray
        The input array to split.
    ax : int, optional
        The axis along which to split. Default is 1.
    rng : np.random.Generator, optional
        Numpy random number generator for reproducibility. If None, defaults to np.random.default_rng().

    Returns
    -------
    split1, split2 : ndarray
        The two split halves.
    split_mean1, split_mean2 : ndarray
        The means of the two split halves along the specified axis.
    """
    np.random.seed(seed)

    # Transpose var so that the split axis becomes axis 0 (easier for shuffling along slices)
    var = np.swapaxes(var, 0, ax)
    
    shuffled = var.copy()
    np.apply_along_axis(np.random.shuffle, 0, shuffled)   # randomly shuffle the array along the specific axis (ax) 
    #rng.shuffle(shuffled, axis=0)  # shuffle along the new 0th axis (original ax)
    
    split1, split2 = np.array_split(shuffled, 2, axis=0)
    split_mean1 = np.nanmean(split1, axis=0)
    split_mean2 = np.nanmean(split2, axis=0)

    # Swap axes back to original configuration
    return (
        np.swapaxes(split1, 0, ax),
        np.swapaxes(split2, 0, ax),
        np.swapaxes(split_mean1, 0, ax - 1 if ax > 0 else 0),
        np.swapaxes(split_mean2, 0, ax - 1 if ax > 0 else 0),
    )
    
def get_splithalves__(var, ax=1, seed=0):
    """
    Parameters
    ----------
    var : TYPE
        DESCRIPTION.
    ax : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    split_mean1 : TYPE
        DESCRIPTION.
    split_mean2 : TYPE
        DESCRIPTION.

    """
    np.random.seed(seed)
    
    l1 = np.random.choice(var.shape[ax], size=int(var.shape[ax] / 2), replace=False)
    l2 = np.setdiff1d(np.arange(var.shape[ax]), l1)

    split1 = np.take(var, l1, axis=ax)
    split2 = np.take(var, l2, axis=ax)

    split_mean1 = np.nanmean(split1,axis=ax) # split half 1
    split_mean2 = np.nanmean(split2,axis=ax) # split half 2

    return split1, split2, split_mean1, split_mean2


def spearmanbrown_correction(var): # Spearman Brown Correct the correlation value
    spc_var = (2*var)/(1+var)
    return spc_var


def get_correlation_noise_corrected(var1,var2,nrbs=50,correction_method='spearmanBrown'):
    """
        Parameters
    ----------
    var1 :  variable 1 for correlation (2d array): 2nd dimension has to be trials (repetitions)
    var2 : variable 2 for correlation (2d array): 2nd dimension has to be trials (repetitions)
    nrbs : number of bootstrap repeats. optional, The default is 50.
    correction_method : Split correction applied, optional, The default is 'spearmanBrown'.

    Returns
    -------
    corrected_corr : 1d array of corrected pearson correlation values
        
    """
    corrected_corr = np.empty([nrbs, 1], dtype=float)
    for i in range(nrbs):
        sh_corr_var1 = get_splithalf_corr(var1)
        sh_corr_var2 = get_splithalf_corr(var2)
        den = np.sqrt(sh_corr_var1['split_half_corr']*sh_corr_var2['split_half_corr'])
        if(correction_method=='spearmanBrown'):
            num = stats.pearsonr(np.nanmean(var1,axis=1),np.nanmean(var2,axis=1))
        else:
            var1_split= var1[:,random.sample(list(np.arange(0,np.size(var1, axis=1),1)),int(np.round(np.size(var1, axis=1)/2)))]
            var2_split = var2[:,random.sample(list(np.arange(0,np.size(var2, axis=1),1)),int(np.round(np.size(var2, axis=1)/2)))]
            num = stats.pearsonr(np.nanmean(var1_split,axis=1),np.nanmean(var2_split,axis=1))
        corrected_corr[i] = num[0]/den
    return corrected_corr
        
        
    
def main():
    if __name__ == "__main__":
        main()
     
    
    
  

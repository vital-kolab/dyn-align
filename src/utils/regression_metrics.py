
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from utils.rnn_regression import ShallowRNN

"""
Cross decomposition algorithms find the fundamental relations between two matrices (X and Y). 
They are latent variable approaches to modeling the covariance structures in these two spaces. 
They will try to find the multidimensional direction in the X space that explains the maximum multidimensional variance direction in the Y space. 
In other words, PLS projects both X and Y into a lower-dimensional subspace such that the covariance between transformed(X) and transformed(Y) is maximal.

Covariance in probability theory and statistics is a measure of the joint variability of two random variables.
The sign of the covariance, therefore, shows the tendency in the linear relationship between the variables.
"""
def pls_regress(X_train, Y_train, X_test, ncomp=20):
    """
    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    Y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    ncomp : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    Y_test_pred : TYPE
        DESCRIPTION.

    """
    pls2 = PLSRegression(n_components=ncomp)
    pls2.fit(X_train, Y_train)
    PLSRegression()
    Y_test_pred = pls2.predict(X_test)
    return Y_test_pred

def recurrent_regress(X_train, Y_train, X_test, ncomp=20, model_config=None):
    """
    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    Y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    ncomp : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    Y_test_pred : TYPE
        DESCRIPTION.

    """
    #print(X_train.shape, Y_train.shape)

    # input size expected by LSTM
    # X_train: [samples, time steps, features]
    # Y_train: [samples, 1]
    model = ShallowRNN(input_dim=X_train.shape[2], 
                        hidden_dim=25 if model_config is None else model_config['hidden_dim'], 
                        output_dim=1 if model_config is None else model_config['output_dim'], 
                        model='lstm' if model_config is None else model_config['model'],
                        patience=100 if model_config is None else model_config['patience'], 
                        max_epochs=100 if model_config is None else model_config['max_epochs'], #50,
                        batch_size=X_train.shape[0],
                        learning_rate=1e-2 if model_config is None else model_config['learning_rate'], #1e-2, #0.005,
                        verbose=True if model_config is None else model_config['verbose'])
    model.fit(X_train, Y_train.reshape(Y_train.shape[0], 1))

    # Make predictions
    Y_test_pred = model.predict(X_test)
    Y_test_pred = Y_test_pred.reshape((Y_test_pred.shape[0]))
    return Y_test_pred

# return disjoint indices of train and test samples, based on the totalIndices numebr fo indices
def get_train_test_indices(totalIndices, nrfolds=10,foldnumber=0, seed=1):
    """
    

    Parameters
    ----------
    totalIndices : TYPE
        DESCRIPTION.
    nrfolds : TYPE, optional
        DESCRIPTION. The default is 10.
    foldnumber : TYPE, optional
        DESCRIPTION. The default is 0.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    train_indices : TYPE
        DESCRIPTION.
    test_indices : TYPE
        DESCRIPTION.

    """
    
    np.random.seed(seed)
    inds = np.arange(totalIndices)
    np.random.shuffle(inds)
    splits = np.array_split(inds,nrfolds)
    test_indices = inds[np.isin(inds,splits[foldnumber])]
    train_indices = inds[np.logical_not(np.isin(inds, test_indices))]
    return train_indices, test_indices







def main():
    if __name__ == "__main__":
        main()    
    
    
    
    
    
    
    
    
    
    
    
    

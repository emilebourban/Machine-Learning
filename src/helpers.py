# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(path_dataset):
    """Load data and convert it to the metric system."""
    y = np.genfromtxt(path_dataset, delimiter=',', usecols=1, 
                      dtype=str, skip_header=1)
    cols = np.genfromtxt(path_dataset, delimiter=',', skip_footer=len(y), 
                         dtype=str)
    data = np.genfromtxt(path_dataset, delimiter=",",  
                         usecols=range(2, 32), skip_header=1)
    return y, data, cols

def clean_data(data, err_list, err_lim):
    '''  '''
    for i in range(data.shape[1]):
    if (err_list[i]) > err_lim:
        data = np.delete(data, [i-n_del],  axis=1)
        data_te = np.delete(data_te, [i-n_del],  axis=1)
        n_del +=1
    return data

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

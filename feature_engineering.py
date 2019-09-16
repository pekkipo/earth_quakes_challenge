#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:51:30 2019

@author: pedro
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.signal.windows as window

import entropy.entropy as S

def accumulate_energy(E : float, x : float) -> float:
    """
    Adds the energy of a single signal sample to an existing
    energy pool
    """
    return E+x**2

def signal_energy(array : np.array) -> float:
    """
    Optimized energy calculation for an array
    It is pressumed the array signals are 1D columns
    """
    return np.sum(array ** 2, axis=0)

def signal_svd_entropy(array : np.array) -> float:
    """
    Computes the signal entropy using a single value decomposition method
    """
    return S.svd_entropy(array, order=3, delay=1, normalize=True)
    
def signal_fourier(array : np.array) -> np.array:
    """
    Computes the signal's fourier transform with a hardcoded signal frequency 
    and cutoff frequency
    """
    freq = 909090893.1880873
    f, S = signal.welch(array, freq, nperseg=len(array)) # 150
    limit = sum(f<1.4e8)
    return S[0:limit]

def pred_mean_slope(array : np.array) -> float:
    return np.nanmean(np.diff(array))

def accumulate_signal_energy(array : np.array) -> np.array:
    """
    Optimized cumulative energy calculation for an array
    """
    return np.cumsum(array ** 2, axis=0)

def accumulate_signal_energy_gaussian(array : np.array) -> np.array:
    """
    Computes the cumulative energy from an array considering a Gaussian
    weighted window. This is equivalent to a Gaussian filtered signal
    Parameters of the window: 100 points, std=7 
    """
    # TODO: optimize window based on noise/NN
    gaussian_window = window.gaussian(100, std=2)
    gaussian_window = gaussian_window/np.sum(gaussian_window)
    filtered = signal.convolve(array, gaussian_window)
    return np.cumsum(filtered ** 2, axis=0)

def accumulate_signal_energy_butter(array : np.array) -> np.array:
    """
    Computes the cumulative energy from an 1D array after performing
    a Butterworth filtering of the input signal.
    """
    # TODO: optimize filter params based on noise and the input freq
    N = 3
    Wd = 0.2
    b, a = signal.butter(N, Wd)
    filtered = signal.lfilter(b, a, array)
    return np.cumsum(filtered ** 2, axis=0)

def derivative(x : np.array, t : np.array) -> np.array:
    """
    Computes simple numpy array derivative given time t
    """
    return np.gradient(x, t)

def create_decreasing_intervals(array : np.array, N_interv : int) -> list:
    """
    Creates N intervals from an array, returns a list of arrays
    """
    size = len(array)
    if not size % N_interv == 0:
        raise ValueError("Length of data not divisible by num of intervals")
    
    delta = int(size / N_interv)
    array_list = []
    for ii in range(N_interv):
        data = array[(size-delta*(ii+1)):size]
        array_list.append(data)
    return array_list

def create_static_features(array : np.array, N_interv : int, functions : list) -> dict:
    """
    Creates features based on a decreasing interval
    """
    intervals = create_decreasing_intervals(array, N_interv)
    feats = {}
    for ii in range(N_interv):
        for jj in range(len(functions)):
            feat_name = '{}_{}'.format(str(ii),functions[jj].__name__)
            feats[feat_name] = functions[jj](intervals[ii])
    return feats

def unravel_feature(key : str, vals) -> dict:
    if not hasattr(vals, '__len__') or len(vals)==1:
        return {key : vals}
    else:
        return {key+'_'+str(ii) : val for ii, val in enumerate(vals)}

def create_rolling_features(array : np.array, functions : list) -> dict:
    feats = {}
    for jj in range(len(functions)):
        feat_name = functions[jj].__name__
        feats.update(unravel_feature(feat_name, functions[jj](array)))
    return feats

def create_rolling_features_dataframe(array : np.array) -> pd.DataFrame:
    l_10 = lambda x : np.percentile(x,q=10) 
    l_10.__name__ = 'perc_10'
    l_25 = lambda x : np.percentile(x,q=25) 
    l_25.__name__ = 'perc_25'
    l_50 = lambda x : np.percentile(x,q=50) 
    l_50.__name__ = 'perc_50'
    l_75 = lambda x : np.percentile(x,q=75) 
    l_75.__name__ = 'perc_75'
    l_90 = lambda x : np.percentile(x,q=90) 
    l_90.__name__ = 'perc_90'
    fourier = lambda x : signal_fourier(x)
    fourier.__name__ = 'fourier'
    functions = [
            np.mean,
            np.median,
            np.std,
            np.max,
            np.min,
            np.var,
            np.ptp, #Peak-to-peak is like range
            l_10,
            l_25,
            l_50,
            l_75,
            l_90,
            signal_energy,
            signal_svd_entropy,
            fourier,
            ]
    return pd.DataFrame(create_rolling_features(array, 
                                               functions), index=[0])

def create_static_features_dataframe(array : np.array) -> pd.DataFrame:
    """
    Creates a dataframe with features for easy handling
    """
    #HARDCODED
    N_interv = 10
    
    l_10 = lambda x : np.percentile(x,q=10) 
    l_10.__name__ = 'perc_10'
    l_25 = lambda x : np.percentile(x,q=25) 
    l_25.__name__ = 'perc_25'
    l_50 = lambda x : np.percentile(x,q=50) 
    l_50.__name__ = 'perc_50'
    l_75 = lambda x : np.percentile(x,q=75) 
    l_75.__name__ = 'perc_75'
    l_90 = lambda x : np.percentile(x,q=90) 
    l_90.__name__ = 'perc_90'
    functions = [
            np.mean,
            np.median,
            np.std,
            np.max,
            np.min,
            np.var,
            np.ptp, #Peak-to-peak is like range
            l_10,
            l_25,
            l_50,
            l_75,
            l_90,
            signal_energy,
            signal_svd_entropy,
            ]
    return pd.DataFrame(create_static_features(array, 
                                               N_interv, 
                                               functions), index=[0])
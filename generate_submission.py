# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:08:27 2019

@author: aleks
"""

import glob
import pandas as pd
import os
import numpy as np

from feature_engineering import create_rolling_features, pred_mean_slope


path_to_test = 'data/test_feat_1500_big/'

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
        pred_mean_slope,
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
        ]

submit = {}

for file in glob.glob(path_to_test+'*.csv'):
    seg = os.path.basename(file).replace('feats_1500_','')
    feats = pd.read_csv(file, header=[0])
    
    # APPLY FIRST MODEL ON feats, GET A VECTOR OF 100 ELEMENTS
    lstm_prediction = np.zeros(0,100)
    
    feat_row = pd.DataFrame(create_rolling_features(lstm_prediction, 
                                                    functions), index=[0])
    
    # APPLY SECOND MODEL, GET FINAL VALUE
    final_prediction = 0
    
    submit[seg] = final_prediction
    
# PRINT DICT TO CSV
# EASY PEASY
    
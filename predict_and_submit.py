# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:29:53 2019

@author: aleks
"""


import params
import pandas as pd
from joblib import dump, load
import numpy as np


X_train = load("X_train.joblib")
rf = load("rf_model.joblib")


submission = pd.read_csv(params.submission_file, index_col = 'seg_id')

X_test = pd.DataFrame(columns=X_train.columns, index=submission.index)


## FE
from feature_engineering import signal_energy, signal_svd_entropy

for i in X_test.index:
    
    #Read in that segments csv file
    #By putting f before the string we can put any values between {} and it will be treated as a string
    seg = pd.read_csv('{}{}.csv'.format(params.test_folder, i)) 
                                            
    #Grab the acoustic_data values
    x = seg['acoustic_data'].values

    #These are the same features we calcuted on the training data
    X_test.loc[i, 'mean'] = np.mean(x)
    X_test.loc[i, 'median'] = np.median(x)
    X_test.loc[i, 'std'] = np.std(x)
    X_test.loc[i, 'max'] = np.max(x)
    X_test.loc[i, 'min'] = np.min(x)
    X_test.loc[i, 'var'] = np.var(x)
    X_test.loc[i, 'ptp'] = np.ptp(x)
    X_test.loc[i, '10p'] = np.percentile(x,q=10) 
    X_test.loc[i, '25p'] = np.percentile(x,q=25)
    X_test.loc[i, '50p'] = np.percentile(x,q=50)
    X_test.loc[i, '75p'] = np.percentile(x,q=75)
    X_test.loc[i, '90p'] = np.percentile(x,q=90)
    
    """
    
    for feat in headers:
        
        
        X_train.loc[i, feat] = 
    """
    
    
    
#Predict on the test data
test_predictions = rf.predict(X_test)

#Assign the target column in our submission to be our predictions
submission['time_to_failure'] = test_predictions

submission.to_csv(params.submission_out_file)

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:57:48 2019

@author: Q466091
"""




import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import os
from attr import dataclass
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
import json
from numpy import newaxis
import datetime as dt
from lstm_model import DataLoader, Model


@dataclass
class DataSet:
    """Model class for a dataset"""
    X: np.ndarray
    y: np.ndarray


import params
import pandas as pd
from sklearn.model_selection import train_test_split
import utils
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from joblib import dump, load

print("Start")
train_df = utils.read_file(params.train_data_path)
print("Data length is: {}".format(train_df.size))
#X_train, y_train, df_len = utils.load_and_split_train_file(params.train_data_path)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


# Use sampling from some kernel just for similicity
print("Dividing the dataset")
num_samples = int((len(train_df) / params.sample_length)) 

cols = ['mean','median','std','max',
        'min','var','ptp','10p',
        '25p','50p','75p','90p']

## ENERGY BASED FEATS
"""
train_features = load("data/train_features_decreasing.csv")
train_features = train_features.filter(['9_S','9_E'], axis=1)
headers = list(train_features.columns.values)

cols.extend(headers)
"""

# These are empty at this stage
print("Preparing empty dataframe")
X_train, y_train = utils.prepare_empty_dfs(train_df, num_samples, cols)

#Now we create the samples
for i in range(num_samples):
    
    #i*sample_length = the starting index (from train_df) of the sample we create
    #i*sample_length + sample_length = the ending index (from train_df)
    sample = train_df.iloc[i*params.sample_length:i*params.sample_length+params.sample_length]
    
    #Converts to numpy array
    x = sample['acoustic_data'].values
    
    #Grabs the final 'time_to_failure' value
    y = sample['time_to_failure'].values[-1]
    y_train.loc[i, 'time_to_failure'] = y
    
    
   #For every 150,000 rows, we make these calculations
    X_train.loc[i, 'mean'] = np.mean(x)
    X_train.loc[i, 'median'] = np.median(x)
    X_train.loc[i, 'std'] = np.std(x)
    X_train.loc[i, 'max'] = np.max(x)
    X_train.loc[i, 'min'] = np.min(x)
    X_train.loc[i, 'var'] = np.var(x)
    X_train.loc[i, 'ptp'] = np.ptp(x) #Peak-to-peak is like range
    X_train.loc[i, '10p'] = np.percentile(x,q=10) 
    X_train.loc[i, '25p'] = np.percentile(x,q=25) #We can also grab percentiles
    X_train.loc[i, '50p'] = np.percentile(x,q=50)
    X_train.loc[i, '75p'] = np.percentile(x,q=75)
    X_train.loc[i, '90p'] = np.percentile(x,q=90)
    
    
    # Adding energy based ones
    #for feat in headers:
    #    X_train.loc[i, feat] = train_features.loc[i, feat]


def split_data(X, y):
    X_tr, X_test, y_tr, y_test = train_test_split(X, y)

    train = DataSet(X=X_tr, y=y_tr)
    test = DataSet(X=X_test, y=y_test)
    
    return train, test

train, test = split_data(X_train, y_train)
data = DataLoader(train, test)



configs = json.load(open('config.json', 'r'))
input_dim = train.X.shape[1] 

model = Model(configs)
model.build_model(input_dim)


x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )


# in-memory training
model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size']
	)


x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )


#predictions1 = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])

#predictions2 = model.predict_sequence_full(x_test, configs['data']['sequence_length'])

predictions3 = model.predict_point_by_point(x_test)

print("All predictions are done")
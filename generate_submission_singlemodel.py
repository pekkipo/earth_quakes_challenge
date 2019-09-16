# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:08:27 2019

@author: aleks
"""

import glob
import pandas as pd
import os
import numpy as np

from joblib import load

import lightgbm as lgb
from joblib import dump, load
from lstm_model import DataLoader, Model
import json
from attr import dataclass
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import utils

path_to_test = 'data/test_feat_1500_big/'


@dataclass
class DataSet:
    X: np.ndarray
    y: np.ndarray
    

def scale(X):
    
    # normalize the dataset
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #scaler = load("scaler_lstm.joblib")
    X = scaler.fit_transform(X)
  
    return X

submit = {}
models_path = "classifiers/"

configs = json.load(open('config.json', 'r'))
lstm_path = "good_lstm.h5"
model_lstm = Model(configs)
model_lstm.load_h5_model('{}{}'.format(models_path, lstm_path))

for file in glob.glob(path_to_test+'*.csv'):
    seg = os.path.basename(file).replace('feats_1500_','')
    feats = pd.read_csv(file, header=[0])
    feats = feats.drop(columns=['Unnamed: 0'])
    
    """
    X = scale(feats)
    
    train = DataSet(X=X, y=X[:,0])
    data = DataLoader(None, train)
    
    X, _ = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
    
    predictions_lstm = model_lstm.predict_point_by_point(X)
    """
    #resized = 4000*(predictions_lstm-0.4).clip(0,20)
    #final_prediction = np.mean(resized)-0.01948129
    
    
    lgb_model = "LGBM_MODEL_1500_246feats.txt"
    lgbm = lgb.Booster(model_file='{}{}'.format(models_path, lgb_model))
    print("LGBM predictig...")
    predictions_lgbm = lgbm.predict(feats)
    print("Len y predictions lgbm: {}".format(len(predictions_lgbm)))
    
    #submit[seg] = predictions_lgbm[:-1] #final_prediction
    submit[seg] = predictions_lgbm[-1] #final_prediction
    
dump(submit, "sub_dict.joblib")
    
with open('submission.csv', 'w') as f:
    for key in submit.keys():
        #f.write("%s,%s\n"%(key,submit[key]))
        f.write("%s,%s\n"%(key,submit[key][-1])) # write only last value
    
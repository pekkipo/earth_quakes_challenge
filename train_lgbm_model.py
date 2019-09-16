# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:32:47 2019

@author: aleks

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import os
import gc
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import feature_engineering as fe
import get_models
import params

gc.enable()


NAME = 'lgbm'
    
def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path):
    
    model = get_models.get_lgbm_2() # Lol this will not work
    
    
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=3500, 
              early_stopping_rounds=3500) # why early stopping?
      
    cv_val = model.predict(X_val)
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, NAME, counter+1)
    model.booster_.save_model(save_to)
    
    return cv_val
    
def train_stage(df, y_df, df_ids, lgb_path):
    
    lgb_cv_result = np.zeros(df.shape[0])
    
    skf = StratifiedKFold(n_splits=params.num_folds, shuffle=True, random_state=42) # what is this doing?
    skf.get_n_splits(df_ids, y_df) # and this? -> splits the data in train and test?
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]
    
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path)
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    mae_lgb  = round(mean_absolute_error(y_df, lgb_cv_result), 6)
    print('\nLightGBM VAL MAE: {}'.format(mae_lgb))
    return 0
    
    
def prediction_stage(df, lgb_path, submit=True):
    
    lgb_models = sorted(os.listdir(lgb_path))
    lgb_result = np.zeros(df.shape[0])

    print('\nMake predictions...\n')
    
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df.values)

    lgb_result /= len(lgb_models)
    
    if submit:
        submission = pd.read_csv(params.submission_file)
        submission['time_to_failure'] = lgb_result
        submission.to_csv(params.submission_out_file, index=False)

    return 0
    

############ RUN

train_path = 'data/train.csv'
test_path  = 'data/test.csv'

print('Load Train Data.')
df_train = pd.read_csv(train_path)
print('\nShape of Train Data: {}'.format(df_train.shape))

print('Load Test Data.')
df_test = pd.read_csv(test_path)
print('\nShape of Test Data: {}'.format(df_test.shape))




#Create dir for models
#os.mkdir(lgb_path)

print('Train Stage.\n')
train_stage(df_train, params.lgb_path)

print('Prediction Stage.\n')
prediction_stage(df_test, params.lgb_path, False)

print('\nDone.')
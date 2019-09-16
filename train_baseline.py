# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:20:29 2019

@author: aleks

Basic script to get us started
"""

import params
import pandas as pd
from sklearn.model_selection import train_test_split
import utils
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
from feature_engineering import signal_energy, signal_svd_entropy, signal_fourier
from lstm_model import DataLoader, Model

print("Start")
#train_df = utils.read_file(params.train_data_path)

#train_df = train_df.iloc[:1500]
#print("Data length is: {}".format(train_df.size))
#X_train, y_train, df_len = utils.load_and_split_train_file(params.train_data_path)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


# Use sampling from some kernel just for similicity
print("Dividing the dataset")
#num_samples150k = int((len(train_df) / params.sample_length)) 

cols = ['mean','median','std','max',
        'min','var','ptp','10p',
        '25p','50p','75p','90p']

## ENERGY BASED FEATS
#train_features = load("data/train_features_decreasing.csv")
#train_features = train_features.filter(['9_S','9_E'], axis=1)
#headers = list(train_features.columns.values)

#cols.extend(headers)

# These are empty at this stage
#print("Dividing the dataset")
#num_samples150 = int((len(train_df) / 150)) 
#num_samples1500 = int((len(train_df) / 1500))
#num_samples500 = int((len(train_df) / 500))
"""
X_train150, y_train150 = utils.prepare_empty_dfs(train_df, num150, cols)
X_train500, y_train500 = utils.prepare_empty_dfs(train_df, num500, cols)
X_train1500, y_train1500 = utils.prepare_empty_dfs(train_df, num1500, cols)
"""

## so let's try to use 150 not K as num of samples. Will have shittine of rows

def fill_and_save_dataset(train_df, num_samples):
    
    num = int((len(train_df) / num_samples))
    print("Num is...{}".format(num))
    
    print("Preparing empty dataframe")
    X_train, y_train = utils.prepare_empty_dfs(train_df, num_samples, cols)
    
    print("Filling dataset for {} samples".format(num))
    for i in range(num_samples):
    
        #i*sample_length = the starting index (from train_df) of the sample we create
        #i*sample_length + sample_length = the ending index (from train_df)
        sample = train_df.iloc[i*params.sample_length:i*params.sample_length+params.sample_length]
        
        #Converts to numpy array
        x = sample['acoustic_data'].values
        
        #Grabs the final 'time_to_failure' value
        y = sample['time_to_failure'].values[-1]
        y_train.loc[i, 'time_to_failure'] = y
          
       #For every num_samples rows, we make these calculations
       # 150 now
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
        
        se = signal_energy(x)
        svd = signal_svd_entropy(x)
        
        X_train.loc[i, 'se'] = se
        X_train.loc[i, 'svd'] = svd
        
        
        sf = signal_fourier(x)
        
        for n in range(len(sf)-1):
            X_train.loc[i, 'sf_{}'.format(n)] = sf[n]
            
    print("Saving dataset for {} samples".format(num))       
    dump(X_train, 'X_train_{}.joblib'.format(num))
    dump(y_train, 'y_train_{}.joblib'.format(num))
    print("Saving dataset for {} samples is done".format(num))
    
    del X_train, y_train
            
        
### FILLING THE DATASET
#fill_and_save_dataset(train_df, num_samples1500)
#fill_and_save_dataset(train_df, num_samples150)
#fill_and_save_dataset(train_df, num_samples500)
#fill_and_save_dataset(train_df, num_samples150k)
###


"""
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
      
   #For every num_samples rows, we make these calculations
   # 150 now
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
    
    se = signal_energy(x)
    svd = signal_svd_entropy(x)
    
    X_train.loc[i, 'se'] = se
    X_train.loc[i, 'svd'] = svd
    
    
    sf = signal_fourier(x)
    
    for n in range(len(sf)-1):
        #print(n)
        #print(sf[n])
        X_train.loc[i, 'sf_{}'.format(n)] = sf[n]
"""      
    
    
    
    # Adding energy based ones
    #for feat in headers:
    #    X_train.loc[i, feat] = train_features.loc[i, feat]
        
    
### TRAIN RF
        
    



### TRAIN RANDOM FOREST REGRESSOR
"""
X_train = load("X_train_1500.joblib")
y_train = load("y_train_1500.joblib")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

#This creates the Randomforest with the given parameters
rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
                          max_features=0.5, #Max number of features each tree can use 
                          min_samples_leaf=30, #Min amount of samples in each leaf
                          random_state=42)

print("Fitting the RF model")
#This trains the random forest on our training data
rf.fit(X_train, y_train)
dump(rf, 'classifiers/rf_model.joblib') 

print("Fitting and saving done")


## Prediction
result = mean_absolute_error(y_val, rf.predict(X_val))
print("MAE for RF is {}".format(result))
"""

#### TRAIN LSTM

from attr import dataclass
from sklearn import preprocessing
import json

configs = json.load(open('config.json', 'r'))

@dataclass
class DataSet:
    X: np.ndarray
    y: np.ndarray
    

def scale(X_train, X_test):
    
    # normalize the dataset
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    dump(scaler, "scaler_lstm.joblib")
    
    return X_train, X_test



def merge_data(X_train, y_train):
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, shuffle=False)

    if configs["data"]["scale"]:
        X_train, X_val = scale(X_train, X_val)
                   
    train = DataSet(X=X_train, y=y_train)
    validation = DataSet(X=X_val, y=y_val)        
    
    return train, validation


def predict_lstm(X_test, y_test, predictions):
    
    predictions = predictions.flatten().clip(0,1)
    
    mae = round(mean_absolute_error(y_test, predictions), 6)
    print('\n{} VAL MAE: {}'.format("LSTM", mae))

#X_train = load("X_train_1500.joblib")
#y_train = load("y_train_1500.joblib")
"""
X_train, y_train = utils.read_prepared_csv("data/train_features_1500_X.csv", "data/train_features_1500_Y.csv")



train, test = merge_data(X_train, y_train)  
del X_train, y_train 

input_dim = train.X.shape[1] + 1 # has to be + 1 in order for the lstm to work in this case

data = DataLoader(train, test)
del train, test


## Option 1: Train model

model = Model(configs)
print("Building LSTM model...")
model.build_model(input_dim)
 
print("Getting train data...")
x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

print("Training LSTM model...")
# in-memory training
model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size']
	)

print("Training LSTM is done...")

del x, y

print("Getting test data...")
x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )


predictions = model.predict_point_by_point(x_test)


predict_lstm(x_test, y_test, predictions)


del data, x_test, y_test
#### FIRST TRY LGBM
"""


import lightgbm as lgb

 
def get_lgbm():
    
     model = lgb.LGBMRegressor(objective = "regression", 
        boosting = "gbdt",
        metric="mean_absolute_error",
        boost_from_average=False,
        tree_learner="serial",
        num_threads=8,
        learning_rate =0.01,
        num_leaves =16,
        max_depth=-1,
        feature_fraction =0.05,
        bagging_freq =5,
        bagging_fraction =0.4,
        min_data_in_leaf =100,
        min_sum_hessian_in_leaf =11.0,
        verbosity =1,
        num_iterations=99999999,
        seed=44000,
        random_state=42)
     
     return model

    
def fit_and_predict_lgb(X_fit, y_fit, X_val, y_val):
    
    print("Loading LGBM model...")
    model = get_lgbm()
        
    print("Fitting LGBM model...")
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=3500, 
              early_stopping_rounds=3500)
    print("Fitting is done...")
    
    lgb_path = "classifiers"
    name = "LGBM_MODEL_1500"
    
    #Save LightGBM Model
    print("Saving the LGBM model...")
    save_to = '{}/{}.txt'.format(lgb_path, name)
    model.booster_.save_model(save_to)
    print("LGBM model is saved...")
    
    # Predict  
    print("Prediction on X_val...")
    predicted = model.predict(X_val)
    
    print("Getting MAE score on y_val...")
    mae_lgb = round(mean_absolute_error(y_val, predicted), 6)
    print('\nLightGBM VAL MAE: {}'.format(mae_lgb))
      
    return predicted

print("Reading the data...")
X_train, y_train = utils.read_prepared_csv("data/train_features_1500_X.csv", "data/train_features_1500_Y.csv")

print("Train test split...")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)

print("Fit predict..")
fit_and_predict_lgb(X_train, y_train, X_val, y_val)




"""

FOR recurrent, use 1000 points window
skip first frist 1000

So for each point starting 1001st, we genetate statistical info from the previous 1000 points
And then we move the window by 1

"""

print("Removing train df")
#del train_df
"""

150k chuns is 100 1500 chunks of data
Stack all the features of a 100 of this chunk

target value is the last time of the last of this 100 lines


dump(X_train, 'X_train.joblib')
dump(y_train, 'y_train.joblib')

print("Running test train split")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

#This creates the Randomforest with the given parameters
rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
                          max_features=0.5, #Max number of features each tree can use 
                          min_samples_leaf=30, #Min amount of samples in each leaf
                          random_state=42)

print("Fitting the model")
#This trains the random forest on our training data
rf.fit(X_train, y_train)
dump(rf, 'rf_model.joblib') 

print("Fitting and saving done")


## Prediction
result = mean_absolute_error(y_val, rf.predict(X_val))
print("MAE is {}".format(result))
"""

"""
submission = pd.read_csv(params.submission_file, index_col = 'seg_id')

X_test = pd.DataFrame(columns=X_train.columns, ndtype=np.float64, index=submission.index)

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
    
    
#Predict on the test data
test_predictions = rf.predict(X_test)

#Assign the target column in our submission to be our predictions
submission['time_to_failure'] = test_predictions

submission.to_csv(params.submission_out_file)

"""
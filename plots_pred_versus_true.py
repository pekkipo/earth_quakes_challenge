# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:20:24 2019

@author: aleks
"""

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


#X = load("X_train_1500.joblib")
#y = load("y_train_1500.joblib")
X, y = utils.read_prepared_csv("data/train_features_1500_X.csv", "data/train_features_1500_Y.csv")
X = X.loc[0:50000,:]
y = y.loc[0:50000,:]

print("Len X: {}".format(len(X)))
print("Len y: {}".format(len(y)))
#### Load models
models_path = "classifiers/"


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    
def plot_preds(predicted_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

# LGBM
"""
lgb_model = "LGBM_MODEL_1500.txt"
lgbm = lgb.Booster(model_file='{}{}'.format(models_path, lgb_model))
print("LGBM predictig...")
predictions_lgbm = lgbm.predict(X)
print("Len y predictions lgbm: {}".format(len(predictions_lgbm)))

"""

## RF
"""
rf_model = "rf_1500.joblib"
rf = load(models_path + rf_model)
print("RF predictig...")
predictions_rf = rf.predict(X)
print("Len y predictions rf: {}".format(len(predictions_rf)))

plot_results(predictions_rf, y)
"""
## LSTM

@dataclass
class DataSet:
    X: np.ndarray
    y: np.ndarray
    

def scale(X):
    
    # normalize the dataset
    #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = load("scaler_lstm.joblib")
    X = scaler.transform(X)
  
    return X

def predict_lstm(X_test, y_test, predictions):
    
    predictions = predictions.flatten().clip(0,1)
    
    mae = round(mean_absolute_error(y_test, predictions), 6)
    print('\n{} VAL MAE: {}'.format("LSTM", mae))
    
    return predictions

configs = json.load(open('config.json', 'r'))
lstm_path = "good_lstm.h5"
model_lstm = Model(configs)
model_lstm.load_h5_model('{}{}'.format(models_path, lstm_path))

# A bit ridicolous but whatever
X = scale(X)
train = DataSet(X=X, y=y)
data = DataLoader(None, train)

Y=y
#del X, y, train

X, y = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

predictions_lstm = model_lstm.predict_point_by_point(X)

#plot_preds(predictions_lstm)
plot_results(4000*(predictions_lstm-0.4).clip(0,20), Y)
#plot_results(predictions_lstm, Y)


#predictions_lstm = np.load("preds_lstm.npy")
#plot_results(predictions_lstm, y)

#plot_preds(predictions_lstm)

#### Plot



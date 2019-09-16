# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:44:31 2019

@author: aleks
"""

import params
import utils

"""
file_train = params.train_data_path
file_cut_train = params.data_path + "train_cut.csv"
rows = 750000
utils.reduce_csv_file(file_train, file_cut_train, rows)
"""


import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

file = pd.read_csv("train_features_decreasing_interval.csv") #, header = [0,1])
#headers = list(file.columns.values)

def rename(col):
    return "{}_{}".format(col[0], col[1])
    
#file.columns = file.columns.to_series().apply(rename)

file = file.loc[:, file.columns != '10_time']
#pedro_feats = ["{}_{}".format(col_name[0], col_name[1]) for col_name in headers]
dump(file, "data/train_features_decreasing.csv")
print("Done")
"""
new_header = df.iloc[0] 
file = file[1:] 
file.columns = pedro_feats #set the header row as the df header
dump(headers, 'pedros_headers.joblib') 
dump(file, "data/train_features_start_new_header.csv")
"""




"""
plt.figure()
file.time_to_failure.plot()
plt.figure()
file.acoustic_data.plot()
"""



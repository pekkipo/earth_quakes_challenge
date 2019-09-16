#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:00:00 2019

In the training data the events have simply been appended
This file splits them

@author: pedro
"""

import pandas as pd
import json

from utils import j, train_path
from feature_engineering import create_static_features_dataframe, create_rolling_features_dataframe

import params

chunksize = 150000 #1500
N = 10
sample_num = 629145000
total_chunks = sample_num/chunksize

if not chunksize % N == 0:
    raise ValueError("Chunksize must be divisible by N!")

delta = int(chunksize/N)

# read in chunks
#df_chunks = pd.read_csv(j(train_path(), 'train.csv'), chunksize=chunksize)

# My code. Alex
df_chunks = pd.read_csv(params.train_data_path, chunksize=chunksize)

split_train = False
generate_features = False
generate_rolling = True

event_n = 0
chunk_list = []
counter = 0
global_stuff = pd.DataFrame()
for chunk in df_chunks:
    counter += 1
    # This part of the code splits the training set into events
    if split_train:
        # if there is a jump in time, save up to there
        t = chunk.time_to_failure
        dt = t.diff()[~pd.isna(t.diff())]
        if any(dt>0):
            # positive jump!
            ind = dt.index[dt>0]
            chunk_list.append(chunk.loc[chunk.index[0]:(ind.values[0]-1),:].copy())
            event = pd.concat(chunk_list)
            event.to_csv('data/train/train_event{:02d}.csv'.format(event_n),
                         index=False)
            event_n += 1
            chunk_list = []
        else:
            chunk_list.append(chunk)
        # Final chunk, hoping it is not *exactly* the chunksize 
        if len(chunk.index) < chunksize:
            event_n += 1
            event.to_csv('data/train/train_event{:02d}.csv'.format(event_n),
                         index=False)
    # This part of the code generates and writes the features
    if generate_features:
        # DIRTY skip last chunk
        if len(chunk.index)<chunksize:
            continue
        chunk = chunk.reset_index(drop=True)
        feat_row = create_static_features_dataframe(chunk.acoustic_data.values)
        feat_row['10_time'] = chunk.loc[max(chunk.index), 'time_to_failure']
        chunk_list.append(feat_row)
    if generate_rolling:
        if counter % 1000 == 0:
            print(str(counter/total_chunks*100)+"%")
        # DIRTY skip last chunk
        if len(chunk.index)<chunksize:
            continue
        chunk = chunk.reset_index(drop=True)
        feat_row = create_rolling_features_dataframe(chunk.acoustic_data.values)
        feat_row['time'] = chunk.loc[max(chunk.index), 'time_to_failure']
        chunk_list.append(feat_row)
        if counter % 10000 == 0:
            train_data = pd.concat(chunk_list)
            train_data = train_data.reset_index(drop=True)
            train_data.time.to_csv('data/train_features_{}_Y.csv'.format(chunksize),
                              mode='a',
                              header='false',
                              index=False)
            train_data = train_data.drop(columns=['time'])
            train_data.to_csv('data/train_features_{}_X.csv'.format(chunksize),
                              mode='a',
                              header='false',
                              index=False)
            chunk_list = []
            
            global_stuff = train_data

if generate_rolling:
    with open('data/train_features_{}_headers.json'.format(chunksize), 'w') as f:
        json.dump(list(global_stuff.columns), f)
        

if generate_features:
    train_data = pd.concat(chunk_list, axis=	1).T
    train_data = train_data.reset_index(drop=True)
    train_data.to_csv('data/train/train_features_decreasing_interval_2.csv',
                         index=False)

    
        
        
        
        
        
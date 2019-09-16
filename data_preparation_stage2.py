# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from utils import j, train_generated
from feature_engineering import create_rolling_features, pred_mean_slope

chunksize = 100
sample_num = 629145000 / 1500
total_chunks = sample_num/chunksize

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

# read in chunks
# TWO COLUMNS, PREDICTIONS AND TARGET
df_chunks = pd.read_csv(j(train_generated(), 'predictions_target.csv'),
                        chunksize=chunksize)

chunk_list = []
for chunk in df_chunks:
    if len(chunk.index)<chunksize:
            continue
    chunk = chunk.reset_index(drop=True)
    feat_row = pd.DataFrame(create_rolling_features(chunk.prediction.values, 
                                                    functions), index=[0])
    feat_row['target'] = chunk.loc[max(chunk.index), chunk.target]
    chunk_list.append(feat_row)
    
train_data = pd.concat(chunk_list, axis=	1).T
train_data = train_data.reset_index(drop=True)
train_data.to_csv(j(train_generated(), 'train_features_1500_stage2.csv'),
                     index=False) # THIS SHOULD HAVE 4000 rows


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:30:20 2019

@author: pedro
"""

import pandas as pd

from utils import test_path, j
from feature_engineering import create_rolling_features_dataframe

import glob
import os

files = glob.glob(test_path()+"/seg*.csv")


chunksize = 1500


for file in files:
    df_chunks = pd.read_csv(j(test_path(), file),
                            chunksize=chunksize)
    chunk_list = []
    for chunk in df_chunks:
        chunk = chunk.reset_index(drop=True)
        feat_row = create_rolling_features_dataframe(chunk.acoustic_data.values)
        chunk_list.append(feat_row)
    train_data = pd.concat(chunk_list)
    train_data = train_data.reset_index(drop=True)
    train_data.to_csv(j(test_path(), 'feats_1500_'+os.path.basename(file)))
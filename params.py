# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:47:27 2019

@author: aleks

Here I offer to store all the parameters that should be tweaked only here
For example local paths for the data
"""


# Parent folder with all the data
data_path = "D:/Datasets/EarthQuake/"

# Use full train file or the shortened version
# Use "full" or "cut"
#active_file = "full" # "cut"
train_data_path = data_path + "train.csv"
test_folder = data_path + "test/"
submission_file = data_path + "sample_submission.csv"

submission_out_file = "submission_attempt_1.csv" # just a file name for submission file to upload

# For folding
num_folds = 5

#Define the length of each sample for training data
sample_length = 150000 #150000


# Some paths for models
lgb_path = 'lgbm_models/'

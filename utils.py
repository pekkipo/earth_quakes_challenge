# -*- coding: utf-8 -*-
"""
Some useful utils
"""


import pandas as pd
import params # why import a file that executes stuff in it?
import numpy as np
import os

def j(a, *b):
    return os.path.join(a, *b)

def project_root():
    try:
        thispath = os.path.abspath(__file__)
    except NameError:
        thispath = os.getcwd()
    if 'kaggle_earthquake' in thispath:
        thispath = thispath.split('kaggle_earthquake')[0]
    else:
        raise OSError('Check the name of your directory man')
    return j(thispath, 'kaggle_earthquake')

def data_path():
    return j(project_root(), 'data')

def test_path():
    return j(data_path(), 'test')

def train_path():
    return j(data_path(), 'train')

def train_resampled_path():
    return j(data_path(), 'train_resampled')

def generated_path():
    return j(data_path(), 'generated')

def read_prepared_csv(path_X, path_Y):
    'data/train/train_features_1500_Y.csv'
    'data/train/train_features_1500_X.csv'
    Y=pd.read_csv(path_Y, header=0)
    bad_index = Y.index[Y.time=='time']
    Y = Y.drop(index=bad_index)
    Y = Y.reset_index(drop=True)
   # Y = Y.values.squeeze()
    #Y = pd.DataFrame(Y)#pd.to_numeric(Y)
    Y = Y.astype(float)
    
    X=pd.read_csv(path_X, header=0)
    X = X.drop(index=bad_index)
    X = X.reset_index(drop=True)
    X = X.astype(float)
    return X, Y
    

def reduce_csv_file(file_in, file_out, num_rows):
    """
    Creates reduced version of file so that we can quickly test some algorithms or ideas
    
    """
    
    print("Cutting the file...")
    df = pd.read_csv(file_in)
    df.sample(num_rows).to_csv(file_out, index=False)
    print("The file was successfully cut and saved")
    

def get_list_of_test_files():
    """
    Loop through all test files and save them to the list
    
    Returns: 
    List[String]: List of files
    """

    return os.listdir(params.test_folder)

def get_test_file(file_id):
    """
    Gives one particular test file based on its id (without csv extension)
    
    Parameters:
    file_id (String): Id of the file    
    Returns: 
    pandas dataframe
    """
    file_name = file_id + ".csv"
    
    return pd.read_csv(file_name)

def read_all_test_files():
    """
    Get all test files as Pandas DataFrames
    
    Returns: 
    List[DataFrame]: List of dataframes
    """
    files = get_list_of_test_files()
    list_of_dataframes = [pd.read_csv(file) for file in files]
    return list_of_dataframes
    

def switch_active_file(mode):
    """
        Switches train file from short to full one (10GB)
    """
    if mode == "cut":
        return "train_cut.csv"
    else:
        return "train.csv"
    
def split_train(df, as_np=False):
    """
    Splits into train and target. Can be of use for folds
    
    Parameters: 
    df input dataframe
    as_np if True if need to get np.array insted of DataFrame
    
    Returns:
    pd.DataFrame train data X
    pd.DataFrame train data targets y
    pd.DataFrame list of ids
    """
    
    y_df = df['time_to_failure']                        
    df_ids = df.index                   
    df.drop(['time_to_failure'], axis=1, inplace=True)
    
    return df, y_df, df_ids

def read_file(file):
    #Read in the training data
    train_df = pd.read_csv(file,
                    dtype={'acoustic_data': np.int16,
                           'time_to_failure': np.float64}) 
    return train_df


def prepare_empty_dfs(df_train, num_samples, cols):
    
    X_train = pd.DataFrame(index=range(num_samples), #The index will be each of our new samples
                       dtype=np.float64, #Assign a datatype
                       columns=cols) #The columns will be the features we listed above
    
    y_train = pd.DataFrame(index=range(num_samples),
                       dtype=np.float64, 
                       columns=['time_to_failure']) #Our target variable
    
    return X_train, y_train

def load_and_split_train_file(train_df):
    
    X_train = train_df.acoustic_data
    y_train = train_df.time_to_failure
    
    return X_train, y_train
        

def remove_skips(t : pd.Series) -> pd.Series:
    """
    Removes irregular jumps from time_to_failure
    """
    pass

### Run code
"""
file_train = params.train_data_path
file_cut_train = params.data_path + "train_cut.csv"
rows = 10000
reduce_csv_file(file_train, file_cut_train, rows)
"""



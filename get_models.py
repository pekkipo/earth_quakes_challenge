# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:26:38 2019

@author: aleks
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:10:13 2019

@author: aleks
"""

import lightgbm as lgb
import xgboost as xgb

 
def get_lgbm():
    
     model = lgb.LGBMClassifier(objective = "regression", 
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
        num_iterations =99999999,
        seed=44000,
        random_state=42)
     
     return model

def get_xgboost():
    
    model = xgb.XGBClassifier(
            objective="regression",
            tree_method= 'hist',
             eval_metric='auc',
             learning_rate = 0.0936165921314771,
             max_depth = 2,
             colsample_bytree= 0.3561271102144279,
             subsample= 0.8246604621518232,
             min_child_weight= 53,
             gamma= 9.943467991283027,
             silent= 1,
            random_state=42)
    
    return model
    
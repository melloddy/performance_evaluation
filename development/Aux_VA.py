#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:02:29 2020

@author: kfrl011
"""

import numpy as np
import pandas as pd
import os
# os.chdir("/home/kfrl011/GitRepo/performance_evaluation/development/")

from VennABERS import ScoresToMultiProbs
from scipy.sparse import csc_matrix
import argparse

import sklearn.metrics

parser = argparse.ArgumentParser(description="Get Venn-ABERS report")
parser.add_argument("--pred_train", help="distances_single.csv path", type=str, required=True)
parser.add_argument("--pred_test", help="distances_multi.csv path", type=str, required=True)
parser.add_argument("--labels_train", help="distances_multi.csv path", type=str, required=True)
parser.add_argument("--labels_test", help="distances_multi.csv path", type=str, required=True)
parser.add_argument("--folding_train", help="distances_multi.csv path", type=str, required=True)
parser.add_argument("--folding_test", help="distances_multi.csv path", type=str, required=True)
args = parser.parse_args()


""" ONLY FOR TESTING
"""
# pred_file = "/projects/mai/Marc/experiments_full/recent_run_param_k10_small_small/MELLODDY_splits/split_00/y_hat.npy"
# labels_file = "/projects/mai/Marc/experiments_full/recent_run_param_k10_small_small/MELLODDY_splits/split_00/data_prep/split_00_CH/files_4_ml/T10_y.npy"
# folding_file = "/projects/mai/Marc/experiments_full/recent_run_param_k10_small_small/MELLODDY_splits/split_00/data_prep/split_00_CH/files_4_ml/T11_fold_vector.npy"


# pred_all = pd.DataFrame(np.load(pred_file))
# labels  = pd.DataFrame(np.load(labels_file, allow_pickle=True).tolist().todense())
# folding = pd.DataFrame(np.load(folding_file))


# # mask out unrequired predictions from yhat
# pred = pred_all.mask(labels == 0, 0)  

# def train_test_half(data):
#     #exemplarily split into trainng and test 
#     train = data[:round(len(data)/2)].reset_index(drop=True)
#     test = data[round(len(data)/2):].reset_index(drop=True)
#     return train, test


# pred_train, pred_test = train_test_half(pred)
# labels_train, labels_test = train_test_half(labels)
# folding_train, folding_test = train_test_half(folding)

"""
"""

pred_train    = pd.DataFrame(np.load(args.pred_train))
pred_test     = pd.DataFrame(np.load(args.pred_test))
labels_train  = pd.DataFrame(np.load(args.labels_train   
                                     , allow_pickle=True).tolist().todense()))
labels_test   = pd.DataFrame(np.load(args.labels_test    
                                     , allow_pickle=True).tolist().todense()))
folding_train = pd.DataFrame(np.load(args.folding_train))
folding_test  = pd.DataFrame(np.load(args.folding_test))

pred_train = pred_train.mask(labels_train == 0, 0) 
pred_test  = pred_test.mask(labels_test == 0, 0) 

# folding
va_fold = 0

pred_train_prop = pred_train.iloc[list(folding_train[(folding_train[0] != va_fold)].index),:]
pred_test_prop  = pred_test.iloc[list(folding_test[(folding_test[0] != va_fold)].index),:]

pred_train_val  = pred_train.iloc[list(folding_train[(folding_train[0] == va_fold)].index),:]
pred_test_val   = pred_test.iloc[list(folding_test[(folding_test[0] == va_fold)].index),:]

labels_train_prop = labels_train.iloc[list(folding_train[(folding_train[0] != va_fold)].index),:]
labels_test_prop  = labels_test.iloc[list(folding_test[(folding_test[0] != va_fold)].index),:]

labels_train_val = labels_train.iloc[list(folding_train[(folding_train[0] == va_fold)].index),:]
labels_test_val = labels_test.iloc[list(folding_test[(folding_test[0] == va_fold)].index),:]


    
    
def get_VA(train_preds, train_labels, test_preds, test_labels):
    """cal points
    """
    def get_indices_data(df):
        df = csc_matrix(df)
        indices = df.nonzero()
        data = df.data
        tuples = np.array(list(zip(indices[0], indices[1])))
        return tuples, data
    

    # # train quorum columns filter
    # num_pos = (train_labels == +1).sum(0)
    # num_neg = (train_labels == -1).sum(0)
    # cols55  = pd.Series(np.array((num_pos >= 5) & (num_neg >= 5)).flatten())
    
    # train_preds = train_preds[cols55[cols55==True].index.tolist()]
    # train_labels = train_labels[cols55[cols55==True].index.tolist()]
    
    # test quorum columns filter
    num_pos = (test_labels == +1).sum(0)
    num_neg = (test_labels == -1).sum(0)
    cols55  = pd.Series(np.array((num_pos >= 5) & (num_neg >= 5)).flatten())
    
    test_preds  = test_preds[cols55[cols55==True].index.tolist()]
    test_labels = test_labels[cols55[cols55==True].index.tolist()]
    train_preds = train_preds[cols55[cols55==True].index.tolist()]
    train_labels= train_labels[cols55[cols55==True].index.tolist()]
    
    
    # # get inner tasks only
    # idx_train = pd.DataFrame(train_preds.columns)
    # idx_test = pd.DataFrame(test_preds.columns)

    # inner_tasks = idx_train.merge(idx_test)[0].tolist()
    
    # train_preds   = train_preds[inner_tasks]
    # train_labels  = train_labels[inner_tasks]
    # test_preds    = test_preds[inner_tasks]
    # test_labels   = test_labels[inner_tasks]
    
    
    # preds
    train_tuples, train_scores =  get_indices_data(train_preds)
   

    # labels
    _, train_labels = get_indices_data(train_labels)
    train_labels = np.array([1 if i==1 else 0 for i in train_labels])
    
    _, test_labels = get_indices_data(test_labels)
    test_labels = np.array([1 if i==1 else 0 for i in test_labels])
    
    
    # cal points
    calData = np.vstack([train_tuples[:,0],train_tuples[:,1], train_scores, train_labels]).T
    
    # test scores
    test_tuples, test_scores = get_indices_data(test_preds)
    
    
   
    # get current task's cal points and test tuples 
    # calibrate on these only 
    results = pd.DataFrame()
    for task in train_preds.columns.tolist():
        calPts = calData[calData[:,1] == task][:,2:]
        calPts = [(i[0], i[1]) for i in calPts]
        
        p0, p1 = ScoresToMultiProbs(calPts, test_scores[test_tuples[:,1] == task]) 
        
        results_tmp = pd.DataFrame(np.vstack([test_tuples[test_tuples[:,1] == task][:,0], 
                                              test_tuples[test_tuples[:,1] == task][:,1], 
                                              test_scores[test_tuples[:,1] == task], 
                                              test_labels[test_tuples[:,1] == task], 
                                              p0, p1, p1-p0]).T)
        results = pd.concat([results, results_tmp], axis = 0)
    
    cols = ["compound", "task", "pred", "true", "p0", "p1", "p1-p0"]
    results.columns = cols    
    
    return  results

"""
# all folds
"""
results = get_VA(pred_train, labels_train,  pred_test, labels_test)
results.to_csv("results_all_folds.csv")

"""
# cal on all exc and pred on only validation fold 
"""
val_results = get_VA(pred_train_prop, labels_train_prop,  pred_test_val, labels_test_val)
val_results.to_csv("results_val_fold.csv")



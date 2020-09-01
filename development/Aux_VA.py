#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:02:29 2020

@author: kfrl011
"""

import numpy as np
import pandas as pd
import os
import argparse

from VennABERS import ScoresToMultiProbs

from scipy.sparse import csc_matrix
import sklearn.metrics


parser = argparse.ArgumentParser(description="Get Venn-ABERS report for a given LSH fold")
parser.add_argument("--pred", help="distances_single.csv path", type=str, required=True)
parser.add_argument("--true", help="distances_multi.csv path", type=str, required=True)
parser.add_argument("--folding", help="distances_multi.csv path", type=str, required=True)
parser.add_argument("--v_fold", help="distances_multi.csv path", type=int, required=True)
args = parser.parse_args()

true    = pd.DataFrame(np.load(args.true, allow_pickle=True).tolist().todense())
pred    = pd.DataFrame(np.load(args.pred, allow_pickle=True))
pred    = pred.mask(true == 0, 0) 
folding = pd.DataFrame(np.load(args.folding))
                        
va_fold = args.v_fold


# folding
pred_prop = pred.iloc[list(folding[(folding[0] != va_fold)].index),:]
true_prop = true.iloc[list(folding[(folding[0] != va_fold)].index),:]

pred_val  = pred.iloc[list(folding[(folding[0] == va_fold)].index),:]
true_val  = true.iloc[list(folding[(folding[0] == va_fold)].index),:]

    
def get_VA(train_preds, train_labels, test_preds, test_labels):
    """cal points
    """
    def get_indices_data(df):
        df = csc_matrix(df)
        indices = df.nonzero()
        data = df.data
        tuples = np.array(list(zip(indices[0], indices[1])))
        return tuples, data
  
    
    # test quorum columns filter
    num_pos = (test_labels == +1).sum(0)
    num_neg = (test_labels == -1).sum(0)
    cols55  = pd.Series(np.array((num_pos >= 5) & (num_neg >= 5)).flatten())
    
    test_preds  = test_preds[cols55[cols55==True].index.tolist()]
    test_labels = test_labels[cols55[cols55==True].index.tolist()]
    train_preds = train_preds[cols55[cols55==True].index.tolist()]
    train_labels= train_labels[cols55[cols55==True].index.tolist()]

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
    
    return results

results = get_VA(pred_prop, true_prop,  pred_val, true_val)
results.to_csv("results_all_folds.csv")

import numpy as np
import pandas as pd
import os
import argparse

from VennABERS import ScoresToMultiProbs

from scipy.sparse import csc_matrix
import sklearn.metrics


parser = argparse.ArgumentParser(description="Get Venn-ABERS report for a given LSH fold")
parser.add_argument("--ypred", help="complete Yhat sparse matrix .npy (all folds)", type=str, required=True)
parser.add_argument("--ytrue", help="complete Y labels matrix .npy (all folds)", type=str, required=True)
parser.add_argument("--folding", help="folding vector .npy", type=str, required=True)
parser.add_argument("--fold_va", help="fold id to use as validation fold. VA will be calibrated on remainder folds and calculated for the validation fold", type=int, required=True)
parser.add_argument("--output", help="output file name", type=str, required=True)
args = parser.parse_args()


true    = np.load(args.ytrue, allow_pickle=True).item()
pred    = np.load(args.ypred, allow_pickle=True).item()
folding = np.load(args.folding)

va_fold = args.fold_va


# extract the predictions/labels for train/test
true_tr = true[folding != va_fold]
pred_tr = pred[folding != va_fold]

true_va = true[folding == va_fold]
pred_va = pred[folding == va_fold]


# find columns with less 5-5 in test
n_pos = (true_va>0).sum(axis=0).tolist()[0]
n_neg = (true_va<0).sum(axis=0).tolist()[0]

pass_5pos = np.array(n_pos)>=5
pass_5neg = np.array(n_neg)>=5

valid_tasks = np.where(np.logical_and(pass_5pos, pass_5neg))[0]

# for each task compute vennABERS p0/p1
dat = []
for task in valid_tasks:
    if task%200==0:print(task)
    if not valid_tasks[task]:continue
    
    # setup the calibration data (training set?)
    train_labels = np.where(true_tr[:,task].data>0, 1, 0) # change negative labels (-1) to zero for the VennABERS function to work 
    train_pred   = pred_tr[:, task].data
    calPts = [(train_pred[i], train_labels[i]) for i in range(len(train_labels))]
    

    # setup the prediction data (validation set)
    predPts  = pred_va[:,task].data
    predLab  = true_va[:,task].data
    descr_id = true_va[:,task].nonzero()[0]

    # compute vennABERS probas
    p0, p1 = ScoresToMultiProbs(calPts, predPts)
    
    # store in list 
    [dat.append([task, descr_id[idx], predPts[idx], predLab[idx], p0[idx], p1[idx]]) for idx, val in enumerate(p0)]


df = pd.DataFrame(dat, columns=['cont_classification_task_id', 'cont_descriptor_vector_id', 'yhat', 'label', 'p0', 'p1'])


df.to_csv(args.output, index=None)

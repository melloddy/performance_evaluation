import os
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
import json
import scipy.sparse as sparse
from scipy.stats import spearmanr
from  pathlib import Path

parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, required=True)
parser.add_argument("--y_pred_onpremise", help="Yhat prediction output from onpremise run (<single pharma dir>/y_hat.npy)", type=str, required=True)
parser.add_argument("--y_pred_substra", help="Pred prediction output from substra platform (./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred)", type=str, required=True)
parser.add_argument("--folding", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, required=True)
parser.add_argument("--substra_performance_report", help="JSON file with global reported performance from substra platform (i.e. ./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, required=True)
parser.add_argument("--task_map", help="Taskmap from MELLODDY_tuner output of single run (i.e. from results/weight_table_T3_mapped.csv)", required=True)
parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
parser.add_argument("--use_venn_abers", help="Toggle to turn on Venn-ABERs code", action='store_true', default=False)
parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])

args = parser.parse_args()

def vprint(s="",derisk_check=False):
   if args.verbose:
      print()
      if derisk_check:
         print('====================================================================================================================================')
         print(s)
         print('====================================================================================================================================')
      else: print(s)

vprint(args)


y_pred_onpremise_path = Path(args.y_pred_onpremise)
y_pred_substra_path = Path(args.y_pred_substra)
assert y_pred_onpremise_path.suffix == '.npy', "On-premise prediction file needs to be '.npy'"
assert y_pred_substra_path.stem == 'pred', "Substra prediction file needs to be 'pred'"
task_map = pd.read_csv(args.task_map)

if args.filename is not None:
   name = args.filename
else:
   name = f"derisk_{os.path.basename(args.y_true_all)}_{args.y_pred_onpremise}_{args.y_pred_substra.split('/')[0]}_{os.path.basename(args.folding)}"
   vprint(f"Run name is '{name}'.")
assert not os.path.exists(name), f"{name} already exists... exiting"
os.makedirs(name)


#load the folding/true data
folding = np.load(args.folding)
vprint(f'Loading y_true: {args.y_true_all}') 
y_true_all = np.load(args.y_true_all, allow_pickle=True).item()
y_true_all = y_true_all.tocsc()

## filtering out validation fold
fold_va = 1
y_true = y_true_all[folding == fold_va]
y_true_all = None #clear all ytrue to save memory

## default weights are set to 1.0 for the derisking (required)
tw_df = np.ones(y_true.shape[1], dtype=np.uint8)

def find_max_f1(precision, recall):
   F1   = np.zeros(len(precision))
   mask = precision > 0
   F1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
   return F1.max()

## Check performance reports for federated (substra) output
def substra_global_perf_from_json(performance_report):
   with open(performance_report, "r") as fi:
      json_data = json.load(fi)
   assert 'all' in json_data.keys(), "expected 'all' in the performance report"
   assert len(json_data.keys()) == 1, "only expect one performance report"
   reported_performance = json_data["all"]
   assert 0.0 <= reported_performance <= 1.0, "reported performance does not range between 0.0-1.0"
   return reported_performance

## write performance reports for global aggregation
def write_global_report(global_performances,onpremise_or_substra):
   global name
   cols = ['aucpr_mean','aucroc_mean','maxf1_mean', 'kappa_mean', 'tn', 'fp', 'fn', 'tp']
   if args.use_venn_abers: cols += ['vennabers_mean']
   perf_df = pd.DataFrame([global_performances],columns=cols)
   fn = name + '/' + onpremise_or_substra + "_global_performances_derisk.csv"
   perf_df.to_csv(fn, index=None)
   vprint(f"Wrote {onpremise_or_substra} global performance report to: {fn}")
   return perf_df

## write performance reports per-task & per-task_assay
def write_aggregated_report(local_performances, onpremise_or_substra):
   global name
   global task_map
   vprint(f"{onpremise_or_substra} performances shape: " + str(local_performances.shape))
   # write per-task report
   df = local_performances[:]
   df['classification_task_id'] = df['classification_task_id'].astype('int32')
   df = df.merge(task_map, right_on=["classification_task_id","assay_type"], left_on=["classification_task_id","assay_type"], how="left")
   fn1 = name + '/' + onpremise_or_substra + "_per-task_performances_derisk.csv"
   df.to_csv(fn1, index=False)
   vprint(f"Wrote {onpremise_or_substra} per-task report to: {fn1}")
   # write per-assay_type, ignore task id
   df2 = local_performances.loc[:,'assay_type':].groupby('assay_type').mean()
   fn2 = name + '/' + onpremise_or_substra + "_per-assay_performances_derisk.csv"
   df2.to_csv(fn2, index=False)
   vprint(f"Wrote {onpremise_or_substra} per-assay report to: {fn2}")
   return

def mask_y_hat(onpremise_path, substra_path):
    global folding
    global fold_va
    global y_true
    
    # load the data
    vprint(f'Loading onpremise (npy) predictions: {onpremise_path}') 
    onpremise_yhat = np.load(onpremise_path, allow_pickle=True).item().tocsr().astype('float32')
    vprint(f'Loading substra (pred) output: {substra_path}') 
    substra_yhat = torch.load(substra_path).astype('float32')
            
    # only keep validation fold
    onpremise_yhat = onpremise_yhat[folding == fold_va]
    try: substra_yhat = substra_yhat[folding == fold_va]
    except IndexError: pass
    
    true_data = y_true.astype(np.uint8).todense()
    assert true_data.shape == onpremise_yhat.shape, f"True shape {true_data.shape} and Pred shape {onpremise_yhat.shape} need to be identical"
    assert true_data.shape == substra_yhat.shape, f"True shape {true_data.shape} and Pred shape {substra_yhat.shape} need to be identical"

    return [onpremise_yhat, substra_yhat]

## check the pre_calculated_performance with the reported performance json
def per_run_performance(y_pred, performance_report, onpremise_or_substra, tasks_table):
   global y_true
   global tw_df
   global args
   if args.use_venn_abers: from VennABERS import get_VA_margin_median_cross

   if onpremise_or_substra == 'substra':
      global_pre_calculated_performance = substra_global_perf_from_json(performance_report)
   else:
      y_pred = sparse.csc_matrix(y_pred)

   ## checks to make sure y_true and y_pred match
   assert y_true.shape == y_pred.shape, f"y_true shape do not match {onpremise_or_substra} y_pred ({y_true.shape} & {y_pred.shape})"
   assert y_true.nnz == y_pred.nnz, f"y_true number of nonzero values do not match {onpremise_or_substra} y_pred"
   assert (y_true.indptr == y_pred.indptr).all(), f"y_true indptr do not match {onpremise_or_substra} y_pred"
   assert (y_true.indices == y_pred.indices).all(), f"y_true indices do not match {onpremise_or_substra} y_pred"

   task_id = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
   assay_type = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
   aucpr   = np.full(y_true.shape[1], np.nan)
   aucroc  = np.full(y_true.shape[1], np.nan)
   maxf1   = np.full(y_true.shape[1], np.nan)
   kappa   = np.full(y_true.shape[1], np.nan)
   tn     = np.full(y_true.shape[1], np.nan)
   fp     = np.full(y_true.shape[1], np.nan)
   fn     = np.full(y_true.shape[1], np.nan)
   tp     = np.full(y_true.shape[1], np.nan)
   if args.use_venn_abers: vennabers = np.full(y_true.shape[1], np.nan)

   num_pos = (y_true == +1).sum(0)
   num_neg = (y_true == -1).sum(0)
   cols55  = np.array((num_pos >= 5) & (num_neg >= 5)).flatten()


   for col in range(y_true.shape[1]):
      y_true_col = y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1
      y_pred_col = y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]]
      y_true_col, y_pred_col = y_true_col.astype(np.uint8), y_pred_col.astype('float32')

      if args.use_venn_abers: pts = np.vstack((y_pred_col, y_true_col)).T # points for Venn-ABERS

      if y_true_col.shape[0] <= 1:
         ## not enough data for current column, skipping
         continue
      if (y_true_col[0] == y_true_col).all():
         continue
      task_id[col] = tasks_table["classification_task_id"][tasks_table["cont_classification_task_id"]==col].iloc[0]
      assay_type[col] = tasks_table["assay_type"][tasks_table["cont_classification_task_id"]==col].iloc[0]
      y_classes   = np.where(y_pred_col > 0.5, 1, 0).astype(np.uint8)
      precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true_col, probas_pred = y_pred_col)
      aucpr[col]  = sklearn.metrics.auc(x = recall, y = precision)
      aucroc[col] = sklearn.metrics.roc_auc_score(y_true  = y_true_col, y_score = y_pred_col)
      maxf1[col]  = find_max_f1(precision, recall)
      kappa[col]  = sklearn.metrics.cohen_kappa_score(y_true_col, y_classes)
      tn[col], fp[col], fn[col], tp[col] = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()
      ##per-task performance:
      cols = ['classification_task_id', 'assay_type', 'aucpr','aucroc','maxf1','kappa','tn','fp','fn','tp']
      if args.use_venn_abers: 
         vennabers[col] = get_VA_margin_median_cross(pts)
         local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],aucpr[cols55],aucroc[cols55],maxf1[cols55],\
            kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55], vennabers[cols55]]).T, columns=cols+['vennabers'])
      else:
         local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],aucpr[cols55],aucroc[cols55],maxf1[cols55],\
            kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55]]).T, columns=cols)


   ##correct the datatypes for numeric columns
   vprint(local_performance)
   for c in local_performance.iloc[:,2:].columns:
      local_performance.loc[:,c] = local_performance.loc[:,c].astype('float32')
   ##write per-task & per-assay_type performance:
   write_aggregated_report(local_performance, onpremise_or_substra)

   ##global aggregation:
   #tw just here for compatibility (not used for derisk)
   tw_weights=tw_df[cols55]
   aucpr_mean  = np.average(aucpr[cols55],weights=tw_weights)
   aucroc_mean = np.average(aucroc[cols55],weights=tw_weights)
   maxf1_mean  = np.average(maxf1[cols55],weights=tw_weights)
   kappa_mean  = np.average(kappa[cols55],weights=tw_weights)
   tn_sum = tn[cols55].sum()
   fp_sum = fp[cols55].sum()
   fn_sum = fn[cols55].sum()
   tp_sum = tp[cols55].sum()

   if onpremise_or_substra == 'substra':
      if not np.allclose([global_pre_calculated_performance],[aucpr_mean], rtol=1e-05, atol=1e-05):
         vprint(f"(Phase 2 de-risk check #2): WARNING! Reported performance in {performance_report} ({global_pre_calculated_performance}) not close to calculated performance for {onpremise_or_substra} ({aucpr_mean}) (tol:1e-05)",derisk_check=True)
      else:
         vprint(f"(Phase 2 de-risk check #2): Check passed! Reported performance in {performance_report} ({global_pre_calculated_performance}) close to the calculated performance for {onpremise_or_substra} ({aucpr_mean}) (tol:1e-05)",derisk_check=True)
   if args.use_venn_abers:
      vennabers_mean  = np.average(vennabers[cols55],weights=tw_weights)
      global_performance = write_global_report([aucpr_mean,aucroc_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum, vennabers_mean], onpremise_or_substra)
   else: global_performance = write_global_report([aucpr_mean,aucroc_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum], onpremise_or_substra)
   return [local_performance,global_performance]

## calculate the difference between the single- and multi-pharma outputs and write to a file
def calculate_deltas(onpremise_results, substra_results):
   global name
   global task_map

   for idx, delta_comparison in enumerate(['locals','/deltas_global_performances_derisk.csv']):
      assert onpremise_results[idx].shape[0] == substra_results[idx].shape[0], "the number of tasks are not equal between the on-premise and substra outputs for {delta_comparison}"
      assert onpremise_results[idx].shape[1] == substra_results[idx].shape[1], "the number of reported metrics are not equal between the on-premise and reported substra outputs for {delta_comparison}"
      # add assay aggregation if local
      if(delta_comparison == 'locals'):
         cti = substra_results[idx]["classification_task_id"]
         at = substra_results[idx]["assay_type"]
         delta = (substra_results[idx].loc[:, "aucpr":]-onpremise_results[idx].loc[:, "aucpr":])
         tdf = pd.concat([cti, at, delta], axis = 1)
         if not np.allclose(tdf['aucpr'],[0]*len(tdf['aucpr']), rtol=1e-05, atol=1e-05):
            vprint(f"(Phase 2 de-risk output check #3): WARNING! Calculated per-task deltas are not all close to zero (tol:1e-05)",derisk_check=True)
         else:
            vprint(f"(Phase 2 de-risk output check #3): Check passed! Calculated per-task deltas close to zero (tol:1e-05)",derisk_check=True)
         fn1 = name + '/deltas_per-task_performances_derisk.csv'
         pertask = tdf[:]
         pertask = pertask.merge(task_map, right_on=["classification_task_id","assay_type"], left_on=["classification_task_id","assay_type"], how="left")
         pertask.to_csv(fn1, index= False)
         vprint(f"Wrote per-task delta report to: {fn1}")

         # aggregate on assay_type level
         fn2 = name + '/deltas_per-assay_performances_derisk.csv'
         tdf.groupby("assay_type").mean().to_csv(fn2)
         vprint(f"Wrote per-assay delta report to: {fn2}")
      else:
         print(onpremise_results[idx])
         allclose = np.allclose(onpremise_results[idx]['aucpr_mean'], substra_results[idx]['aucpr_mean'], rtol=1e-05, atol=1e-05)
         #phase 2 derisk output (by each pharma) check that the aggregated performance on the platform is
         #numerically identical (difference < 1e-5) to the aggregated performance computed from the model on the pharma premises.
         if not allclose:
               vprint(f"(Phase 2 de-risk output check #4): WARNING! Calculated global aggregation performance for on-premise {onpremise_results[idx]['aucpr_mean'].values} and substra {substra_results[idx]['aucpr_mean'].values} are not close (tol:1e-05)",derisk_check=True)
         else:
               vprint(f"(Phase 2 de-risk output check #4): Check passed! Calculated global aggregation performance for on-premise {onpremise_results[idx]['aucpr_mean'].values} and substra {substra_results[idx]['aucpr_mean'].values} are close (tol:1e-05)",derisk_check=True)
         (substra_results[idx]-onpremise_results[idx]).to_csv(name + delta_comparison, index=False)

##function to call allclose check for yhats
def yhat_allclose_check(yhat1,yhat2):
   nnz1 = yhat1.nonzero()
   nnz2 = yhat2.nonzero()
   allclose = np.allclose(yhat1[nnz1], yhat2[nnz2], rtol=1e-05, atol=1e-05)
   vprint(f"Spearmanr rank correlation coefficient of the {args.y_pred_onpremise}' and '{args.y_pred_substra} yhats = {spearmanr(yhat1[nnz1],yhat2[nnz2],axis=1)}")
   if not allclose:
      vprint(f"(Phase 2 de-risk output check #1): ERROR! yhats not close between on-premise (generated from the substra model) .npy and substra 'pred' file (tol:1e-05)",derisk_check=True)
   else:
      vprint(f"(Phase 2 de-risk output check #1): Check passed! yhats close between on-premise (generated from the substra model) .npy and substra 'pred' file (tol:1e-05)",derisk_check=True)

vprint(f"Masking if necessary")
onprmise_yhat, substra_yhat = mask_y_hat(y_pred_onpremise_path,y_pred_substra_path)
folding, fold_va = None, None #clear folding - no longer needed
vprint(f"Checking np.allclose for between for '{args.y_pred_onpremise}' and '{args.y_pred_substra}' yhats")
yhat_allclose_check(onprmise_yhat,substra_yhat)

vprint(f"Calculating '{args.y_pred_onpremise}' performance for '.npy' type (on-premise) input files")
onpremise_results = per_run_performance(onprmise_yhat,None, "on-premise", task_map)
onprmise_yhat = None #clear yhat from memory - no longer needed

vprint(f"Calculating '{args.y_pred_substra}' for 'pred' substra output files")
substra_results = per_run_performance(substra_yhat,args.substra_performance_report, "substra", task_map)
substra_yhat=None #clear yhat from memory - no longer needed
y_true, tw_df = None, None #clear ytrue and weights since both per_run_performance finished - no longer needed

vprint(f"Calculating delta between '{args.y_pred_onpremise}' and '{args.y_pred_substra}' performances.")
calculate_deltas(onpremise_results,substra_results)

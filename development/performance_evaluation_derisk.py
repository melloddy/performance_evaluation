import os
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
import json
import ast
import scipy.sparse as sparse
from  pathlib import Path
from VennABERS import get_VA_margin_median_cross

parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, default="T10_y.npy")
parser.add_argument("--y_pred_onpremise", help="Yhat prediction output from onpremise run (<single pharma dir>/y_hat.npy)", type=str, required=True)
parser.add_argument("--y_pred_substra", help="Pred prediction output from substra platform (./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred)", type=str, required=True)
parser.add_argument("--folding", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, default="folding.npy")
parser.add_argument("--onpremise_performance_report", help="JSON file with global reported single-pharma performance", type=str, default=None, required=False)
parser.add_argument("--substra_performance_report", help="JSON file with global reported performance from substra platform (i.e. ./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=True)
parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
parser.add_argument("--task_map", help="Taskmap from MELLODDY_tuner output of single run (results/weight_table_T3_mapped.csv)", required=True)

args = parser.parse_args()

def vprint(s=""):
   if args.verbose:
      print()
      print(s)    
vprint(args)

assert float(pd.__version__[:4]) >=0.25, "Pandas version must be >=0.25"

y_pred_onpremise_path = Path(args.y_pred_onpremise)
y_pred_substra_path = Path(args.y_pred_substra)
assert y_pred_onpremise_path.suffix == '.npy', "On-premise prediction file needs to be '.npy'"
assert y_pred_substra_path.stem == 'pred', "Substra prediction file needs to be 'pred'"
task_map = pd.read_csv(args.task_map)


if args.filename is not None:
   name = args.filename
else:
   name = f"derisk_{os.path.basename(args.y_true_all)}_{args.y_pred_onpremise.split('/')[0]}_{y_pred_substra_path.split('/')[0]}_{os.path.basename(args.folding)}"
vprint(f"Run name is '{name}'.")
assert not os.path.exists(name), f"{name} already exists... exiting"
os.makedirs(name)


#load the folding/true data
folding = np.load(args.folding)
y_true_all = np.load(args.y_true_all, allow_pickle=True).item()
y_true_all = y_true_all.tocsc()


## filtering out validation fold
fold_va = 1
y_true = y_true_all[folding == fold_va]


## default weights are set to 1.0 for the derisking (required)
tw_df = np.ones(y_true.shape[1], dtype=np.float32)


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

## Check performance reports for on-premise sparsechem outputs
def onpremise_global_perf_from_json(performance_report):
   with open(performance_report, "r") as fi:
      json_data = json.load(fi)
      reported_performance = ast.literal_eval(json_data['results_agg']['va'])['auc_pr']
   assert 0.0 <= reported_performance <= 1.0, "reported performance does not range between 0.0-1.0"
   return reported_performance



## write performance reports for global aggregation
def write_global_report(args_name,global_performances,onpremise_or_substra):
   global name
   perf_df = pd.DataFrame([global_performances],columns=\
      ['aucpr_mean','aucroc_mean','maxf1_mean', 'kappa_mean', 'tn', 'fp', 'fn', 'tp', 'vennabers_mean'])
   fn = name + '/' + onpremise_or_substra + "_" + os.path.basename(args_name) + '_global_performances_derisk.csv'
   perf_df.to_csv(fn)
   vprint(f"Wrote {onpremise_or_substra} global performance report to: {fn}")
   return perf_df

## write performance reports per-task & per-task_assay
def write_aggregated_report(args_name,local_performances, onpremise_or_substra):
   vprint(f"{args_name} performances shape: " + str(local_performances.shape))
   # write per-task report
   fn1 = name + '/' + onpremise_or_substra + "_" + os.path.basename(args_name) + '_per-task_performances_derisk.csv'
   vprint(f"Wrote {onpremise_or_substra} per-task report to: {fn1}")
   local_performances.to_csv(fn1)
   # write per-assay type, ignore task id
   df = local_performances.loc[:,'assay type':].groupby('assay type').mean()
   fn2 = name + '/' + onpremise_or_substra + "_" + os.path.basename(args_name) + '_per-assay_performances_derisk.csv'
   df.to_csv(fn2)
   vprint(f"Wrote {onpremise_or_substra} per-assay report to: {fn2}")
   return


## check the pre_calculated_performance with the reported performance json
# 
def per_run_performance(y_pred_arg, performance_report, onpremise_or_substra, tasks_table):
   global y_true
   global tw_df
   global args
   
   if onpremise_or_substra == 'substra':
      y_pred = torch.load(y_pred_arg)
      global_pre_calculated_performance = substra_global_perf_from_json(performance_report)
   else:
      y_pred = sparse.csc_matrix(y_pred_arg)   
      global_pre_calculated_performance = onpremise_global_perf_from_json(performance_report)
   ## checks to make sure y_true and y_pred match
   assert y_true.shape == y_pred.shape, f"{onpremise_or_substra} y_true shape do not match y_pred ({y_true.shape} & {y_pred.shape})"
   assert y_true.nnz == y_pred.nnz, f"{onpremise_or_substra} y_true number of nonzero values do not match y_pred"
   assert (y_true.indptr == y_pred.indptr).all(), f"{onpremise_or_substra} y_true indptr do not match y_pred"
   assert (y_true.indices == y_pred.indices).all(), f"{onpremise_or_substra} y_true indices do not match y_pred"
   
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
   vennabers = np.full(y_true.shape[1], np.nan)
   
   num_pos = (y_true == +1).sum(0)
   num_neg = (y_true == -1).sum(0)
   cols55  = np.array((num_pos >= 5) & (num_neg >= 5)).flatten()


   for col in range(y_true.shape[1]):
      y_true_col = y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1
      y_pred_col = y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]]
      
      pts = np.vstack((y_pred_col, y_true_col)).T # points for Venn-ABERS
      
      if y_true_col.shape[0] <= 1:
         ## not enough data for current column, skipping
         continue
      if (y_true_col[0] == y_true_col).all():
         continue
      task_id[col] = tasks_table["classification_task_id"][tasks_table["cont_classification_task_id"]==col].iloc[0]
      assay_type[col] = tasks_table["assay_type"][tasks_table["cont_classification_task_id"]==col].iloc[0]
      y_classes   = np.where(y_pred_col > 0.5, 1, 0)
      precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true_col, probas_pred = y_pred_col)
      aucpr[col]  = sklearn.metrics.auc(x = recall, y = precision)
      aucroc[col] = sklearn.metrics.roc_auc_score(y_true  = y_true_col, y_score = y_pred_col)
      maxf1[col]  = find_max_f1(precision, recall)
      kappa[col]  = sklearn.metrics.cohen_kappa_score(y_true_col, y_classes)
      tn[col], fp[col], fn[col], tp[col] = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()
      vennabers[col] = get_VA_margin_median_cross(pts)

   ##local performance:
   local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],aucpr[cols55],aucroc[cols55],maxf1[cols55],\
                  kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55], vennabers[cols55]]).T,\
                  columns=['task id', 'assay type', 'aucpr','aucroc','maxf1','kappa','tn','fp','fn','tp', 'vennabers'])
   
   ##correct the datatypes for numeric columns
   vprint(local_performance)
   for c in local_performance.iloc[:,2:].columns:
      local_performance.loc[:,c] = local_performance.loc[:,c].astype(float)
   write_aggregated_report(y_pred_arg,local_performance, onpremise_or_substra)
                  
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
   vennabers_mean  = np.average(vennabers[cols55],weights=tw_weights)
   
   assert global_pre_calculated_performance == aucpr_mean, f"Reported performance in {performance_report} ({global_pre_calculated_performance}) does not match calculated performance for {y_pred_arg} ({aucpr_mean})"
   vprint(f"Check passed: Reported performance in {performance_report} ({global_pre_calculated_performance}) match the calculated performance for {y_pred_arg} ({aucpr_mean})")
   global_performance = write_global_report(y_pred_arg,[aucpr_mean,aucroc_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum, vennabers_mean], onpremise_or_substra)
   
   return y_pred,[local_performance,global_performance]

## calculate the difference between the single- and multi-pharma outputs and write to a file
def calculate_deltas(onpremise_results, substra_results):
   for idx, delta_comparison in enumerate(['locals','/deltas_global_performances_derisk.csv']):

      assert onpremise_results[idx].shape[0] == substra_results[idx].shape[0], "the number of tasks are not equal between the single- and multi-pharma runs"
      assert onpremise_results[idx].shape[1] == substra_results[idx].shape[1], "the number of reported metrics are not equal between the single- and multi-pharma runs"
      
      # add assay aggregation if local
      if(delta_comparison == 'locals'):
         at = substra_results[idx]["assay type"]
         delta = (substra_results[idx].loc[:, "aucpr":]-onpremise_results[idx].loc[:, "aucpr":])
         tdf = pd.concat([at, delta], axis = 1)
         fn1 = name + '/deltas_per-task_performances_derisk.csv'
         tdf.to_csv(fn1)
         vprint(f"Wrote per-task delta report to: {fn1}")
         if not (tdf==0).all().all():
            vprint(f"ERROR! (Phase 2 de-risk output check): per-task deltas are not zeros (tol:1e-05)")
         
         # aggregate on assay_type level
         fn2 = name + '/deltas_per-assay_performances_derisk.csv'
         tdf.groupby("assay type").mean().to_csv(fn2)
         vprint(f"Wrote per-assay delta report to: {fn2}")
      else:
         allclose = np.allclose(onpremise_results[idx], substra_results[idx], rtol=1e-05, atol=1e-05)
         #phase 2 derisk output (by each pharma) check that the aggregated performance on the platform is
         #numerically identical (difference < 1e-5) to the aggregated performance computed from the model on the pharma premises.
         if not allclose:
               vprint(f"ERROR! (Phase 2 de-risk output check): there is a mistake in the aggregated metrics or in the performance reported by the substra platform (tol:1e-05)")
         (substra_results[idx]-onpremise_results[idx]).to_csv(name + delta_comparison)

##function to call allclose check for yhats
def pred_mode_allclose_check(onpremise_yhat,substra_yhat):
   allclose = np.allclose(onpremise_yhat.data, substra_yhat.data, rtol=1e-05, atol=1e-05)
   if not allclose:
      vprint(f"ERROR! (Phase 2 de-risk output check): there is problem in the substra platform, yhats not close (tol:1e-05)")


vprint(f"Calculating '{args.y_pred_onpremise}' performance for '.npy' type (on-premise) input files")
onpremise_yhat, onpremise_results = per_run_performance(args.y_pred_onpremise,args.onpremise_performance_report, "on-premise", task_map)

vprint(f"Calculating '{args.y_pred_substra}' for 'pred' substra output files")
substra_yhat, substra_results = per_run_performance(args.y_pred_substra,args.substra_performance_report, "substra", task_map)

vprint(f"Checking np.allclose for between for '{args.y_pred_onpremise}' and '{args.y_pred_substra}' yhats")
pred_mode_allclose_check(onpremise_yhat,substra_yhat)

vprint(f"Calculating delta between '{args.y_pred_onpremise}' and '{args.y_pred_substra}' performances.")
calculate_deltas(onpremise_results,substra_results)



import os
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
import json 
import scipy.sparse as sparse
from  pathlib import Path
from VennABERS import get_VA_margin_median_cross

parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, default="T10_y.npy")
parser.add_argument("--y_pred_single", help="Yhat prediction output from single-pharma run (./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred) or (<single pharma dir>/y_hat.npy)", type=str, required=True)
parser.add_argument("--y_pred_multi", help="Yhat prediction output from multi-pharma run (./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred) or (<multi pharma dir>/y_hat.npy)", type=str, required=True)
parser.add_argument("--folding", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, default="folding.npy")
parser.add_argument("--task_weights", help="CSV file with columns task_id and weight (i.e.  files_4_ml/T9_red.csv)", type=str, default=None)
parser.add_argument("--single_performance_report", help="JSON file with global reported single-pharma performance (i.e. ./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=False)
parser.add_argument("--multi_performance_report", help="JSON file with global reported multi-pharma performance (i.e. ./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=True)
parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
parser.add_argument("--task_map_single", help="Taskmap from MELLODDY_tuner output of single run (results/weight_table_T3_mapped.csv)", default = None)
parser.add_argument("--task_map_multi", help="Taskmap from MELLODDY_tuner output of single run (results/weight_table_T3_mapped.csv)", default = None)
parser.add_argument("--derisk", help="Phase 2 derisk check toggle", action='store_true', default=False)

args = parser.parse_args()

def vprint(s=""):
   if args.verbose:
      print()
      print(s)    
vprint(args)

assert float(pd.__version__[:4]) >=0.25, "Pandas version must be >=0.25"

y_pred_single_path = Path(args.y_pred_single)
y_pred_multi_path = Path(args.y_pred_multi)

assert all([pfile in ['pred','npy'] for pfile in [y_pred_single_path.stem, y_pred_multi_path.stem]]), "All prediction files need to be pred or .npy"

# decide if on-premise predictions or federated output based on file input
if y_pred_single_path.stem == "pred":
   pred = True
elif y_pred_single_path.suffix == '.npy':
   pred = False
single_tasks = pd.read_csv(args.task_map_single)
multi_tasks = pd.read_csv(args.task_map_multi)


if args.filename is not None:
   name = args.filename
else:
   name = f"{os.path.basename(args.y_true_all)}_{args.y_pred_single.split('/')[0]}_{args.y_pred_multi.split('/')[0]}_{os.path.basename(args.folding)}"
   if args.task_weights is not None:
      name += f"_{os.path.basename(args.task_weights)}"
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


## Loading task weights (ported from WP2 sparse chem pred.py code)
if args.task_weights is not None:
   tw_df = pd.read_csv(args.task_weights)
   assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
   assert tw_df.shape[1] == 2, "task weight file (CSV) must only have 2 columns"
   assert "weight" in tw_df.columns, "weight is missing in task weights CVS file"
   assert y_true.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
   assert (0 <= tw_df.weight).all(), "task weights must not be negative"
   assert (tw_df.weight <= 1).all(), "task weights must not be larger than 1.0"
   assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
   assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
   assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
   assert tw_df.shape[0]==y_true.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({y_true.shape[1]})."
   tw_df.sort_values("task_id", inplace=True)
else:
   ## default weights are set to 1.0
   tw_df = np.ones(y_true.shape[1], dtype=np.float32)


def find_max_f1(precision, recall):
   F1   = np.zeros(len(precision))
   mask = precision > 0
   F1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
   return F1.max()

## Check performance reports for federated output
def global_perf_from_json(performance_report):
   with open(performance_report, "r") as fi:
      json_data = json.load(fi)
   assert 'all' in json_data.keys(), "expected 'all' in the performance report"
   assert len(json_data.keys()) == 1, "only expect one performance report"
   reported_performance = json_data["all"]
   assert 0.0 <= reported_performance <= 1.0, "reported performance does not range between 0.0-1.0" #is this correct?
   return reported_performance   

## write performance reports for global aggregation
def write_global_report(args_name,global_performances,single_multi):
   global name
   perf_df = pd.DataFrame([global_performances],columns=\
      ['aucpr_mean','aucroc_mean','maxf1_mean', 'kappa_mean', 'tn', 'fp', 'fn', 'tp', 'vennabers_mean'])
   fn = name + '/' + single_multi + "_" + os.path.basename(args_name) + '_global_performances.csv'
   perf_df.to_csv(fn)
   vprint(f"Wrote {single_multi} global report to: {fn}")
   return perf_df

## write performance reports per-task & per-task_assay
def write_aggregated_report(args_name,local_performances, single_multi):
   vprint(f"{args_name} performances shape: " + str(local_performances.shape))
   # write per-task report
   fn1 = name + '/' + single_multi + "_" + os.path.basename(args_name) + '_per-task_performances.csv'
   vprint(f"Wrote {single_multi} per-task report to: {fn1}")
   local_performances.to_csv(fn1)
   # write per-assay type, ignore task id
   df = local_performances.loc[:,'assay type':].groupby('assay type').mean()
   fn2 = name + '/' + single_multi + "_" + os.path.basename(args_name) + '_per-assay_performances.csv'
   df.to_csv(fn2)
   vprint(f"Wrote {single_multi} per-assay report to: {fn2}")
   return

## Convert the on-premise predictions into the sparse output expected for performance eval.
def mask_y_hat(true_path, pred_path_single, pred_path_multi, single_task_input, multi_task_input):
   """
   in single pharma runs, the input file of predictions is not a pred, but Yhat file.
   the Yhat file (pred_data) needs to be prepared to be used for performance evaluation
   - only keep compounds from valdidation fold
   - remove unneded tasks
   - mask predictions (full matrix) to represent sparse matrix of true lables
   In:
      - true_path <string>: path to true labels matrix (sparsely populated)
      - pred_path <string>: path to predicted labels matrix (densely populated)
   Out:
      - pred_data <pandas df>: df containing the predictions in same shape and mask as te true labels matrix
   """
   # ToDO: put data reading in different function 
   # ToDO: test with weights
   global folding
   global fold_va
   
   # load the data
   true_data = np.load(true_path, allow_pickle = True)
   pred_data_single = np.load(pred_path_single, allow_pickle = True)
   pred_data_multi = np.load(pred_path_multi, allow_pickle = True)

   true_data = true_data.tolist()   
   true_data = true_data.tocsr()
   
   #debugging
   vprint(true_data.shape)
   vprint(pred_data_single.shape)   
   vprint(pred_data_multi.shape)   
      
   # only keep validation fold
   true_data = true_data[folding == fold_va] #only keeping validation fold
   pred_data_single = pred_data[folding == fold_va] #only keeping validation fold
   pred_data_multi = pred_data[folding == fold_va] #only keeping validation fold

   vprint(true_data.shape)
   vprint(pred_data_single.shape)   
   vprint(pred_data_multi.shape)   
   
   true_data = true_data.todense()

   assert true_data.shape[1] != pred_data_single.shape[1], "True data should have different shape to SP data to mask"
   
   """ remove tasks 
   
   filter out tasks, that we did not predict on (the tasks from SP y hat), 
   since the overall model predicts on all tasks it saw during training, 
   not only the ones relevant for prediction.
   """
   
   """ read data required

   file: results/weight_table_T3_mapped.csv, for the single and the multi pharma data
   contains mappings of the taks, so we can filter the tasks out for the MP y_hat file, since we ust not compare tasks that are not in SP y hat
   """
   SP_tasks_rel = single_task_input[["classification_task_id", "cont_classification_task_id"]] 
   MP_tasks_rel = multi_task_input[["classification_task_id", "cont_classification_task_id"]] 

   #create overall df
   global_index = pd.DataFrame({"id": range(multi_task_input.shape[0] + 1)})

   """
   # we drop all MP row indices from SP and MP index table
   # all remaining relative position-indices (not "id", nor the pandas index, but only position in the table) in SP are the ones we need to keep
   """

   # cont class id as first column
   # extend by essay_type

   # merge global index with task indices from MP and SP task table, which are subsets of the global index
   MP_df = global_index.merge(MP_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')      
   SP_df = global_index.merge(SP_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')

   #get filled indices from MP and keep onlye thse in SP and MP 
   keep_MPs = [not i for i in MP_df["cont_classification_task_id"].isnull()]

   # drop taks missing in MP (to make them same shaped)
   MP_df = MP_df[keep_MPs]
   SP_df = SP_df[keep_MPs]

   # get the final indices to be kept in y hat prediction matrix
   # drop remaining
   task_ids_keep = [not i for i in SP_df["cont_classification_task_id"].isnull()]
   pred_data_single = pred_data_single[:, task_ids_keep]
   pred_data_multi = pred_data_multi[:, task_ids_keep]

   # debugging
   vprint("SP Shape after task removal:" + str(pred_data_single.shape))
   vprint("MP Shape after task removal:" + str(pred_data_multi.shape))


   """ masking y hat

   mask the pred matrix the same as true label matrix, 
   i.e. set all values in pred zero, that are zero (not predicted) in true label matrix
   """

   for row in range(pred_data_single.shape[0]):
      for col in range(pred_data_single.shape[1]):
         if true_data[row, col]== 0:
            pred_data[row, col] = 0

   for row in range(pred_data_multi.shape[0]):
      for col in range(pred_data_multi.shape[1]):
         if true_data[row, col]== 0:
            pred_data[row, col] = 0
            
   assert true_data.shape == pred_data_single.shape, f"True shape {true_data.shape} and SP Pred shape {pred_data.shape} need to be identical"
   assert true_data.shape == pred_data_multi.shape, f"True shape {true_data.shape} and MP Pred shape {pred_data.shape} need to be identical"
   
   #phase 2 derisk output (by each pharma) check that the aggregated performance on the platform is
   #numerically identical (difference < 1e-5) to the aggregated performance computed from the model on the pharma premises.
   allclose = np.allclose(pred_data_single, pred_data_multi, rtol=1e-05, atol=1e-05)
   if args.derisk and not allclose:
         vprint(f"WARNING!! (Phase 2 de-risk output check [--derisk]): there is problem with {pred_path_single} and {pred_path_multi}. The yhat supplied by the substra platform are not close (tol:1e-05)")
   #if not derisk and all are close then may be issue with on-premise outputs
   if not args.derisk and allclose:
         vprint(f"WARNING!! (Federated vs on comparison check): y-hats for {pred_path_single} and {pred_path_multi} appear to be too close (tol:1e-05)")



   # debugging
   vprint("out shape" + str(pred_data_single.shape) + str(pred_data_multi.shape))
   return pred_data_single, pred_data_multi

## run performance code for single- or multi-pharma run +
## check the global_pre_calculated_performance with the reported performance json
def per_run_performance(y_pred_arg, performance_report, single_multi, tasks_table):
   global y_true
   global tw_df
   global pred
   global args
   
   if pred:
      y_pred = torch.load(y_pred_arg)
   else:
      y_pred = sparse.csc_matrix(y_pred_arg)   
   
   ## checks to make sure y_true and y_pred match
   assert y_true.shape == y_pred.shape, f"{single_multi} y_true shape do not match y_pred ({y_true.shape} & {y_pred.shape})"
   assert y_true.nnz == y_pred.nnz, f"{single_multi} y_true number of nonzero values do not match y_pred"
   assert (y_true.indptr == y_pred.indptr).all(), f"{single_multi} y_true indptr do not match y_pred"
   assert (y_true.indices == y_pred.indices).all(), f"{single_multi} y_true indices do not match y_pred"
   
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
   # local_performance.loc[:,"aucpr":] = local_performance.loc[:,"aucpr":].apply(pd.to_numeric)
   vprint(local_performance)
   for c in local_performance.iloc[:,2:].columns:
      local_performance.loc[:,c] = local_performance.loc[:,c].astype(float)
   # local_performance.loc[:,"aucpr":] = local_performance.loc[:,"aucpr":].astype(float)
   write_aggregated_report(y_pred_arg,local_performance, single_multi)
                  
   ##global aggregation:
   if args.task_weights: tw_weights=tw_df['weight'].values[cols55]
   else: tw_weights=tw_df[cols55]
   aucpr_mean  = np.average(aucpr[cols55],weights=tw_weights)
   aucroc_mean = np.average(aucroc[cols55],weights=tw_weights)
   maxf1_mean  = np.average(maxf1[cols55],weights=tw_weights)
   kappa_mean  = np.average(kappa[cols55],weights=tw_weights)
   tn_sum = tn[cols55].sum()
   fp_sum = fp[cols55].sum()
   fn_sum = fn[cols55].sum()
   tp_sum = tp[cols55].sum()
   vennabers_mean  = np.average(vennabers[cols55],weights=tw_weights)
   
   if pred and performance_report:
      global_pre_calculated_performance = global_perf_from_json(performance_report)
      #only assert pre-calculated performance if not weight averaging for compatability
      if not args.task_weights:
         assert global_pre_calculated_performance == aucpr_mean, f"Reported performance in {performance_report} ({global_pre_calculated_performance}) does not match calculated performance for {y_pred_arg} ({aucpr_mean})"
         vprint(f"Check passed: Reported performance in {performance_report} ({global_pre_calculated_performance}) match the calculated performance for {y_pred_arg} ({aucpr_mean})")
   global_performance = write_global_report(y_pred_arg,[aucpr_mean,aucroc_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum, vennabers_mean], single_multi)
   #if pred then we need to return the yhats to check phase 2 derisk output (by each pharma)
   if pred: return y_pred,[local_performance,global_performance]
   else: return [local_performance,global_performance]

## calculate the difference between the single- and multi-pharma outputs and write to a file
def calculate_deltas(single_results, multi_results):
   for idx, delta_comparison in enumerate(['locals','/deltas_global_performances.csv']):

      assert single_results[idx].shape[0] == multi_results[idx].shape[0], "the number of tasks are not equal between the single- and multi-pharma runs"
      assert single_results[idx].shape[1] == multi_results[idx].shape[1], "the number of reported metrics are not equal between the single- and multi-pharma runs"
      
      # add assay aggregation if local
      if(delta_comparison == 'locals'):
         at = multi_results[idx]["assay type"]
         # delta = (multi_results[idx].iloc[:, 1:]-single_results[idx].iloc[:, 1:])
         delta = (multi_results[idx].loc[:, "aucpr":]-single_results[idx].loc[:, "aucpr":])
         tdf = pd.concat([at, delta], axis = 1)
         fn1 = name + '/deltas_per-task_performances.csv'
         tdf.to_csv(fn1)
         vprint(f"Wrote per-task delta report to: {fn1}")
         
         # aggregate on assay_type level
         fn2 = name + '/deltas_per-assay_performances.csv'
         tdf.groupby("assay type").mean().to_csv(fn2)
         vprint(f"Wrote per-assay delta report to: {fn2}")
      else:
         allclose = np.allclose(single_results[idx], multi_results[idx], rtol=1e-05, atol=1e-05)
         #phase 2 derisk output (by each pharma) check that the aggregated performance on the platform is
         #numerically identical (difference < 1e-5) to the aggregated performance computed from the model on the pharma premises.
         if args.derisk and not allclose:
               vprint(f"WARNING!! (Phase 2 de-risk output check [--derisk]): there is a mistake in the aggregated metrics or in the performance reported by the substra platform (tol:1e-05)")
         #if not derisk and all are close then may be issue with on-premise outputs
         if not args.derisk and allclose:
               vprint(f"WARNING!! (Federated vs on-premise comparison check): calculated global single- vs. multi-pharma deltas appear to be too close (tol:1e-05)")
         (multi_results[idx]-single_results[idx]).to_csv(name + delta_comparison)

##function to call allclose check for pred files (masking means this is already done for npy files)
def pred_mode_allclose_check(single_yhat,multi_yhat):
   allclose = np.allclose(single_yhat.data, multi_yhat.data, rtol=1e-05, atol=1e-05)
   if args.derisk and not allclose:
      vprint(f"WARNING!! (Phase 2 de-risk output check [--derisk]): there is problem in the substra platform, yhats not close (tol:1e-05)")
   if not args.derisk and allclose:
      vprint(f"WARNING!! (Federated vs on premise comparison check): on-premise yhats appear to be too close (tol:1e-05)")


#if federated output then no need to mask
if pred:
   vprint(f"Calculating '{args.y_pred_single}' and '{args.y_pred_multi}' performance for 'pred' type input (federated output) files")
   single_yhat, single_partner_results = per_run_performance(args.y_pred_single,args.single_performance_report, "single", single_tasks)
   multi_yhat, multi_partner_results = per_run_performance(args.y_pred_multi,args.multi_performance_report, "multi", multi_tasks)
   pred_mode_allclose_check(single_yhat,multi_yhat)
#if not pred, then output is on-premise and requires 'mask_y_hat'
else:
   vprint(f"Calculating '{args.y_pred_single}' and '{args.y_pred_multi}' performance for 'npy' on-premise files (with masking)")
   masked_single_data, masked_multi_data=mask_y_hat(args.y_true_all, args.y_pred_single, args.y_pred_multi, single_tasks, multi_tasks)
   single_partner_results = per_run_performance(masked_single_data,args.multi_performance_report, "single", single_tasks)
   multi_partner_results = per_run_performance(masked_multi_data,args.multi_performance_report, "multi", multi_tasks)

vprint(f"Calculating delta between '{args.y_pred_single}' & '{args.y_pred_multi}' performances.")
calculate_deltas(single_partner_results,multi_partner_results)
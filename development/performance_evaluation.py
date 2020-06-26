import os, sys
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
import json 
from VennABERS import get_VA_margin_median_cross


parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, default="T10_y.npy")
parser.add_argument("--y_pred_single", help="Yhat prediction output from single-pharma run (./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred)", type=str, required=True)
parser.add_argument("--y_pred_multi", help="Yhat prediction output from multi-pharma run (./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred)", type=str, required=True)
parser.add_argument("--folding", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, default="folding.npy")
parser.add_argument("--task_weights", help="CSV file with columns task_id and weight (i.e. from files_4_ml/)", type=str, default=None)
parser.add_argument("--single_performance_report", help="JSON file with global reported single-pharma performance (i.e. ./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=True)
parser.add_argument("--multi_performance_report", help="JSON file with global reported multi-pharma performance (i.e. ./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=True)
parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
args = parser.parse_args()


def vprint(s=""):
    if args.verbose:
        print(s)     
vprint(args)

assert pd.__version__[:4] >='0.25', "Pandas version must be >=0.25"

if args.filename is not None:
    name = args.filename
else:
    name  = f"{os.path.basename(args.y_true_all)}_{os.path.basename(args.y_pred_single)}_{os.path.basename(args.y_pred_multi)}_{os.path.basename(args.folding)}"
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
    assert "weight" in tw_df.columns, "weight is missing in task weights CVS file"
    assert tw_df.shape[1] == 2, "task weight file (CSV) must only have 2 columns"

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

def global_perf_from_json(performance_report):
	with open(performance_report, "r") as fi:
		json_data = json.load(fi)
	assert 'all' in json_data.keys(), "expected 'all' in the performance report"
	assert len(json_data.keys()) == 1, "only expect one performance report"
	reported_performance = json_data["all"]
	assert 0.0 <= reported_performance <= 1.0, "reported performance does not range between 0.0-1.0" #is this correct?
	return reported_performance	

def write_global_report(args_name,global_performances):
	global name
	perf_df = pd.DataFrame([global_performances],columns=\
		['aucpr_mean','aucroc_mean','maxf1_mean', 'kappa_mean', 'tn', 'fp', 'fn', 'tp', 'vennabers_mean'])
	perf_df.to_csv(name + '/'+ os.path.basename(args_name) + '_global_performances.csv')
	return perf_df

def write_local_report(args_name,local_performances):
	local_performances.to_csv(name + '/'+ os.path.basename(args_name) + '_local_performances.csv')
	return

## run performance code for single- or multi-pharma run
def per_run_performance(y_pred_arg, performance_report):
	global y_true
	global tw_df
	y_pred = torch.load(y_pred_arg)
	## checks to make sure y_true and y_pred match
	assert y_true.shape == y_pred.shape, f"y_true shape do not match y_pred ({y_true.shape} & {y_pred.shape})"
	assert y_true.nnz == y_pred.nnz, "y_true number of nonzero values do not match y_pred"
	assert (y_true.indptr == y_pred.indptr).all(), "y_true indptr do not match y_pred"
	assert (y_true.indices == y_pred.indices).all(), "y_true indices do not match y_pred"

	aucpr   = np.full(y_true.shape[1], np.nan)
	aucroc  = np.full(y_true.shape[1], np.nan)
	maxf1   = np.full(y_true.shape[1], np.nan)
	kappa   = np.full(y_true.shape[1], np.nan)
	tn      = np.full(y_true.shape[1], np.nan)
	fp      = np.full(y_true.shape[1], np.nan)
	fn      = np.full(y_true.shape[1], np.nan)
	tp      = np.full(y_true.shape[1], np.nan)
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

		y_classes   = np.where(y_pred_col > 0.5, 1, 0)
		precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true_col, probas_pred = y_pred_col)
		aucpr[col]  = sklearn.metrics.auc(x = recall, y = precision)
		aucroc[col] = sklearn.metrics.roc_auc_score(y_true  = y_true_col, y_score = y_pred_col)
		maxf1[col]  = find_max_f1(precision, recall)
		kappa[col]  = sklearn.metrics.cohen_kappa_score(y_true_col, y_classes)
		tn[col], fp[col], fn[col], tp[col] = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()
		vennabers[col] = get_VA_margin_median_cross(pts)

	##local performance:
	local_performance=pd.DataFrame(np.array([aucpr[cols55],aucroc[cols55],maxf1[cols55],\
						kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55], vennabers[cols55]]).T,\
						columns=['aucpr','aucroc','maxf1','kappa','tn','fp','fn','tp', 'vennabers'])
	write_local_report(y_pred_arg,local_performance)
						
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

	global_pre_calculated_performance = global_perf_from_json(performance_report)
	#only assert pre-calculated performance if not weight averaging for compatability
	if not args.task_weights:
		assert global_pre_calculated_performance == aucpr_mean, f"reported performance in {performance_report} ({global_pre_calculated_performance}) does not match calculated performance for {y_pred_arg} ({aucpr_mean})"
	global_performance = write_global_report(y_pred_arg,[aucpr_mean,aucroc_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum, vennabers_mean])
	return [local_performance,global_performance]


def calculte_deltas(single_results, muilti_results):
	for idx, delta_comparison in enumerate(['local_deltas.csv','global_deltas.csv']):
		assert single_results[idx].shape[0] == muilti_results[idx].shape[0], "the number of tasks are not equal between the single- and multi-pharma runs"
		assert single_results[idx].shape[1] == muilti_results[idx].shape[1], "the number of reported metrics are not equal between the single- and multi-pharma runs"
		(muilti_results[idx]-single_results[idx]).to_csv(name + '/' + delta_comparison)

vprint(f"Calculating '{args.y_pred_single}' performance.")
single_partner_results=per_run_performance(args.y_pred_single,args.single_performance_report)
vprint(f"Calculating '{args.y_pred_multi}' performance.")
muilti_partner_results=per_run_performance(args.y_pred_multi,args.multi_performance_report)
vprint(f"Calculating delta between '{args.y_pred_single}' & '{args.y_pred_multi}' performances.")
calculte_deltas(single_partner_results,muilti_partner_results)
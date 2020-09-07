import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
import json
import scipy.sparse as sparse
from scipy.stats import spearmanr
from  pathlib import Path

def init_arg_parser():
	parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
	parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, required=True)
	parser.add_argument("--task_map", help="Taskmap from MELLODDY_tuner output of single run (i.e. from results/weight_table_T3_mapped.csv)", required=True)
	parser.add_argument("--folding", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, required=True)
	parser.add_argument("--task_weights", help="(Optional) CSV file with columns task_id and weight (i.e.  files_4_ml/T9_red.csv)", type=str, default=None)
	parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
	parser.add_argument("--use_venn_abers", help="Toggle to turn on Venn-ABERs code", action='store_true', default=False)
	parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
	parser.add_argument("--f1", help="Output from the first run to compare (pred or .npy)", type=str, required=True)
	parser.add_argument("--f2", help="Output from the second run to compare (pred or .npy)", type=str, required=True)
	args = parser.parse_args()
	return args

def vprint(s=""):
	if args.verbose:
		print(s)

#find f1 
def find_max_f1(precision, recall):
	F1	= np.zeros(len(precision))
	mask = precision > 0
	F1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
	return F1.max()

## write performance reports for global aggregation
def write_global_report(global_performances, fname, name):	
	cols = ['aucpr_mean','aucroc_mean','logloss_mean','maxf1_mean', 'kappa_mean', 'tn', 'fp', 'fn', 'tp']
	if args.use_venn_abers: cols += ['vennabers_mean']
	perf_df = pd.DataFrame([global_performances],columns=cols)
	fn = name + '/' + fname + "_global_performances.csv"
	perf_df.to_csv(fn, index=None)
	vprint(f"Wrote {fname} global performance report to: {fn}")
	return perf_df

## write performance reports per-task & per-task_assay
def write_aggregated_report(local_performances, fname, name, task_map):
	# write per-task report
	df = local_performances[:]
	df['classification_task_id'] = df['classification_task_id'].astype('int32')
	df = df.merge(task_map, right_on=["classification_task_id","assay_type"], left_on=["classification_task_id","assay_type"], how="left")
	fn1 = name + '/' + fname + "_per-task_performances.csv"
	df.to_csv(fn1, index=False)
	vprint(f"Wrote {fname} per-task report to: {fn1}")
	# write per-assay_type, ignore task id
	df2 = local_performances.loc[:,'assay_type':].groupby('assay_type').mean()
	fn2 = name + '/' + fname + "_per-assay_performances.csv"
	df2.to_csv(fn2, index=False)
	vprint(f"Wrote {fname} per-assay report to: {fn2}")
	return

#load either pred or npy yhats and mask if needed, for an input filename
def load_yhats(input_f, folding, fold_va, y_true):
	# load the data
	if input_f.suffix == '.npy':
		vprint(f'Loading (npy) predictions for: {input_f}') 
		yhats = np.load(input_f, allow_pickle=True).item().tocsr().astype('float32')
		ftype = 'npy'
	else:
		vprint(f'Loading (pred) output for: {input_f}') 
		yhats = torch.load(input_f).astype('float32')
		ftype = 'pred'
	# mask validation fold if possible
	try: yhats = yhats[folding == fold_va]
	except IndexError: pass
	return yhats, ftype

#perform masking, report any error in shapes, and return data for f1 and f2
def mask_y_hat(f1_path, f2_path, folding, fold_va, y_true):
	true_data = y_true.astype(np.uint8).todense()
	f1_yhat, f1_ftype = load_yhats(f1_path, folding, fold_va, y_true)
	assert true_data.shape == f1_yhat.shape, f"True shape {true_data.shape} and {args.f1} shape {f1_yhat.shape} need to be identical"
	f2_yhat, f2_ftype = load_yhats(f2_path, folding, fold_va, y_true)
	assert true_data.shape == f2_yhat.shape, f"True shape {true_data.shape} and {args.f2} shape {f2_yhat.shape} need to be identical"
	return [f1_yhat, f2_yhat, f1_ftype, f2_ftype]

## check the pre_calculated_performance with the reported performance json
def per_run_performance(y_pred, pred_or_npy, tasks_table, y_true, tw_df, task_map, name, flabel):	
	if args.use_venn_abers: from VennABERS import get_VA_margin_median_cross
	if pred_or_npy == 'npy': y_pred = sparse.csc_matrix(y_pred)
	## checks to make sure y_true and y_pred match
	assert y_true.shape == y_pred.shape, f"y_true shape do not match {pred_or_npy} y_pred ({y_true.shape} & {y_pred.shape})"
	assert y_true.nnz == y_pred.nnz, f"y_true number of nonzero values do not match {pred_or_npy} y_pred"
	assert (y_true.indptr == y_pred.indptr).all(), f"y_true indptr do not match {pred_or_npy} y_pred"
	assert (y_true.indices == y_pred.indices).all(), f"y_true indices do not match {pred_or_npy} y_pred"

	task_id = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
	assay_type = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
	aucpr	= np.full(y_true.shape[1], np.nan)
	logloss = np.full(y_true.shape[1], np.nan)
	aucroc  = np.full(y_true.shape[1], np.nan)
	maxf1	= np.full(y_true.shape[1], np.nan)
	kappa	= np.full(y_true.shape[1], np.nan)
	tn	 = np.full(y_true.shape[1], np.nan)
	fp	 = np.full(y_true.shape[1], np.nan)
	fn	 = np.full(y_true.shape[1], np.nan)
	tp	 = np.full(y_true.shape[1], np.nan)
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
		y_classes	= np.where(y_pred_col > 0.5, 1, 0).astype(np.uint8)
		precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true_col, probas_pred = y_pred_col)
		aucpr[col]  = sklearn.metrics.auc(x = recall, y = precision)
		#logloss must be float64 to avoid issues with nans in output (e.g. https://stackoverflow.com/questions/50157689/)
		logloss[col]  = sklearn.metrics.log_loss(y_true=y_true_col.astype("float64"), y_pred=y_pred_col.astype("float64"))
		aucroc[col] = sklearn.metrics.roc_auc_score(y_true=y_true_col, y_score=y_pred_col)
		maxf1[col]  = find_max_f1(precision, recall)
		kappa[col]  = sklearn.metrics.cohen_kappa_score(y_true_col, y_classes)
		tn[col], fp[col], fn[col], tp[col] = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()
		##per-task performance:
		cols = ['classification_task_id', 'assay_type', 'aucpr','aucroc','logloss','maxf1','kappa','tn','fp','fn','tp']
		if args.use_venn_abers: 
			vennabers[col] = get_VA_margin_median_cross(pts)
			local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],aucpr[cols55],aucroc[cols55],logloss[cols55],maxf1[cols55],\
				kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55], vennabers[cols55]]).T, columns=cols+['vennabers'])
		else:
			local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],aucpr[cols55],aucroc[cols55],logloss[cols55],maxf1[cols55],\
				kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55]]).T, columns=cols)

	##correct the datatypes for numeric columns
	for c in local_performance.iloc[:,2:].columns:
		local_performance.loc[:,c] = local_performance.loc[:,c].astype('float32')
	##write per-task & per-assay_type performance:
	write_aggregated_report(local_performance, flabel, name, task_map)

	##global aggregation:
	tw_weights=tw_df[cols55]
	aucpr_mean  = np.average(aucpr[cols55],weights=tw_weights)
	aucroc_mean = np.average(aucroc[cols55],weights=tw_weights)
	logloss_mean = np.average(logloss[cols55],weights=tw_weights)
	maxf1_mean  = np.average(maxf1[cols55],weights=tw_weights)
	kappa_mean  = np.average(kappa[cols55],weights=tw_weights)
	tn_sum = tn[cols55].sum()
	fp_sum = fp[cols55].sum()
	fn_sum = fn[cols55].sum()
	tp_sum = tp[cols55].sum()

	if args.use_venn_abers:
		vennabers_mean  = np.average(vennabers[cols55],weights=tw_weights)
		global_performance = write_global_report([aucpr_mean,aucroc_mean,logloss_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum, vennabers_mean], flabel, name)
	else: global_performance = write_global_report([aucpr_mean,aucroc_mean,logloss_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum], flabel, name)
	return [local_performance,global_performance]

## calculate the difference between the single- and multi-pharma outputs and write to a file
def calculate_deltas(f1_results, f2_results, name, task_map):
	for idx, delta_comparison in enumerate(['locals','/deltas_global_performances.csv']):
		assert f1_results[idx].shape[0] == f2_results[idx].shape[0], "the number of tasks are not equal between the --f1 and --f2 outputs for {delta_comparison}"
		assert f1_results[idx].shape[1] == f2_results[idx].shape[1], "the number of reported metrics are not equal between the --f1 and --f2 outputs for {delta_comparison}"
		# add assay aggregation if local
		if(delta_comparison == 'locals'):
			cti = f2_results[idx]["classification_task_id"]
			at = f2_results[idx]["assay_type"]
			delta = (f2_results[idx].loc[:, "aucpr":]-f1_results[idx].loc[:, "aucpr":])
			tdf = pd.concat([cti, at, delta], axis = 1)
			fn1 = name + '/deltas_per-task_performances.csv'
			pertask = tdf[:]
			pertask['classification_task_id'] = pertask['classification_task_id'].astype('int32')
			pertask = pertask.merge(task_map, right_on=["classification_task_id","assay_type"], left_on=["classification_task_id","assay_type"], how="left")
			pertask.to_csv(fn1, index= False)
			vprint(f"Wrote per-task delta report to: {fn1}")

			# aggregate on assay_type level
			fn2 = name + '/deltas_per-assay_performances.csv'
			tdf.groupby("assay_type").mean().to_csv(fn2)
			vprint(f"Wrote per-assay delta report to: {fn2}")
		else:
			(f2_results[idx]-f1_results[idx]).to_csv(name + delta_comparison, index=False)

def main(args):
	vprint(args)
	y_pred_f1_path = Path(args.f1)
	y_pred_f2_path = Path(args.f2)

	assert all([(p.suffix == '.npy') or (p.stem in ['pred']) for p in [y_pred_f1_path, y_pred_f2_path]]), "Pediction files need to be either 'pred' or '*.npy'"
	task_map = pd.read_csv(args.task_map)

	if args.filename is not None:
		name = args.filename
	else:
		name = f"{os.path.basename(args.y_true_all)}_{os.path.basename(args.f1)}_{os.path.basename(args.f2)}_{os.path.basename(args.folding)}"
		vprint(f"\nRun name is '{name}'\n")
	assert not os.path.exists(name), f"{name} already exists... exiting"
	os.makedirs(name)
	with open(f'{name}/run_params.json', 'wt') as f:
		json.dump(vars(args), f, indent=4)
		vprint(f"Wrote input params to '{name}/run_params.json'\n")

	#load the folding/true data
	folding = np.load(args.folding)
	vprint(f'Loading y_true: {args.y_true_all}') 
	y_true_all = np.load(args.y_true_all, allow_pickle=True).item()
	y_true_all = y_true_all.tocsc()

	## filtering out validation fold
	fold_va = 1
	y_true = y_true_all[folding == fold_va]
	y_true_all = None #clear all ytrue to save memory

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
		tw_df = np.ones(y_true.shape[1], dtype=np.uint8)

	f1_yhat, f2_yhat, f1_ftype, f2_ftype = mask_y_hat(y_pred_f1_path,y_pred_f2_path, folding, fold_va, y_true)
	folding, fold_va = None, None #clear folding - no longer needed

	vprint(f"\nCalculating '{args.f1}' performance")
	f1_yhat_results = per_run_performance(f1_yhat, f1_ftype, task_map, y_true, tw_df, task_map, name, 'f1')
	f1_yhat = None #clear yhat from memory - no longer needed

	vprint(f"\nCalculating '{args.f2}' performance")
	f2_yhat_results = per_run_performance(f2_yhat, f2_ftype, task_map, y_true, tw_df, task_map, name, 'f2')
	f2_yhat=None #clear yhat from memory - no longer needed

	y_true, tw_df = None, None #clear ytrue and weights - no longer needed
	vprint(f"\nCalculating delta between '{args.f1}' and '{args.f2}' performances")
	calculate_deltas(f1_yhat_results,f2_yhat_results, name, task_map)
	vprint(f"\nRun name '{name}' is finished.")
	return

if __name__ == '__main__':
	start = time.time()
	args = init_arg_parser()
	vprint('\n=== WP3 Performance evaluation script for npy and pred files ===\n')
	try:
		from memory_profiler import memory_usage
		mem_usage = memory_usage(main(args))
		vprint('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
		vprint('Maximum memory usage: %s' % max(mem_usage))
	except ModuleNotFoundError:
		vprint('Not monitoring memory usage (no import for "memory_profiler" module)\n')
		main(args)
	end = time.time()
	vprint(f'Performance evaluation took {end - start:.08} seconds.')

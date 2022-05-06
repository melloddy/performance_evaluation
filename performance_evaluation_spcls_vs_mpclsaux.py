import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.metrics import confusion_matrix, brier_score_loss
import rdkit
from rdkit.ML.Scoring.Scoring import CalcBEDROC

import json
import scipy.sparse as sparse
from scipy.stats import spearmanr
from  pathlib import Path
from scipy.io import mmread
from scipy.interpolate import interp1d
import sparsechem as sc
import significance_analysis
		
def init_arg_parser():
	parser = argparse.ArgumentParser(description="MELLODDY Year 3 Performance Evaluation")
	parser.add_argument("--y_cls", help="Classification activity file (npz) (e.g. cls_T10_y.npz)", type=str, default=None)
	parser.add_argument("--y_clsaux", help="Aux classification activity file (npz) (e.g. cls_T10_y.npz)", type=str, default=None)

	parser.add_argument("--y_cls_single_partner", "--y_cls_run1", help="Yhat cls prediction output from an single-partner run (e.g. <single pharma dir>/<cls_prefix>-class.npy)", type=str, default=None)
	parser.add_argument("--y_clsaux_single_partner", "--y_clsaux_run1", help="Yhat clsaux prediction from an single-partner run (e.g. <single pharma dir>/<clsaux_prefix>-class.npy)", type=str, default=None)

	parser.add_argument("--y_cls_multi_partner", "--y_cls_run2", help="Classification prediction output for comparison (e.g. pred from the multi-partner run)", type=str, default=None)
	parser.add_argument("--y_clsaux_multi_partner", "--y_clsaux_run2", help="Classification w/ aux prediction output for comparison (e.g. pred from the multi-partner run)", type=str, default=None)

	parser.add_argument("--folding_cls", help="Folding file (npy) (e.g. cls_T11_fold_vector.npy)", type=str, default=None)
	parser.add_argument("--folding_clsaux", help="Folding file (npy) (e.g. cls_T11_fold_vector.npy)", type=str, default=None)

	parser.add_argument("--t8c_cls", help="T8c file for classification in the results_tmp/classification folder", type=str, default=None)
	parser.add_argument("--t8c_clsaux", help="T8c file for classification w/ auxiliary in the results_tmp/classification folder", type=str, default=None)

	parser.add_argument("--t10c_cls", type=str, help="Path to <...>/matrices/cls/T10c_cont.csv", required=True)
	parser.add_argument("--t10c_clsaux", type=str, help="Path to <...>/matrices/clsaux/T10c_cont.csv", required=True)

	parser.add_argument("--weights_cls", help="CSV file with columns task_id and weight (e.g. cls_weights.csv)", type=str, default=None)
	parser.add_argument("--weights_clsaux", help="CSV file with columns task_id and weight (e.g cls_weights.csv)", type=str, default=None)

	parser.add_argument("--run_name", help="Run name directory for results from this output (timestemp used if not specified)", type=str, default=None)
	parser.add_argument("--output_task_sensitive_files", help="Output files with task level sensitive information (1 = Yes, 0 = No sensitive files", type=int, default=0, choices=[0, 1])
	parser.add_argument("--n_best_assays", help="Number of best assays to consider (100 = default)", type=int, default=100)
	parser.add_argument("--use_venn_abers", help="Toggle to turn on Venn-ABERs code", action='store_true', default=False)
	parser.add_argument("--perf_bins_cls", help="calibrated AUCPR/ROCAUC performance bins to identify flip tasks", type=str, nargs='+', default=[0.5,0.6,0.7,0.8,0.9],required=False)
	parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
	parser.add_argument("--validation_fold", help="Validation fold to used to calculate performance", type=int, default=[0], nargs='+', choices=[0, 1, 2, 3, 4])
	parser.add_argument("--aggr_binning_scheme_perf", help="Shared aggregated binning scheme for performances", type=str, nargs='+', default=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],required=False)
	parser.add_argument("--aggr_binning_scheme_perf_delta", help="Shared aggregated binning scheme for delta performances", type=str, nargs='+', default=[-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2],required=False)
	parser.add_argument("--version", help="Version of this script", type=str, default="0.1.7", choices=["0.1.7"])
	args = parser.parse_args()
	assert len(args.aggr_binning_scheme_perf) == 11, f"len of aggr_binning_scheme_perf should be 11, got {len(args.aggr_binning_scheme_perf)}"
	assert len(args.aggr_binning_scheme_perf_delta) == 9, f"len of aggr_binning_scheme_perf_delta should be 9, got {len(args.aggr_binning_scheme_perf_delta)}"
	try:  from statsmodels.distributions.empirical_distribution import ECDF
	except ImportError as e: print(e,': statsmodels needs to be installed for ECDF function... quitting'); quit()
	assert Path()
	args.aggr_binning_scheme_perf=list(map(np.float64,args.aggr_binning_scheme_perf))
	args.aggr_binning_scheme_perf_delta=list(map(np.float64,args.aggr_binning_scheme_perf_delta))
	return args


def vprint(s="", model_category=False, vtype='INFO'):
	separator = '='*135
	if args.verbose:
		print()
		if model_category:
			print(f'{separator}\n{s}\n{separator}')
		else: print(f'[{vtype}]: {s}')


def cut(x, bins, lower_infinite=True, upper_infinite=True, **kwargs):		
	"""
	Custom cut function to cut reported performances via common binning scheme
	"""
	num_labels	  = len(bins) - 1
	include_lowest  = kwargs.get("include_lowest", False)
	right			= kwargs.get("right", True)
	bins_final = bins.copy()
	if upper_infinite:
		bins_final.insert(len(bins),float("inf"))
		num_labels += 1
	if lower_infinite:
		bins_final.insert(0,float("-inf"))
		num_labels += 1
	symbol_lower  = "<=" if include_lowest and right else "<"
	left_bracket  = "(" if right else "["
	right_bracket = "]" if right else ")"
	symbol_upper  = ">" if right else ">="
	labels=[]
	def make_label(i, lb=left_bracket, rb=right_bracket):
		return "{0}{1}-{2}{3}".format(lb, bins_final[i], bins_final[i+1], rb)	
	for i in range(0,num_labels):
		new_label = None
		if i == 0:
			if lower_infinite: new_label = "{0} {1}".format(symbol_lower, bins_final[i+1])
			elif include_lowest: new_label = make_label(i, lb="[")
			else: new_label = make_label(i)
		elif upper_infinite and i == (num_labels - 1): new_label = "{0} {1}".format(symbol_upper, bins_final[i])
		else: new_label = make_label(i)
		labels.append(new_label)
	return pd.cut(x, bins_final, labels=labels, **kwargs)


def getheader(run_type):
	"""
	Set the classification or regression header for pandas etc. to use
	"""
	if run_type in ['cls','clsaux']: return 'classification'
	else: return 'regression'
	
	
def validate_cls_clsuax_regr_inputs(args):
	"""
	Check required files for cls, clsaux or regr are supplied
	"""
	assert args.y_cls and args.y_clsaux, "Must provide y_cls & y_cls_aux"
	if args.y_cls:
		assert all([args.y_cls_single_partner,args.t8c_cls,args.weights_cls]), "Must provide y_cls_single_partner, y_cls_multi_partner, t8c_cls, weights_cls when supplying y_cls"
	if args.y_clsaux:
		assert all([args.y_clsaux_multi_partner,args.t8c_clsaux,args.weights_clsaux]), "Must provide y_clsaux_single_partner, y_clsaux_multi_partner, t8c_clsaux, weights_clsaux when supplying y_clsaux"
	return


def validate_ytrue_ypred(y_true, y_pred, pred_or_npy):
	assert y_true.shape == y_pred.shape, f"y_true shape do not match {pred_or_npy} y_pred ({y_true.shape} & {y_pred.shape})"
	assert y_true.nnz == y_pred.nnz, f"y_true number of nonzero values {y_true.nnz} do not match {pred_or_npy} y_pred {y_pred.nnz}"
	assert (y_true.indptr == y_pred.indptr).all(), f"y_true indptr do not match {pred_or_npy} y_pred"
	assert (y_true.indices == y_pred.indices).all(), f"y_true indices do not match {pred_or_npy} y_pred"
	return


def load_yhats(input_f, folding, fold_va, y_true):
	"""
	Load and mask yhats for those considered for evaluation 
	"""
	# load the data
	if input_f.suffix == '.npy':
		vprint(f'Loading (npy) predictions for: {input_f}') 
		yhats = np.load(input_f, allow_pickle=True).item().tocsr().astype('float64')
		ftype = 'npy'
	else:
		vprint(f'Loading (pred) output for: {input_f}') 
		yhats = torch.load(input_f).astype('float64').tocsr()
		ftype = 'pred'
	# mask validation fold if possible
	mask=np.array([True if i in fold_va else False for i in folding])
	try: yhats = yhats[mask]
	except IndexError: pass
	return yhats, ftype
	
def map_for_clsaux():
	"""
	Get info to concvert CLSAUX predictions to compare with CLS
	"""
	tw = pd.read_csv(args.weights_cls)
	twx = pd.read_csv(args.weights_clsaux)
	dtypes = {'cont_classification_task_id':int, 'input_assay_id':int, 'threshold':float}
	t10c = pd.read_csv(args.t10c_cls,usecols=dtypes.keys(),dtype=dtypes).drop_duplicates()
	t10cx = pd.read_csv(args.t10c_clsaux,usecols=dtypes.keys(),dtype=dtypes).drop_duplicates()

	tw_t10c = tw.merge(t10c,right_on=['cont_classification_task_id'],left_on=['task_id'],how='inner')
	tw_t10cx = twx.merge(t10cx,right_on=['cont_classification_task_id'],left_on=['task_id'],how='inner')
	twx_dict = pd.merge(tw_t10c, tw_t10cx, on=["input_assay_id", "threshold"], suffixes=["_cls", "_clsaux"])
	twx_dict=dict(zip(twx_dict['task_id_clsaux'],twx_dict['task_id_cls']))

	task2considx = twx.loc[twx['aggregation_weight']==1]['task_id'].values
	task2consid = tw.loc[tw['aggregation_weight']==1]['task_id'].values

	twx=twx.query('task_type != "AUX_HTS"').rename({'task_id':'clsaux_task_id'},axis=1)
	twx['task_id']=twx['clsaux_task_id'].map(twx_dict)
	twx=twx.dropna(subset=[column for column in twx.columns if column !='catalog_id'])

	task2considx = twx.loc[twx['aggregation_weight']==1]['task_id'].values
	task2consid = tw.loc[tw['aggregation_weight']==1]['task_id'].values
	assert (task2considx == task2consid).all(), 'not (task2considx == task2consid).all()'
	return task2consid,twx


def mask_y_hat(f_path, folding, fold_va, y_true, header_type):
	"""
	Load yhats using load_yhats and validate the shapes
	"""
	if header_type == 'classification': true_data = y_true.copy().astype(np.uint8).todense()
	else: true_data = y_true.copy().astype('float64').todense()
	f_yhat, f_ftype = load_yhats(f_path, folding, fold_va, y_true)
	assert true_data.shape == f_yhat.shape, f"True shape {true_data.shape} and {f_path.stem} shape {f_yhat.shape} need to be identical"
	return [f_yhat, f_ftype]


def mask_ytrue(ftype,fname,folding,fold_va):
	"""
	Mask the y-input labels for the validation fold
	"""
	vprint(f'Loading {ftype}: {fname}')
	try: y_all = sc.load_sparse(fname)
	except AttributeError: 
		y_all = mmread(fname)
	y_all = y_all.tocsc()
	mask=np.array([True if i in fold_va else False for i in folding])
	y_all_true = y_all[mask]
	return y_all_true


def check_weights(tw_df, y_true, header_type):
	"""
	Validate task weight file
	"""
	assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
	assert tw_df.shape[1] == 5, "task weight file (CSV) must only have 5 columns"
	assert "training_weight" in tw_df.columns, "weight is missing in task weights CVS file"
	assert "aggregation_weight" in tw_df.columns, "weight is missing in task weights CVS file"
	assert y_true.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
	assert (0 <= tw_df.training_weight).all(), "task weights must not be negative"
	assert (tw_df.training_weight <= 1).all(), "task weights must not be larger than 1.0"
	assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
	assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
	assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
	assert tw_df.shape[0]==y_true.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({y_true.shape[1]})."
	return

def interpolate_ecdf(distribution,metric=None):
	"""
	Interpolate the ECDF
	"""
	from statsmodels.distributions.empirical_distribution import ECDF
	ecdf = ECDF(distribution.values)
	return_values = np.linspace(-1,1,100)
	return ecdf(return_values).round(4), return_values

def calculate_ecdf(full_df, sc_columns, pertask_fn=None, perassay_fn=None):
	"""
	Calculate emperical cumulative distribution function (ECDF).
	Used/calculated in two ways:
		Calculated for both SP and MP (twice) to produce [Y = CDF1:SP(AUC)  – CDF:MP(AUC)] to be plot as a function of AUC
		Calculated directly (once) on the distribution of deltas overall delta calculation
	"""
	ecdf_df=pd.DataFrame()
	ecdf_df_assay_type=pd.DataFrame()
	for metric_bin in full_df.loc[:, f"{sc_columns[0]}":f"{sc_columns[-1]}"].columns:
		ecdf=interpolate_ecdf(full_df[metric_bin], metric=metric_bin)
		ecdf_df=pd.concat((ecdf_df, \
			pd.DataFrame({'Density':ecdf[0], \
			'Metric Value':ecdf[1], \
			'Metric':metric_bin})))

		for assay_type, grouped_df_metric in full_df.groupby('assay_type'):
			ecdf_at=interpolate_ecdf(grouped_df_metric[metric_bin], metric=metric_bin)
			ecdf_df_assay_type=pd.concat((ecdf_df_assay_type, \
			pd.DataFrame({'Density':ecdf_at[0], \
				'Metric Value':ecdf_at[1], \
				'Metric':metric_bin, \
				'Assay_type':assay_type})))
	if pertask_fn: ecdf_df.to_csv(pertask_fn, index= False)
	if perassay_fn: ecdf_df_assay_type.to_csv(perassay_fn, index= False)
	return ecdf_df, ecdf_df_assay_type

def run_performance_calculation(run_type, y_pred, pred_or_npy, y_true, tw_df, task_map, run_name, flabel, rlabel, y_true_cens = None):
	"""
	Calculate performance for one run, bin results and then individual performance reports including aggregation by assay/globally
	"""
	vprint(f"=== Calculating {flabel} performance ===")
	flabel = Path(flabel).stem
	header_type = getheader(run_type)
	y_pred = sparse.csc_matrix(y_pred)
	if header_type == 'classification':
		sc_columns = sc.utils.all_metrics([0],[0],None).columns.tolist()  #get the names of reported metrics from the sc utils
		sc_columns.extend(['brier','s_auc_pr', 'positive_rate','tnr', 'fpr', 'fnr', 'tpr', 'bedroc']) # added for calibrated auc_pr
		if args.use_venn_abers:
			from VennABERS import get_median_VA_margin_cross
			sc_columns.extend(['vam'])
	else:
		sc_columns = sc.utils.all_metrics_regr([0],[0]).columns.tolist()  #get the names of reported metrics from the sc utils
	validate_ytrue_ypred(y_true, y_pred, pred_or_npy) # checks to make sure y_true and y_pred match
	if y_true_cens is not None: validate_ytrue_ypred(y_true_cens, y_pred, pred_or_npy)  # checks to make sure y_cens and y_pred match
	calculated_performance = pd.DataFrame()
	for col_idx, col in enumerate(range(y_true.shape[1])):
		task_id = task_map[f"{header_type}_task_id"][task_map[f"cont_{header_type}_task_id"]==col].iloc[0]
		y_pred_col = (y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]])
		
		#setup for classification metrics
		if header_type == 'classification':
			y_true_col = (y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1)
			if (len(y_true_col) > 0) and not (y_true_col[0] == y_true_col).all():
				try: positive_rate_for_col = np.sum(y_true_col) / len(y_true_col)
				except ZeroDivisionError:
					positive_rate_for_col = 0
				with np.errstate(divide='ignore'):
					sc_calculation = sc.utils.all_metrics(y_true_col,y_pred_col,positive_rate_for_col)
				brier = brier_score_loss(y_true_col,y_pred_col)
				s_auc_pr = calculate_s_auc_pr(sc_calculation.auc_pr[0], y_true_col, positive_rate_for_col)
				tn, fp, fn, tp = confusion_matrix(y_true_col,y_pred_col>0.5).ravel()
				sc_calculation['brier'] = [brier]
				sc_calculation['s_auc_pr'] = [s_auc_pr]
				sc_calculation['positive_rate'] = [positive_rate_for_col]
				sc_calculation['tnr'] = tn/(tn+fp)
				sc_calculation['fpr'] = fp/(fp+tn)
				sc_calculation['fnr'] = fn/(fn+tp)
				sc_calculation['tpr'] = tp/(tp+fn)
				sc_calculation['bedroc'] = rdkit.ML.Scoring.Scoring.CalcBEDROC(sorted(zip(y_pred_col,y_true_col)),1,20)
				if args.use_venn_abers: sc_calculation['vam'] = get_median_VA_margin_cross(y_pred_col, y_true_col)
		details = pd.DataFrame({f'{header_type}_task_id': pd.Series(task_id, dtype='int32'),
								'task_size': pd.Series(len(y_true_col), dtype='int32')})

		if y_true_col.shape[0] <= 1: continue
		if (y_true_col[0] == y_true_col).all(): continue
		sc_metrics = pd.concat([details,sc_calculation],axis=1)
		calculated_performance = pd.concat([calculated_performance, sc_metrics],axis=0)	
	#merge calculated performances with the details of the tasks
	calculated_performance = calculated_performance.merge(task_map, left_on=f'{header_type}_task_id', right_on=f'{header_type}_task_id',how='left')
	##write per-task & per-assay_type performance:
	write_aggregated_report(run_name, run_type, flabel, calculated_performance, sc_columns, header_type, rlabel)
	##global aggregation:
	globally_calculated = write_global_report(run_name, run_type, flabel, calculated_performance, sc_columns, rlabel)
	#calculate ecdf:
	calculated_ecdfs = calculate_ecdf(calculated_performance.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1'), sc_columns)
	return calculated_performance, calculated_ecdfs, sc_columns

def calculate_flipped_tasks(f1_results, f2_results, run_name, run_type, header_type,calc_name1, calc_name2, a_thresh):
	
	f1_results = f1_results.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')
	f2_results = f2_results.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')
	assert f1_results.shape[0] == f2_results.shape[0], "the number of tasks are not equal between the outputs for comparison}"
	assert f1_results.shape[1] == f2_results.shape[1], "the number of reported metrics are not equal between the outputs for comparison"
	header_type = getheader(run_type)
	if run_type in ['cls', 'clsaux']:
		metrics = ['auc_pr_cal', 'roc_auc_score']
	else:
		metrics = ['rsquared', 'corrcoef']
	columns = [header_type + '_task_id'] + metrics
	df = f1_results[columns].merge(f2_results[columns], on = columns[0], how = 'outer',\
		 suffixes=['_'+calc_name1, '_'+calc_name2])
	n_tasks_total = len(df)
	l = []

	for metric in metrics:
		for thresh in a_thresh:
			n_tasks_1 = len(df[df[metric + '_' + calc_name1] > thresh])
			n_tasks_2 = len(df[df[metric + '_' + calc_name2] > thresh])    
			delta_tasks = 100 * (n_tasks_2 - n_tasks_1) / n_tasks_total
			try: delta_tasks_percent = 100 * (n_tasks_2 - n_tasks_1) / n_tasks_1
			except ZeroDivisionError: delta_tasks_percent = 0
			
			l.append(
					{
						'metric': metric,
						'thresh': thresh,
						'n_tasks_'+calc_name1: n_tasks_1,
						'n_tasks_'+calc_name2: n_tasks_2,
						'delta_tasks': delta_tasks,
						'delta_tasks_percent': delta_tasks_percent
					}
				)
    
	df_task_count = pd.DataFrame(l)	
	if args.output_task_sensitive_files:
		filename = f"{run_name}/{run_type}/deltas/tasks_perf_bin_count_NOUPLOAD.csv"
		df_task_count.to_csv(filename, index=False)
		vprint(f"Wrote flipped tasks perf count report to: {filename}")
	n_tasks_total = len(df)
	l = []

	for metric in metrics:
		for thresh in a_thresh:
			n_tasks_MP = len(df[df[metric + '_MP'] > thresh])
			
			n_task_flipped_to_1 = 100 * (len(df[(df[metric + '_' + calc_name1] > thresh) & (df[metric + '_' + calc_name2] <= thresh)])) / n_tasks_total        
			n_task_flipped_to_2 = 100 * (len(df[(df[metric + '_' + calc_name1] <= thresh) & (df[metric + '_' + calc_name2] > thresh)])) / n_tasks_total
				
			l.append(
					{
						'metric': metric,
						'thresh': thresh,
						'n_task_flipped_to_'+calc_name1: n_task_flipped_to_1,
						'n_task_flipped_to_'+calc_name2: n_task_flipped_to_2
					}
				)
		
	df_task_flipped = pd.DataFrame(l)
	filename = f"{run_name}/{run_type}/deltas/tasks_perf_bin_flipped.csv"
	df_task_flipped.to_csv(filename, index=False)
	vprint(f"Wrote flipped tasks perf bin report to: {filename}")
	return
	
def calculate_best_assays(full_df, sc_columns, n_best_assays = None, best_fn = None):
	best_df=pd.DataFrame()
	for metric_bin in full_df.loc[:, f"{sc_columns[0]}":f"{sc_columns[-1]}"].columns:
		avg_best=full_df.sort_values(metric_bin,ascending=False).head(n_best_assays)[metric_bin].mean()
		best_df=pd.concat((best_df, \
			pd.DataFrame({'Metric':[metric_bin], \
			'Average':avg_best})))
	if best_fn: best_df.to_csv(best_fn, index=False)
	return

def calculate_delta(f1_results, f2_results, run_name, run_type, sc_columns, header_type, calc_name1, calc_name2, sig = None, n_best_assays = None):
	"""
	Calculate the delta between the outputs and write to a file
	"""
	f1_results = f1_results.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')
	f2_results = f2_results.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')
	assert f1_results.shape[0] == f2_results.shape[0], "the number of tasks are not equal between the outputs for comparison}"
	assert f1_results.shape[1] == f2_results.shape[1], "the number of reported metrics are not equal between the outputs for comparison"
	header_type = getheader(run_type)
	task_id = f2_results[f"{header_type}_task_id"]
	at = f2_results["assay_type"]
	delta = (f2_results.loc[:, sc_columns[0]:sc_columns[-1]]-f1_results.loc[:, sc_columns[0]:sc_columns[-1]])
	tdf = pd.concat([task_id, at, delta], axis = 1)
	pertask = tdf.copy()
	pertask=pertask.dropna(subset=['{header_type}_task_id'])
	try: pertask.loc[:,f'{header_type}_task_id'] = pertask[f'{header_type}_task_id'].astype('int32')
	except: pass
	#add per-task perf aggregated performance delta bins to output
	for metric in pertask.loc[:, sc_columns[0]:sc_columns[-1]].columns:
		pertask.loc[:,f'{metric}_percent'] = cut(pertask[metric].astype('float64'), \
		args.aggr_binning_scheme_perf_delta,include_lowest=True,right=True)
	#merge calculated significances (if set) with the calculated performances
	if sig is not None: pertask = pertask.merge(sig, left_on=f'{header_type}_task_id', right_on=f'{header_type}_task_id',how='left')
	#write per-task perf aggregated performance delta
	if args.output_task_sensitive_files:
		fn1 = f"{run_name}/{run_type}/deltas/deltas_per-task_performances_NOUPLOAD.csv"
		pertask.to_csv(fn1, index= False)
		vprint(f"Wrote per-task delta report to: {fn1}")

	#write the avg performances for n best assays
	best_fn =  f"{run_name}/{run_type}/deltas/delta_best_{n_best_assays}_assays.csv"
	calculate_best_assays(pertask, sc_columns, n_best_assays = n_best_assays, best_fn = best_fn)
	vprint(f"Wrote best {n_best_assays} task reports to: {best_fn}")
	
	#write binned per-task aggregated performance deltas
	agg_deltas=[]
	for metric_bin in pertask.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf=(pertask.groupby(metric_bin)[f'{header_type}_task_id'].agg('count')/len(pertask)).reset_index().rename(columns={f'{header_type}_task_id': f'bin_{metric_bin}'})
		agg_deltas.append(agg_perf.set_index(metric_bin))
	fnagg = f"{run_name}/{run_type}/deltas/deltas_binned_per-task_performances.csv"
	pd.concat(agg_deltas,axis=1).astype(np.float64).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
	vprint(f"Wrote binned performance per-task delta report to: {fnagg}")

	# aggregate on assay_type level
	fn2 = f"{run_name}/{run_type}/deltas/deltas_per-assay_performances.csv"
	per_assay_delta=tdf[['assay_type'] + sc_columns].groupby("assay_type").mean()
	per_assay_delta.to_csv(fn2)
	vprint(f"Wrote per-assay delta report to: {fn2}")

	#write binned per-assay aggregated performance deltas
	agg_deltas2=[]
	for metric_bin in pertask.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf2=(pertask.groupby(['assay_type',metric_bin])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{metric_bin}'})
		agg_perf2[f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (pertask.assay_type==x['assay_type']).sum() ,axis=1).astype('float64')
		agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
		agg_deltas2.append(agg_perf2.set_index(['assay_type',metric_bin]))
	fnagg2 = f"{run_name}/{run_type}/deltas/deltas_binned_per-assay_performances.csv"	
	pd.concat(agg_deltas2,axis=1).astype(np.float64).reset_index().rename(columns={f'{sc_columns[0]}_percent':'perf_bin',}).to_csv(fnagg2,index=False)
	vprint(f"Wrote binned performance per-assay delta report to: {fnagg}")

	#write globally aggregated performance deltas	
	global_delta = pd.DataFrame(f2_results[sc_columns].mean(axis=0)).T - pd.DataFrame(f1_results[sc_columns].mean(axis=0)).T
	global_delta.to_csv(f"{run_name}/{run_type}/deltas/deltas_global_performances.csv", index=False)

	#if significance flag was set then perform that analysis here
	if sig is not None:
		for p in [calc_name1,calc_name2]:
			#write binned per-task significance
			agg_concat=[]
			agg_perf=(pertask.groupby(f'{p}_significant')[f'{header_type}_task_id'].agg('count')/len(pertask)).reset_index().rename(columns={f'{header_type}_task_id': f'percent_{p}_significant'})
			agg_concat.append(agg_perf.set_index(f'{p}_significant'))
			fnagg = f"{run_name}/{run_type}/deltas/delta_{p}_significance.csv"
			pd.concat(agg_concat,axis=1).astype(np.float64).reset_index().to_csv(fnagg,index=False)
			vprint(f"Wrote {p} significance report to: {fnagg}")
		
			#write assay_type significance
			agg_concat2=[]
			agg_perf2=(pertask.groupby(['assay_type',f'{p}_significant'])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{p}_significant'})
			agg_perf2.loc[:,f'percent_{p}_significant']=agg_perf2.apply(lambda x : x[f'count_{p}_significant'] / (pertask.assay_type==x['assay_type']).sum() ,axis=1).astype('float64')
			agg_perf2.drop(f'count_{p}_significant',axis=1,inplace=True)
			agg_concat2.append(agg_perf2.set_index(['assay_type',f'{p}_significant']))
			fnagg2 = f"{run_name}/{run_type}/deltas/delta_per-assay_{p}_significance.csv"
			pd.concat(agg_concat2,axis=1).astype(np.float64).reset_index().to_csv(fnagg2,index=False)
			vprint(f"Wrote per-assay {p} significance report to: {fnagg2}")
	return

def write_global_report(run_name, run_type, fname, local_performances, sc_columns, rlabel):
	"""
	write performance reports for global aggregation
	"""
	df = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1').copy()
	df = pd.DataFrame(df[sc_columns].mean(axis=0)).T
	fn1 = f"{run_name}/{run_type}/{rlabel}/{fname}_global_performances.csv"
	df.to_csv(fn1, index= False)
	vprint(f"Wrote global report to: {fn1}")
	return df

def write_aggregated_report(run_name, run_type, fname, local_performances, sc_columns, header_type, rlabel):
	"""
	write performance reports per-task & per-task_assay
	"""
	df = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1').copy()
	for metric in df.loc[:, sc_columns[0]:sc_columns[-1]].columns:
		df.loc[:,f'{metric}_percent'] = cut(df[metric].astype('float64'), \
		args.aggr_binning_scheme_perf,include_lowest=True,right=True,lower_infinite=False, upper_infinite=False)
	df.loc[:,f'{header_type}_task_id'] = df[f'{header_type}_task_id'].astype('float').astype('int32')
	os.makedirs(f"{run_name}/{run_type}/{rlabel}/")
	if args.output_task_sensitive_files:
		fn1 = f"{run_name}/{run_type}/{rlabel}/{fname}_per-task_performances_NOUPLOAD.csv"
		df.to_csv(fn1, index=False)
		vprint(f"Wrote per-task report to: {fn1}")
		
	#write binned per-task performances
	agg_concat=[]
	for metric_bin in df.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf=(df.groupby(metric_bin)[f'{header_type}_task_id'].agg('count')/len(df)).reset_index().rename(columns={f'{header_type}_task_id': f'bin_{metric_bin}'})
		agg_concat.append(agg_perf.set_index(metric_bin))
	fnagg = f"{run_name}/{run_type}/{rlabel}/{fname}_binned_per-task_performances.csv"
	pd.concat(agg_concat,axis=1).astype(np.float64).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
	vprint(f"Wrote per-task binned performance report to: {fnagg}")

	#write performance aggregated performances by assay_type
	df2 = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1').copy()[['assay_type'] + sc_columns]
	df2 = df2.loc[:,'assay_type':].groupby('assay_type').mean()
	fn2 = f"{run_name}/{run_type}/{rlabel}/{fname}_per-assay_performances.csv"
	df2.to_csv(fn2)
	vprint(f"Wrote per-assay report to: {fn2}")

	#write binned perf performances by assay_type 
	agg_concat2=[]
	for metric_bin in df.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf2=(df.groupby(['assay_type',metric_bin])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{metric_bin}'})
		agg_perf2.loc[:,f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (df.assay_type==x['assay_type']).sum() ,axis=1).astype('float64')
		agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
		agg_concat2.append(agg_perf2.set_index(['assay_type',metric_bin]))
	fnagg2 = f"{run_name}/{run_type}/{rlabel}/{fname}_binned_per-assay_performances.csv"
	pd.concat(agg_concat2,axis=1).astype(np.float32).reset_index().rename(columns={f'{sc_columns[0]}_percent':'perf_bin',}).to_csv(fnagg2,index=False)
	vprint(f"Wrote per-assay binned report to: {fnagg}")
	return

def calculate_single_partner_multi_partner_results(run_name, run_type, y_true, folding, fold_va, t8, task_weights, single_partner, multi_partner, y_true_cens=None):
	"""
	Calculate cls, clsaux or regr performances for single_partner and multi_partner outputs, then calculate delta, print outputs along the way
	"""
	header_type = getheader(run_type)
	y_true_clsaux = mask_ytrue(run_type,y_true[1],folding[1],fold_va)
	y_true = mask_ytrue(run_type,y_true[0],folding[0],fold_va)
	tw_df = pd.read_csv(task_weights[0])
	tw_df.sort_values("task_id", inplace=True)
	tw_df_clsaux = pd.read_csv(task_weights[1])
	tw_df_clsaux.sort_values("task_id", inplace=True)
	check_weights(tw_df,y_true,header_type)
	check_weights(tw_df_clsaux,y_true_clsaux,header_type)
	t8_clsaux = pd.read_csv(t8[1]) #read t8c or t8r files
	t8 = pd.read_csv(t8[0]) #read t8c or t8r files
	task_map = t8.merge(tw_df,left_on=f'cont_{header_type}_task_id',right_on='task_id',how='left').dropna(subset=[f'cont_{header_type}_task_id'])
	task_map_clsaux = t8_clsaux.merge(tw_df_clsaux,left_on=f'cont_{header_type}_task_id',right_on='task_id',how='left').dropna(subset=[f'cont_{header_type}_task_id'])
	y_single_partner_yhat, y_single_partner_ftype = mask_y_hat(single_partner, folding[0], fold_va, y_true, header_type)
	y_multi_partner_yhat, y_multi_partner_ftype = mask_y_hat(multi_partner, folding[1], fold_va, y_true_clsaux, header_type)
	if '--y_cls_run1' in sys.argv and '--y_cls_run2' in sys.argv:
		calc_name1='RUN1'
		calc_name2='RUN2'
	else:
		calc_name1='SP'
		calc_name2='MP'
	sig = None
	y_single_partner_results, sp_calculated_ecdf, _ = run_performance_calculation(run_type, y_single_partner_yhat, y_single_partner_ftype, y_true, tw_df, task_map, run_name, single_partner, calc_name1, y_true_cens = y_true_cens)
	del y_single_partner_yhat
	y_multi_partner_results, mp_calculated_ecdf, sc_columns = run_performance_calculation(run_type, y_multi_partner_yhat, y_multi_partner_ftype, y_true_clsaux, tw_df_clsaux, task_map_clsaux, run_name, multi_partner, calc_name2, y_true_cens = y_true_cens)
	del y_multi_partner_yhat
	os.makedirs(f"{run_name}/{run_type}/deltas/")
	ecdf_fns = [f"{run_name}/{run_type}/deltas/deltas_cdf{calc_name2}-cdf{calc_name1}.csv", f"{run_name}/{run_type}/deltas/deltas_cdf{calc_name2}-cdf{calc_name1}_assay_type.csv"]
	for ecdf_idx, ecdf_merge_cols in enumerate([['Metric Value','Metric'], ['Metric Value','Metric','Assay_type']]):
		ecdf_out = sp_calculated_ecdf[ecdf_idx].merge(mp_calculated_ecdf[ecdf_idx],left_on= ecdf_merge_cols,right_on=ecdf_merge_cols,how='left', suffixes=[f' {calc_name1}',f' {calc_name2}'])
		ecdf_out[f'{calc_name2}-{calc_name1}_CDF'] = ecdf_out[f'Density {calc_name2}'] - ecdf_out[f'Density {calc_name1}']
		ecdf_out.to_csv(ecdf_fns[ecdf_idx],index=False)
	task2consid,twx=map_for_clsaux()
	y_multi_partner_results['task_id']=y_multi_partner_results['task_id'].map(dict(zip(twx['clsaux_task_id'],twx['task_id'])))
	y_multi_partner_results=y_multi_partner_results.dropna(subset=['task_id']).sort_values('task_id')
	calculate_delta(y_single_partner_results, y_multi_partner_results, run_name, run_type, sc_columns, header_type, calc_name1, calc_name2, sig = sig, n_best_assays = args.n_best_assays)
	if run_type in ['cls', 'clsaux']:
		a_thresh = list(map(np.float64,args.perf_bins_cls))
	else:
		a_thresh = list(map(np.float64,args.perf_bins_regr))
	calculate_flipped_tasks(y_single_partner_results, y_multi_partner_results, run_name, run_type, header_type,calc_name1, calc_name2, a_thresh)
	return

def calculate_s_auc_pr(auc_pr, y_true_col, positive_rate_for_col):
	with np.errstate(divide='ignore'):
		try: s_auc_pr = auc_pr ** (np.log10(0.5)/np.log10(positive_rate_for_col))
		except ZeroDivisionError:
			s_auc_pr = auc_pr
	return s_auc_pr


def main(args):
	vprint(args)
	if args.validation_fold != [0]:
		vprint('Expected validation_fold is 0 for the WP3 report, continuing anyway ...', vtype='WARNING')
	validate_cls_clsuax_regr_inputs(args)

	if args.run_name is not None:
		run_name = args.run_name
	else:
		timestr = time.strftime('%Y%m%d-%H%M%S')
		run_name = f"perf_eval_{timestr}"
		vprint(f"Run name is '{run_name}'")
	assert not os.path.exists(run_name), f"{run_name} already exists... exiting"
	os.makedirs(run_name)
	with open(f'{run_name}/run_params.json', 'wt') as f:
		json.dump(vars(args), f, indent=4)
		vprint(f"Wrote input params to '{run_name}/run_params.json'\n")

	fold_va = args.validation_fold
	folding = np.load(args.folding_cls)
	folding_clsaux = np.load(args.folding_clsaux)
	os.makedirs(f"{run_name}/cls")
	vprint(f"Evaluating cls vs. clsaux performance", model_category=True)
	calculate_single_partner_multi_partner_results(run_name, 'cls', [args.y_cls,args.y_clsaux],[folding,folding_clsaux], \
										fold_va, [args.t8c_cls,args.t8c_clsaux], [args.weights_cls,args.weights_clsaux], \
										Path(args.y_cls_single_partner), Path(args.y_clsaux_multi_partner))
	vprint(f"Run name '{run_name}' is finished.")
	return

if __name__ == '__main__':
	start = time.time()
	args = init_arg_parser()
	vprint('=== WP3 Y3 Performance evaluation script for npy and pred files ===')
	main(args)
	end = time.time()
	vprint(f'Performance evaluation took {end - start:.08} seconds.')

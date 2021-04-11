import os
import time
import argparse
import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np
import torch
import sklearn.metrics
import json
import scipy.sparse as sparse
from scipy.stats import spearmanr
from  pathlib import Path
from scipy.io import mmread
import sparsechem as sc
		
def init_arg_parser():
	parser = argparse.ArgumentParser(description="MELLODDY Year 2 Performance Evaluation De-risk")
	parser.add_argument("--y_cls", help="Classification activity file (npy) (i.e. from files_4_ml/)", type=str, default=None)
	parser.add_argument("--y_clsaux", help="Aux classification activity file (npy) (i.e. from files_4_ml/)", type=str, default=None)
	parser.add_argument("--y_regr", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, default=None)

	parser.add_argument("--y_cls_onpremise", help="Yhat cls prediction output from onpremise run (<single pharma dir>/<cls_prefix>-class.npy)", type=str, default=None)
	parser.add_argument("--y_clsaux_onpremise", help="Yhat clsaux prediction output from onpremise run (<single pharma dir>/<clsaux_prefix>-class.npy)", type=str, default=None)
	parser.add_argument("--y_regr_onpremise", help="Yhat regr prediction output from onpremise run (<single pharma dir>/<regr_prefix>-regr.npy)", type=str, default=None)

	parser.add_argument("--y_cls_substra", help="Pred classification prediction output from substra platform (./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred)", type=str, default=None)
	parser.add_argument("--y_clsaux_substra", help="Pred classification w/ aux prediction output from substra platform", type=str, default=None)
	parser.add_argument("--y_regr_substra", help="Pred regression prediction output from substra platform", type=str, default=None)

	parser.add_argument("--folding_cls", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, default=None)
	parser.add_argument("--folding_clsaux", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, default=None)
	parser.add_argument("--folding_regr", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, default=None)

	parser.add_argument("--t8c_cls", help="T8c file for classification in the results_tmp/classification folder", type=str, default=None)
	parser.add_argument("--t8c_clsaux", help="T8c file for classification w/ auxiliary in the results_tmp/classification folder", type=str, default=None)
	parser.add_argument("--t8r_regr", help="T8r file for regression in the results_tmp/regression folder", type=str, default=None)

	parser.add_argument("--weights_cls", help="CSV file with columns task_id and weight (i.e. T8c.csv)", type=str, default=None)
	parser.add_argument("--weights_clsaux", help="CSV file with columns task_id and weight (i.e. T8c.csv)", type=str, default=None)
	parser.add_argument("--weights_regr", help="CSV file with columns task_id and weight (i.e. T8c.csv)", type=str, default=None)

	parser.add_argument("--perf_json_cls", help="CSV file with columns task_id and weight (i.e. T8c.csv)", type=str, default=None)
	parser.add_argument("--perf_json_clsaux", help="CSV file with columns task_id and weight (i.e. T8c.csv)", type=str, default=None)
	parser.add_argument("--perf_json_regr", help="CSV file with columns task_id and weight (i.e. T8c.csv)", type=str, default=None)

	parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
	parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
	parser.add_argument("--validation_fold", help="Validation fold to used to calculate performance", type=int, default=[0], nargs='+', choices=[0, 1, 2, 3, 4])
	parser.add_argument("--aggr_binning_scheme_perf", help="(Comma separated) Shared aggregated binning scheme for f1/f2 performances", type=str, nargs='+', default=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],required=False)
	parser.add_argument("--aggr_binning_scheme_perf_delta", help="(Comma separated) Shared aggregated binning scheme for delta performances", type=str, nargs='+', default=[-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2],required=False)
	args = parser.parse_args()
	assert len(args.aggr_binning_scheme_perf) == 11, f"len of aggr_binning_scheme_perf should be 11, got {len(args.aggr_binning_scheme_perf)}"
	assert len(args.aggr_binning_scheme_perf_delta) == 9, f"len of aggr_binning_scheme_perf_delta should be 9, got {len(args.aggr_binning_scheme_perf_delta)}"
	args.aggr_binning_scheme_perf=list(map(np.float,args.aggr_binning_scheme_perf))
	args.aggr_binning_scheme_perf_delta=list(map(np.float,args.aggr_binning_scheme_perf_delta))
	return args


def vprint(s="", model_category=False, derisk_check=False):
	separator='='*135
	if args.verbose:
		print()
		if derisk_check or model_category:
			print(separator)
			if model_category: print(separator)
			print(s)
			print(separator)
			if model_category: print(separator)		
		else: print(s)


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
			if lower_infinite:
				new_label = "{0} {1}".format(symbol_lower, bins_final[i+1])
			elif include_lowest:
				new_label = make_label(i, lb="[")
			else:
				new_label = make_label(i)
		elif upper_infinite and i == (num_labels - 1):
			new_label = "{0} {1}".format(symbol_upper, bins_final[i])
		else:
			new_label = make_label(i)
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
	assert args.y_cls or args.y_clsaux or args.y_regr, "Must provide y_cls, y_cls_aux or y_regr"
	if args.y_cls:
		assert args.y_cls_onpremise and args.y_cls_substra and args.t8c_cls and args.weights_cls and args.perf_json_cls, "Must provide y_cls_onpremise, y_cls_substra, t8c_cls, weights_cls & perf_json_cls when supplying y_cls"
		assert Path(args.y_cls_onpremise).suffix == '.npy', "y_cls_onpremise prediction file needs to be '.npy'"
		#assert Path(args.y_cls_substra).stem == 'pred', "y_cls_substra prediction file needs to be 'pred'"
	if args.y_clsaux:
		assert args.y_clsaux_onpremise and args.y_clsaux_substra and args.t8c_clsaux and args.weights_clsaux and args.perf_json_clsaux, "Must provide y_clsaux_onpremise, y_clsaux_substra, t8c_clsaux, weights_clsaux & perf_json_cls when supplying y_clsaux"
		assert Path(args.y_clsaux_onpremise).suffix == '.npy', "y_clsaux_onpremise prediction file needs to be '.npy'"
		#assert Path(args.y_clsaux_substra).stem == 'pred', "y_clsaux_substra prediction file needs to be 'pred'"
	if args.y_regr:
		assert args.y_regr_onpremise and args.y_regr_substra and args.t8r_regr and args.weights_regr and args.perf_json_regr, "Must provide y_regr_onpremise, y_regr_substra, t8r_regr, weights_regr & perf_json_regr when supplying y_regr"
		assert Path(args.y_regr_onpremise).suffix == '.npy', "y_regr_onpremise prediction file needs to be '.npy'"
		#assert Path(args.y_regr_substra).stem == 'pred', "y_regr_substra prediction file needs to be 'pred'"
	return


def validate_ytrue_ypred(ytrue, ypred):
	assert y_true.shape == y_pred.shape, f"y_true shape do not match {pred_or_npy} y_pred ({y_true.shape} & {y_pred.shape})"
	assert y_true.nnz == y_pred.nnz, f"y_true number of nonzero values do not match {pred_or_npy} y_pred"
	assert (y_true.indptr == y_pred.indptr).all(), f"y_true indptr do not match {pred_or_npy} y_pred"
	assert (y_true.indices == y_pred.indices).all(), f"y_true indices do not match {pred_or_npy} y_pred"
	return


def yhat_allclose_check(yhat1,yhat2,f1,f2, tol=1e-05):
	"""
	Check yhats[1/2] are close for the files f[1/2]
	"""
	nnz1, nnz2 = yhat1.nonzero(), yhat2.nonzero()
	allclose = np.allclose(yhat1[nnz1], yhat2[nnz2], rtol=tol, atol=tol)
	spr=f"Spearmanr rank correlation coefficient of the '{f1}' and '{f2}' yhats = {spearmanr(yhat1[nnz1],yhat2[nnz2],axis=1)}"
	if not allclose:
		vprint(f"Phase 2 de-risk check #1: FAILED! yhats NOT close between '{f1}' and '{f2}' (tol:{tol})\n{spr}",derisk_check=True)
		return False
	else:
		vprint(f"Phase 2 de-risk check #1: PASSED! yhats close between '{f1}' and '{f2}' ({tol})\n{spr}",derisk_check=True)
		return True


def global_allclose_check(globally_calculated,perf_agg, tol=1e-05):
	"""
	Check globally aggregated performances are close between those calculated & reported
	"""
	perf_agg=pd.concat([pd.DataFrame([i[1]], columns=[i[0]]) for i in perf_agg.items()],axis=1)
	allclose = np.allclose(globally_calculated, perf_agg[globally_calculated.columns.tolist()], rtol=tol, atol=tol)
	if not allclose: 
		vprint(f"Phase 2 de-risk check #3: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:{tol}) \
							\n{globally_calculated}\n{perf_agg[globally_calculated.columns.tolist()]}",derisk_check=True)
		return False
	else:
		vprint(f"Phase 2 de-risk check #3: PASSED! global reported performance metrics and global calculated performance metrics close (tol:{tol})",derisk_check=True)
		return True


def delta_assay_allclose_check(deltas, tol=1e-05):
	"""
	Check deltas are close to zero across all calculated ∆performances
	"""
	allclose = np.isclose(deltas, 0, rtol=tol, atol=tol).all()
	if not allclose:
		vprint(f"Phase 2 de-risk check #4: FAILED! delta between local & substra assay_type aggregated performances NOT close to 0 across all metrics (tol:{tol})",derisk_check=True)
		return False
	else:
		vprint(f"Phase 2 de-risk check #4: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:{tol})",derisk_check=True)
		return True


def delta_global_allclose_check(deltas, tol=1e-05):
	"""
	Check deltas are close to zero across all calculated ∆performances
	"""
	allclose = np.isclose(deltas, 0,  rtol=tol, atol=tol).all()
	if not allclose:
		vprint(f"Phase 2 de-risk check #5: FAILED! delta performance between global local & global substra performances NOT close to 0 across all metrics (tol:{tol})",derisk_check=True)
		return False
	else:
		vprint(f"Phase 2 de-risk check #5: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:{tol})",derisk_check=True)
		return True
	

def load_yhats(input_f, folding, fold_va, y_true):
	"""
	Load and mask yhats for those considered for evaluation 
	"""
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
	try: yhats = yhats[[i in fold_va for i in folding]]
	except IndexError: pass
	return yhats, ftype


def mask_y_hat(f1_path, f2_path, folding, fold_va, y_true, header_type):
	"""
	Load yhats using load_yhats and validate the shapes
	"""
	if header_type == 'classification': true_data = y_true.astype(np.uint8).todense()
	else: true_data = y_true.astype('float64').todense()
	f1_yhat, f1_ftype = load_yhats(f1_path, folding, fold_va, y_true)
	assert true_data.shape == f1_yhat.shape, f"True shape {true_data.shape} and {f1_path.stem} shape {f1_yhat.shape} need to be identical"
	f2_yhat, f2_ftype = load_yhats(f2_path, folding, fold_va, y_true)
	assert true_data.shape == f2_yhat.shape, f"True shape {true_data.shape} and {f2_path.stem} shape {f2_yhat.shape} need to be identical"
	return [f1_yhat, f2_yhat, f1_ftype, f2_ftype]


def mask_ytrue(ftype,fname,folding,fold_va):
	"""
	Mask the y-input labels for the validation fold
	"""
	vprint(f'Loading {ftype}: {fname}')
	try: y_all = sc.load_sparse(fname)
	except AttributeError: 
		y_all = mmread(fname)
	y_all = y_all.tocsc()
	y_all_true = y_all[[i in fold_va for i in folding]]
	return y_all_true


def check_weights(tw_df, y_true, header_type):
	"""
	Validate task weight file
	"""
	if header_type == 'classification': tw_dfshape = 4
	else: tw_dfshape = 5
	assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
	assert tw_df.shape[1] == tw_dfshape, "task weight file (CSV) must only have 4 columns"
	assert "training_weight" in tw_df.columns, "weight is missing in task weights CVS file"
	assert y_true.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
	assert (0 <= tw_df.training_weight).all(), "task weights must not be negative"
	assert (tw_df.training_weight <= 1).all(), "task weights must not be larger than 1.0"
	assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
	assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
	assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
	assert tw_df.shape[0]==y_true.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({y_true.shape[1]})."
	return


def substra_global_perf_from_json(performance_report,agg,required_headers):
	"""
	Validate reported performance metrics
	"""
	agg = getheader(agg)
	with open(performance_report, "r") as fid:
		jsons = json.load(fid)
	returned_performances = []
	for mode in [f'{agg}', f'{agg}_agg']:
		json_data = json.loads(jsons['validation'][mode])
		returned_performances.append(json_data)
		assert all([i in json_data.keys() for i in required_headers]), f"not all expected headers are in the {agg} performance report"
		assert len(json_data.keys()) >= len(required_headers), f"expected a minimum {len(required_headers)} reported {agg} metrics in the performance report, got {len(json_data.keys())}"
	return returned_performances


def calculate_delta(f1_results, f2_results, run_name, run_type, sc_columns):
	"""
	Calculate the delta between the outputs and write to a file
	"""
	derisk_checks = []
	header_type = getheader(run_type)
	for idx, delta_comparison in enumerate(['locals',f'deltas_global_performances.csv']):
		assert f1_results.shape[0] == f2_results.shape[0], "the number of tasks are not equal between the --f1 and --f2 outputs for {delta_comparison}"
		assert f1_results.shape[1] == f2_results.shape[1], "the number of reported metrics are not equal between the --f1 and --f2 outputs for {delta_comparison}"
		# add assay aggregation if local
		if(delta_comparison == 'locals'):
			task_id = f2_results[f"{header_type}_task_id"]
			at = f2_results["assay_type"]
			delta = (f2_results.loc[:, sc_columns[0]:sc_columns[-1]]-f1_results.loc[:, sc_columns[0]:sc_columns[-1]])
			tdf = pd.concat([task_id, at, delta], axis = 1)
			os.makedirs(f"{run_name}/{run_type}/deltas/")
			fn1 = f"{run_name}/{run_type}/deltas/deltas_per-task_performances.csv"
			pertask = tdf.copy()
			pertask.loc[:,f'{header_type}_task_id'] = pertask[f'{header_type}_task_id'].astype('int32')
			#add per-task perf aggregated performance delta bins to output
			for metric in pertask.loc[:, sc_columns[0]:sc_columns[-1]].columns:
				pertask.loc[:,f'{metric}_percent'] = cut(pertask[metric].astype('float32'), \
				args.aggr_binning_scheme_perf_delta,include_lowest=True,right=True)
			#write per-task perf aggregated performance delta
			pertask.to_csv(fn1, index= False)
			derisk_checks.append(delta_assay_allclose_check(pertask[sc_columns]))
			vprint(f"Wrote per-task delta report to: {fn1}")
			
			#write binned per-task aggregated performance deltas
			agg_deltas=[]
			for metric_bin in pertask.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
				agg_perf=(pertask.groupby(metric_bin)[f'{header_type}_task_id'].agg('count')/len(pertask)).reset_index().rename(columns={f'{header_type}_task_id': f'bin_{metric_bin}'})
				agg_deltas.append(agg_perf.set_index(metric_bin))
			fnagg = f"{run_name}/{run_type}/deltas/deltas_binned_per-task_performances.csv"
			pd.concat(agg_deltas,axis=1).astype(np.float32).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
			vprint(f"Wrote binned performance per-task delta report to: {fnagg}")

			# aggregate on assay_type level
			fn2 = f"{run_name}/{run_type}/deltas/deltas_per-assay_performances.csv"
			tdf[['assay_type'] + sc_columns].groupby("assay_type").mean().to_csv(fn2)
			vprint(f"Wrote per-assay delta report to: {fn2}")

			#write binned per-assay aggregated performance deltas
			agg_deltas2=[]
			for metric_bin in pertask.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
				agg_perf2=(pertask.groupby(['assay_type',metric_bin])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{metric_bin}'})
				agg_perf2[f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (pertask.assay_type==x['assay_type']).sum() ,axis=1).astype('float32')
				agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
				agg_deltas2.append(agg_perf2.set_index(['assay_type',metric_bin]))
			fnagg2 = f"{run_name}/{run_type}/deltas/deltas_binned_per-assay_performances.csv"	
			pd.concat(agg_deltas2,axis=1).astype(np.float32).reset_index().rename(columns={f'{sc_columns[0]}_percent':'perf_bin',}).to_csv(fnagg2,index=False)
			vprint(f"Wrote binned performance per-assay delta report to: {fnagg}")
		else:
			global_delta = pd.DataFrame(f2_results[sc_columns].mean(axis=0)).T - pd.DataFrame(f1_results[sc_columns].mean(axis=0)).T
			derisk_checks.append(delta_global_allclose_check(global_delta))
			global_delta.to_csv(f"{run_name}/{run_type}/deltas/{delta_comparison}", index=False)
	return derisk_checks


def calculate_onpremise_substra_results(run_name, run_type, y_true, folding, fold_va, t8, task_weights, onpremise, substra, performance_report):
	"""
	Calculate cls, clsaux or regr performances for onpremise and substra outputs, then calculate delta, plot outputs along the way
	"""
	derisk_checks = []
	header_type = getheader(run_type)
	y_true = mask_ytrue(run_type,y_true,folding,fold_va)
	tw_df = pd.read_csv(task_weights)
	tw_df.sort_values("task_id", inplace=True)
	check_weights(tw_df,y_true,header_type)
	t8 = pd.read_csv(t8) #read t8c or t8r files
	if run_type == 'regr': t8=t8.reset_index().rename(columns={'index': 'regression_task_id'})
	task_map = t8.merge(tw_df,left_on=f'cont_{header_type}_task_id',right_on='task_id',how='left').query('task_id.notna()')
	y_onpremise_yhat, y_substra_yhat, y_onpremise_ftype, y_substra_ftype = mask_y_hat(onpremise, substra, folding, fold_va, y_true, header_type)
	derisk_checks.append(yhat_allclose_check(y_onpremise_yhat,y_substra_yhat,onpremise.stem,substra.stem)) #add derisk #1
	y_onpremise_results, _ = run_performance_calculation(run_type, y_onpremise_yhat, y_onpremise_ftype, y_true, tw_df, task_map, run_name, onpremise)
	del y_onpremise_yhat
	y_substra_results, sc_columns, derisk_reported = run_performance_calculation(run_type, y_substra_yhat, y_substra_ftype, y_true, tw_df, task_map, run_name, substra, perf_report=performance_report)
	del y_substra_yhat
	derisk_checks += derisk_reported + calculate_delta(y_onpremise_results, y_substra_results, run_name, run_type, sc_columns) #add derisk #2/3 & 4/5
	return derisk_checks


#write performance reports for global aggregation
def write_global_report(run_name, run_type, fname, local_performances, sc_columns):
	df = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0').copy()
	df = pd.DataFrame(df[sc_columns].mean(axis=0)).T
	fn1 = f"{run_name}/{run_type}/{fname}/{fname}_global_performances.csv"
	df.to_csv(fn1, index= False)
	vprint(f"Wrote global report to: {fn1}")
	return df


#write performance reports per-task & per-task_assay
def write_aggregated_report(run_name, run_type, fname, local_performances, sc_columns):
	df = local_performances.copy()
	for metric in df.loc[:, sc_columns[0]:sc_columns[-1]].columns:
		df.loc[:,f'{metric}_percent'] = cut(df[metric].astype('float32'), \
		args.aggr_binning_scheme_perf,include_lowest=True,right=True,lower_infinite=False, upper_infinite=False)
	df.loc[:,f'{header_type}_task_id'] = df[f'{header_type}_task_id'].astype('float').astype('int32')
	os.makedirs(f"{run_name}/{run_type}/{fname}/")
	fn1 = f"{run_name}/{run_type}/{fname}/{fname}_per-task_performances.csv"
	df.to_csv(fn1, index=False)
	vprint(f"Wrote per-task report to: {fn1}")


	#write binned per-task performances
	agg_concat=[]
	for metric_bin in df.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf=(df.groupby(metric_bin)[f'{header_type}_task_id'].agg('count')/len(df)).reset_index().rename(columns={f'{header_type}_task_id': f'bin_{metric_bin}'})
		agg_concat.append(agg_perf.set_index(metric_bin))
	fnagg = f"{run_name}/{run_type}/{fname}/{fname}_binned_per-task_performances.csv"
	pd.concat(agg_concat,axis=1).astype(np.float32).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
	vprint(f"Wrote per-task binned performance report to: {fnagg}")

	#write performance aggregated performances by assay_type
	df2 = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0').copy()[['assay_type'] + sc_columns]
	df2 = df2.loc[:,'assay_type':].groupby('assay_type').mean()
	fn2 = f"{run_name}/{run_type}/{fname}/{fname}_per-assay_performances.csv"
	df2.to_csv(fn2)
	vprint(f"Wrote per-assay report to: {fn2}")

	#write binned per-task perf performances by assay_type 
	agg_concat2=[]
	for metric_bin in df.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf2=(df.groupby(['assay_type',metric_bin])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{metric_bin}'})
		agg_perf2.loc[:,f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (df.assay_type==x['assay_type']).sum() ,axis=1).astype('float32')
		agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
		agg_concat2.append(agg_perf2.set_index(['assay_type',metric_bin]))
	fnagg2 = f"{run_name}/{run_type}/{fname}/{fname}_binned_per-assay_performances.csv"
	pd.concat(agg_concat2,axis=1).astype(np.float32).reset_index().rename(columns={f'{sc_columns[0]}_percent':'perf_bin',}).to_csv(fnagg2,index=False)
	vprint(f"Wrote per-assay binned report to: {fnagg}")
	return


def run_performance_calculation(run_type, y_pred, pred_or_npy, y_true, tw_df, task_map, run_name, flabel, perf_report=None):
	"""
	Calculate performance for one run, bin results and then individual performance reports including aggregation by assay/globally
	"""
	vprint(f"=== Calculating {flabel} performance ===")
	header_type = getheader(run_type)
	if header_type == 'classification':
		sc_columns = sc.utils.all_metrics([0],[0]).columns.tolist()  #get the names of reported metrics from the sc utils
	else:
		sc_columns = sc.utils.all_metrics_regr([0],[0]).columns.tolist()  #get the names of reported metrics from the sc utils
	if pred_or_npy == 'npy': y_pred = sparse.csc_matrix(y_pred)
	## checks to make sure y_true and y_pred match
	validate_ytrue_ypred(y_true, ypred)
	if perf_report:
		perf_columns = [f'{header_type}_task_id',f'cont_{header_type}_task_id']+sc_columns
		perf, perf_agg = substra_global_perf_from_json(perf_report, run_type, sc_columns)
		sc_reported = pd.concat([pd.DataFrame(i[1].items(), \
			columns=[f'cont_{header_type}_task_id',i[0]]).set_index(f'cont_{header_type}_task_id') for i in perf.items()],axis=1).reset_index()
		delta_reported = pd.DataFrame(columns=perf_columns)
	calculated_performance = pd.DataFrame()
	for col_idx, col in enumerate(range(y_true.shape[1])):
		task_id = task_map[f"{header_type}_task_id"][task_map[f"cont_{header_type}_task_id"]==col].iloc[0]
		if header_type == 'classification': #setup for classification metrics
			y_true_col = y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1
			y_pred_col = y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]]
			y_true_col, y_pred_col = y_true_col.astype(np.uint8), y_pred_col.astype('float32')
			y_classes   = np.where(y_pred_col > 0.5, 1, 0).astype(np.uint8)
			tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()
			details = pd.DataFrame(
					{f'{header_type}_task_id': pd.Series(task_id, dtype='int32'),
					'task_size': pd.Series(len(y_true_col), dtype='int32'),
					'tn': pd.Series(tn, dtype='int32'),
					'fp': pd.Series(fp, dtype='int32'),
					'fn': pd.Series(fn, dtype='int32'),
					'tp': pd.Series(tp, dtype='int32')})
			sc_calculation = sc.utils.all_metrics(y_true_col,y_pred_col)
		else: #setup for regression metrics
			y_true_col = y_true.data[y_true.indptr[col] : y_true.indptr[col+1]]
			y_pred_col = y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]]
			details = pd.DataFrame(
					{f'{header_type}_task_id': pd.Series(task_id, dtype='int32'),
					'task_size': pd.Series(len(y_true_col), dtype='int32')})
			sc_calculation = sc.utils.all_metrics_regr(y_true_col,y_pred_col)
		if y_true_col.shape[0] <= 1: continue
		if (y_true_col[0] == y_true_col).all(): continue
		sc_metrics = pd.concat([details,sc_calculation],axis=1)
		#de-risk: track when individual task metrics differ between calculated and reported
		if perf_report:
			this_task = sc_reported[sc_reported[f"cont_{header_type}_task_id"]==str(col_idx)][sc_columns].values
			this_task = pd.DataFrame([[task_id, col_idx] + np.isclose(this_task,sc_calculation, rtol=1e-05, atol=1e-05).tolist()[0]], columns = perf_columns)
			delta_reported = pd.concat((delta_reported,this_task),axis=0)			
		calculated_performance = pd.concat([calculated_performance, sc_metrics],axis=0)	
	#merge calculated performances with the details of the tasks
	calculated_performance = calculated_performance.merge(task_map, left_on=f'{header_type}_task_id', right_on=f'{header_type}_task_id',how='left')
	##write per-task & per-assay_type performance:
	write_aggregated_report(run_name, run_type, flabel, calculated_performance, sc_columns)
	if perf_report:
		if delta_reported[sc_columns].all().all():
			vprint(f'Phase 2 de-risk check #2: PASSED! calculated and reported metrics are the same across individual tasks \
			\n{delta_reported[sc_columns].all()[delta_reported[sc_columns].all()].index.tolist()} are identical',derisk_check=True)
			check2 = True
		else:
			vprint(f'Phase 2 de-risk check #2: FAILED! calculated metrics for one or more individual tasks differ to reported performances (tol:1e-05) \
			\n{delta_reported[sc_columns].all()[~delta_reported[sc_columns].all()].index.tolist()} are the reported metrics with different performances \
			\nCheck the output of {run_name}/{run_type}/{flabel}/{flabel}_closeto_reported_performances.csv for details',derisk_check=True)
			check2 = False
		delta_reported.to_csv(f"{run_name}/{run_type}/{flabel}/{flabel}_closeto_reported_performances.csv",index=False)
		vprint(f"Wrote reported vs. calculated performance delta to: {run_name}/{run_type}/{flabel}_delta-reported_performances.csv")
	##global aggregation & derisk if necessary:
	globally_calculated = write_global_report(run_name, run_type, flabel, calculated_performance, sc_columns)
	if perf_report: return calculated_performance, sc_columns, [check2, global_allclose_check(globally_calculated, perf_agg)]
	else: return calculated_performance, sc_columns


def main(args):
	vprint(args)
	validate_cls_clsuax_regr_inputs(args)

	if args.filename is not None:
		run_name = args.filename
	else:
		timestr = time.strftime('%Y%m%d-%H%M%S')
		run_name = f"perf_derisk_{timestr}"
		vprint(f"\nRun name is '{run_name}'")
	assert not os.path.exists(run_name), f"{run_name} already exists... exiting"
	os.makedirs(run_name)
	with open(f'{run_name}/run_params.json', 'wt') as f:
		json.dump(vars(args), f, indent=4)
		vprint(f"Wrote input params to '{run_name}/run_params.json'\n")

	fold_va = args.validation_fold
	derisk_df = pd.DataFrame(columns=['run_type','check#1','check#2','check#3','check#4','check#5'])
	if args.y_cls:
		folding = np.load(args.folding_cls)
		os.makedirs(f"{run_name}/cls")
		vprint(f"De-risking cls performance", model_category=True)
		derisked = \
			calculate_onpremise_substra_results(run_name, 'cls' ,args.y_cls,folding, \
												fold_va, args.t8c_cls, args.weights_cls, \
												Path(args.y_cls_onpremise),Path(args.y_cls_substra), \
												args.perf_json_cls)
		derisk_df = pd.concat((derisk_df, pd.DataFrame([['cls']+derisked], columns = derisk_df.columns)))
	if args.y_clsaux:
		folding = np.load(args.folding_clsaux)
		os.makedirs(f"{run_name}/clsaux")
		vprint(f"De-risking clsaux performance", model_category=True)
		derisked = \
			calculate_onpremise_substra_results(run_name, 'clsaux' ,args.y_clsaux, \
												folding,fold_va, args.t8c_clsaux, args.weights_clsaux, \
												Path(args.y_clsaux_onpremise),Path(args.y_clsaux_substra), \
												args.perf_json_clsaux)
		derisk_df = pd.concat((derisk_df, pd.DataFrame([['clsaux']+derisked], columns = derisk_df.columns)))
	if args.y_regr:
		folding = np.load(args.folding_regr)
		os.makedirs(f"{run_name}/regr")
		vprint(f"De-risking regr performance", model_category=True)
		derisked = \
			calculate_onpremise_substra_results(run_name, 'regr' ,args.y_regr, \
												folding, fold_va, args.t8r_regr, args.weights_regr, \
												Path(args.y_regr_onpremise),Path(args.y_regr_substra), \
												args.perf_json_regr)
		derisk_df = pd.concat((derisk_df, pd.DataFrame([['regr']+derisked], columns = derisk_df.columns)))
	derisk_df.to_csv(f'{run_name}/derisk_summary.csv',index=False)
	vprint(f"Run name '{run_name}' is finished.")
	return

if __name__ == '__main__':
	start = time.time()
	args = init_arg_parser()
	vprint('=== WP3 Y2 Performance evaluation de-risk script for npy and pred files ===')
	main(args)
	end = time.time()
	vprint(f'Performance evaluation took {end - start:.08} seconds.')
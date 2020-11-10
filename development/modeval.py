import os, sys
import numpy as np
import pandas as pd 
import scipy.sparse
import scipy.stats
import sklearn.metrics
import re
import json 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


sns.set_style("whitegrid")


# function template
# def template():
#    """ description
#     :param dtype name: description
#     :param dtype name: description
#     :return dtype: description
#    """
#    return



def perf_from_json(
        model_dir_or_file, 
        tasks_for_eval=None, 
        aggregate=False, 
        evaluation_set='va',
        model_name='Y',
        fold_va=[0,1,2,3,4],
        filename_mask=None,
        verbose=False
    ):
    """ Collects the performance from thje models/*.json files containing both the model configuration and performance.
      Useful for HP search because it includes HPs details.
#     :param string model_dir_or_file: path to the model folder containing the .json files
#     :param np.array (of indices/integers) tasks_for_eval: tasks to consider for evaluation (default=None)
#     :param bool aggrgate: if True, uses the aggregate result from sparsechem (considering all tasks verifying MIN_SAMPLES).
#     :param string evaluation_set: keyword for result extraction from json file. 
#     :param string model_name: adds a name in a column to resulting dataframe (default=Y)
#     :param list fold_va: validation folds to look for, will only return specifed folds 
#     :param str filename_mask: specify a filename mask to restrict the perf loading to a subset of files
#     :return pandas df containing performance and configuration summaries 
    """
    # pandas v0.24 (pd.read_json) is not returning expected results when used with the arguement "tasks_for_eval" 
    assert pd.__version__ > '0.25', 'Pandas version must be 0.25 or higher' # string comparison will compare first char # should this be placed somewhere else?
    
    for fva in fold_va:
        assert type(fva) is int, f"{fold_va}: contains none integers and should not!"
    
    if tasks_for_eval is not None: 
        assert type(tasks_for_eval) is np.ndarray, "tasks_for_eval must be an np.array"
        assert tasks_for_eval.ndim == 1, "tasks_for_eval must be np.array with ndim=1"
    
        for x in tasks_for_eval:
            assert int(x) == x, "elements in tasks_for_eval must be integer-like"
            assert x>=0, "elements in tasks_for_eval must be > 0"
            
        assert np.unique(tasks_for_eval).shape[0] == tasks_for_eval.shape[0], "tasks_for_eval must not have duplicates"
        
    assert type(fold_va) is list or type(fold_va) is tuple, f"{fold_va} : needs to be a list or a tuple"
    
    res_all = []

    if os.path.isdir(os.path.join(model_dir_or_file)):     
        files = [os.path.join(model_dir_or_file,f) for f in os.listdir(model_dir_or_file) if os.path.isfile(os.path.join(model_dir_or_file,f))]
    elif os.path.isfile(os.path.join(model_dir_or_file)):     
        files = [model_dir_or_file]
    
    for f in tqdm(files):
        if not f.endswith(".json"):# or not os.path.basename(f).startswith("sc_"):
            if verbose: print(f"{f} is not a sparsechem json, hence skipped.")
            continue
        
        if filename_mask is not None and filename_mask not in f:
            if verbose: print(f"{f} does not match filename mask, hence skipped.")
            continue
            
        with open(f, "r") as fi:
            data = json.load(fi)

        if aggregate: 
            assert "results_agg" in data, f"Error: cannot find 'results_agg' in performance summary {f}"
            assert evaluation_set in data["results_agg"], f"Error: cannot find '{evaluation_set}' in data results_agg of {f}"
            assert tasks_for_eval is None, 'tasks_for_eval has no effect in combination with aggregate=True'

            res_df = pd.read_json(data["results_agg"][evaluation_set], typ="series").to_frame().transpose()
            res_df.columns = [x+'_agg' for x in res_df.columns]
        
        else: 
            assert "results" in data, f"Error: cannot find 'results' in {f}"
            assert evaluation_set in data["results"], f"Error: cannot find '{evaluation_set}' in data results"
            
            res_df = pd.read_json(data["results"][evaluation_set])
            
            # mask out some tasks
            if tasks_for_eval is not None:
                res_df = res_df.iloc[tasks_for_eval]
            
            # create task column
            res_df = res_df.reset_index().rename(columns={'index':'task'})

        # add config/hp to the dataframe
        for k,v in data["conf"].items():
            if type(v) == list: v=",".join([str(x) for x in v])
            if k in get_sc_hps() : k = 'hp_' + k
            res_df[k]=v
       
        # add filename to the dataframe
        res_df['filename'] = str(f)
        
        res_all.append(res_df)
 
    assert len(res_all) > 0, "No .json files found or none of them matched the filename_mask"
    
    output_df = pd.concat(res_all)

    # filter out some folds 
    output_df = output_df.query('fold_va in @fold_va').copy()
    
    assert output_df.shape[0] > 0, f"No records found for the specified cv fold {fold_va}"
    
    # TO DO: print a warning when a fold specified by fold_va is not found
    
    
    output_df['model_name'] = model_name
    
    # HP wihtout set value cannot be None otherwise they get dropped in quorum filter as hp columns get set as index and used to mask out
    # only HPs set to None by default in sparsechem can be in this situation
    values = {'hp_task_weights':'noTaskWeights',
             'internal_batch_max':'None',
             'fold_te':'None',
             'input_size_freq':'None',
             'fold_inputs':'None',
             'filename':'None'}
    
    # should be asserted here
    
    return output_df.fillna(value=values)



def verify_cv_runs(metrics_df, fold_va=[0,1,2,3,4]):
    """ From the metrics dataframe yielded by perf_from_json(), cehcks if each hyperparameter was run fold_va
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param list fold_va: validation folds to look for
#     :return void: prints message in stdout
    """    
    assert np.unique(fold_va).shape[0] == len(fold_va), f"{fold_va}: contains duplicates and should not!"
    for fva in fold_va:
        assert type(fva) is int, f"{fold_va}: contains none integers and should not!"
    
    fold_va.sort()
    
    hp = [x for x in metrics_df.columns if x[:3] == 'hp_' and not metrics_df[x].isnull().all()]
    hp.append('model_name')
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_ and model_name"
    
    aggr = metrics_df.dropna(axis='columns', how='all').sort_values('fold_va').groupby(hp)['fold_va'].apply(lambda x: ','.join([str(x) for x in x.unique()]))
    
     # if missing ones, print out a warning message
    valid=True
    warning=False
    folds = ",".join([str(x) for x in fold_va])
    
    aggr = aggr.reset_index()
    aggr['hp_string'] = make_hp_string_col(aggr)
    
    # loop over the rows and find out if missing fold_va
    for idx,row in aggr[['hp_string', 'fold_va']].iterrows():
        
        # missing fold_va
        if row['fold_va'] != folds and row['fold_va'] in folds:
            print(f"HP string: {row['hp_string']:<25} expected fold_va: {fold_va} , found: {row['fold_va']}")
            valid = False
            
        # validation folds found not in fold_va
        elif row['fold_va'] != folds:warning=True

    if warning:
        print(f"# WARNING: Validation folds found not in specified fold_va: {fold_va}")
        print(f"# WARNING: Found: {aggr['fold_va'].unique()}")
        
    return valid



def quorum_filter(metrics_df, min_samples=5, fold_va=[0,1,2,3,4], verbose=True):
    """ Filter the metrics data frame of each model (as defined by a HP set) with a quorum rule: minimum N postive samples and N negative sample in each fold 
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int min_samples: minimum class size per fold
#     :param list fold_va: validation folds to look for, will filter out results in metrics_df if belonging to fold_va not specified
#     :param bool verbose: adds verbosity
#     :return pandas df filtered_df: metrics data frame containing , for every given HPs sets, only tasks present in each of the folds 
    """
    if verbose: 
        print(f"# Quorum on class size applied: {min_samples}-{min_samples}")
        print(f"# validation folds considered: {fold_va}")
    
    assert verify_cv_runs(metrics_df, fold_va=fold_va), "Missing cv run, abort."
    assert 'fold_va' in metrics_df.columns, "fold_va must be present in metrics data frame"
    assert 'task' in metrics_df.columns, "task must be present in metrics data frame"
    assert 'num_pos' in metrics_df.columns, "num_pos must be present in metrics data frame"
    assert 'num_neg' in metrics_df.columns, "num_neg must be present in metrics data frame"
    
    if metrics_df['fold_va'].unique().shape[0] > len(fold_va):
        print(f"# WARNING: Results for validation folds not in {fold_va} will be filtered out at this step")
    
    index_cols = ['fold_va', 'task', 'model_name'] + [col for col in metrics_df if col[:3] == 'hp_' if not metrics_df[col].isna().all()] 
    df = metrics_df.loc[metrics_df['fold_va'].isin(fold_va)].set_index(keys=index_cols, verify_integrity=True).copy()
    
    # add a dummy column to perform the count
    df['_'] = 1
    
    # first filter rows with at least N positives and N negatives in each fold 
    task_mini = df.loc[(df['num_pos']>=min_samples)&(df['num_neg']>=min_samples)]
    
    # then count the number of folds for each <task,hp> and filter (it must be present in each fold)
    levels = [x for x in range(1, len(index_cols))]
    task_count = task_mini['_'].groupby(level=levels).count()
    selected_tasks = task_count[task_count==len(fold_va)].index.unique()
    
    filtered_res = df.reset_index(level=0).loc[selected_tasks, :].reset_index().drop('_',axis=1)
    
    if verbose: 
        print(f"# --> Total number of tasks : {metrics_df.task.unique().shape[0]}")
        print(f"# --> Tasks considerd       : {filtered_res.task.unique().shape[0]}")
        print(f"# --> Valid folds considerd : {filtered_res.fold_va.unique()}")
    
    return filtered_res

    





# =======================================================================
# =======================================================================
# ==== aggregation functions: apply only to individual task performances 
# ==== does not apply to sparsechem aggregate performance metrics

def aggregate_fold_perf(metrics_df, min_samples=5, fold_va=[0,1,2,3,4], stats='basic', score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ HP performance aggregation over folds. Includes a quorum filtering step (at least N actives and N inactives in each of the validation folds )
    From the metrics dataframe yielded by perf_from_metrics(), does the aggregation over the fold (mean, median, std, skewness, kurtosis) results in one perf per fold.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_metrics() 
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param list fold_va: validation folds to look for
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :param string stats: ['basic', 'full'], if full, calculates skewness of kurtosis
#     :param list score_type: ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'] thelist of metrics to aggregate
#     :param bool verbose 
#     :return dtype: pandas df containing performance per task aggregated over each fold
    """    

    hp = [x for x in metrics_df.columns if x[:3] == 'hp_' and not metrics_df[x].isna().all()]
    
    assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    assert 'fold_va' in metrics_df.columns, "metrics dataframe must contain fold_va column"
    
    hp.append('fold_va')
    hp.append('model_name')
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_, fold_va and model_name"
    

    # keep only tasks verifying the min_sample rule: at least N postives and N negatives in each of the 5 folds
    metrics2consider_df = quorum_filter(metrics_df, min_samples=min_samples, fold_va=fold_va, verbose=verbose)
    cols2drop = [col for col in metrics_df.columns if col not in score_type and col not in hp]
    
    if verbose: print("# Aggregatate (performance mean) hyperparameter combinations")
    
    ### Scores must be numeric types!!!
    # should assert that scores are numeric types
    
    # do the mean aggregation
    aggr_mean = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).mean()
    aggr_mean.columns = [x+'_mean' for x in aggr_mean.columns]
    
    # do the stdev aggregation
    aggr_std = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).std()
    aggr_std.columns = [x+'_stdev' for x in aggr_std.columns]

    # do the num_pos, num_neg aggregation
    aggr_num = metrics2consider_df[['num_pos','num_neg']+hp].groupby(hp).sum()  
    
    if stats=='basic':
        results = aggr_num.join(aggr_mean).join(aggr_std)
    
    
    elif stats=='full':
        # do the median aggregation
        aggr_med = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).median()
        aggr_med.columns = [x+'_median' for x in aggr_med.columns]
    
        # do the skew aggregation
        aggr_skew = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).skew()
        aggr_skew.columns = [x+'_skewness' for x in aggr_skew.columns]

        # do the kurtosis aggregation
        aggr_kurt = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).apply(pd.DataFrame.kurt)
        aggr_kurt.columns = [x+'_kurtosis' for x in aggr_kurt.columns]
    
        results = aggr_mean.join(aggr_med).join(aggr_std).join(aggr_skew).join(aggr_kurt)
        
    if verbose: print(f"# --> Found {results.shape[0]} fold performance records")
    return results.reset_index()




def aggregate_task_perf(metrics_df, min_samples=5, fold_va=[0,1,2,3,4], stats='basic', score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ HP performance aggregation over tasks. Includes a quorum filtering step (at least N actives and N inactives in each of the validation folds)
    From the metrics dataframe yielded by perf_from_json(), does the aggregation over the CV (mean, std) results in one perf per task.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param list fold_va: validatio folds to look for
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :param string stats: ['basic', 'full'], if full, calculates skewness of kurtosis
#     :param list score_type: ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'] thelist of metrics to aggregate
#     :param bool verbose
#     :return dtype: pandas df containing performance per task aggregated over CV
    """    

    hp = [x for x in metrics_df.columns if x[:3] == 'hp_' and not metrics_df[x].isna().all()]
    
    assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    
    hp.append('task')
    hp.append('model_name')
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_, task and model_name"
    
    # keep only tasks verifying the min_sample rule: at least N postives and N negatives in each of the 5 folds
    metrics2consider_df = quorum_filter(metrics_df, min_samples=min_samples, fold_va=fold_va, verbose=verbose)
    col2drop = [col for col in metrics_df.columns if col not in score_type and col not in hp]
    
    if verbose: print("# Aggregatate (performance mean) hyperparameter combinations")
    
    ### Scores must be numeric types!!!
    # should assert that scores are numeric types
    
    # do the mean aggregation
    aggr_mean = metrics2consider_df.drop(col2drop,axis=1).groupby(hp).mean()
    aggr_mean.columns = [x+'_mean' for x in aggr_mean.columns]
    
    # do the stdev aggregation
    aggr_std = metrics2consider_df.drop(col2drop,axis=1).groupby(hp).std()
    aggr_std.columns = [x+'_stdev' for x in aggr_std.columns]
    
    # do the num_pos, num_neg aggregation
    aggr_num = metrics2consider_df[['num_pos','num_neg']+hp].groupby(hp).sum()
    
    if stats == 'basic':
        results = aggr_num.join(aggr_mean).join(aggr_std)
        
    
    elif stats=='full':
        # do the median aggregation
        aggr_med = metrics2consider_df.drop(col2drop,axis=1).groupby(hp).median()
        aggr_med.columns = [x+'_median' for x in aggr_med.columns]        
        
        # do the skew aggregation
        aggr_skew = metrics2consider_df.drop(col2drop,axis=1).groupby(hp).skew()
        aggr_skew.columns = [x+'_skewness' for x in aggr_skew.columns]
        
        # do the kurtosis aggregation
        aggr_kurt = metrics2consider_df.drop(col2drop,axis=1).groupby(hp).apply(pd.DataFrame.kurt)
        aggr_kurt.columns = [x+'_kurtosis' for x in aggr_kurt.columns]
        
        results = aggr_num.join(aggr_mean).join(aggr_med).join(aggr_std).join(aggr_skew).join(aggr_kurt)

    if verbose: print(f"# --> Found {results.shape[0]} task performance records")
    return results.reset_index()



    

def aggregate_overall(metrics_df, min_samples=5, stats='basic', fold_va=[0,1,2,3,4], score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ HP performance aggregation overall . Includes a quorum filtering step (at least N actives and N inactives in each of the validation folds)
    From the metrics dataframe yielded by perf_from_json(), does the aggregation over the CV (mean, std) results in one perf per hyperparameter.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param list fold_va: validation to look for 
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :param string stats: ['basic', 'full'], if full, calculates skewness of kurtosis
#     :param list score_type: ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'] thelist of metrics to aggregate
#     :param bool verbose
#     :return dtype: pandas df containing performance per hyperparameter setting
    """ 
        
    hp = [x for x in metrics_df.columns if x[:3] == 'hp_' and not metrics_df[x].isna().all()]
    
    #assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    #assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    
    hp.append('model_name')
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_ and model_name"
    
    
    # keep only tasks verifying the min_sample rule: at least N postives and N negatives in each of the 5 folds
    metrics2consider_df = quorum_filter(metrics_df, min_samples=min_samples, fold_va=fold_va, verbose=verbose)
    cols2drop = [col for col in metrics_df.columns if col not in score_type and col not in hp]
    
    if verbose: print("# Aggregatate (performance mean) hyperparameter combinations")
    
    ### Scores must be numeric types!!!
    # should assert that scores are numeric types
    
    # do the mean aggregation
    aggr_mean = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).mean()
    aggr_mean.columns = [x+'_mean' for x in aggr_mean.columns]    
    
    # do the stdev aggregation
    aggr_std = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).std()
    aggr_std.columns = [x+'_stdev' for x in aggr_std.columns]

    if stats=='basic':
        results = aggr_mean.join(aggr_std)
    
    elif stats=='full':
        # do the median aggregation
        aggr_med = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).median()
        aggr_med.columns = [x+'_median' for x in aggr_med.columns]
    
        # do the skew aggregation
        aggr_skew = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).skew()
        aggr_skew.columns = [x+'_skewness' for x in aggr_skew.columns]

        # do the kurtosis aggregation
        aggr_kurt = metrics2consider_df.drop(cols2drop, axis=1).groupby(hp).apply(pd.DataFrame.kurt)
        aggr_kurt.columns = [x+'_kurtosis' for x in aggr_kurt.columns]
    
        results = aggr_mean.join(aggr_med).join(aggr_std).join(aggr_skew).join(aggr_kurt)

    if verbose: print(f"# --> Found {results.shape[0]} hyperparameters combinations")   
    return results.reset_index()  

    

def get_sc_hps(): 
    hps = [
         'epochs', 
         'hidden_sizes', 
         'middle_dropout', 
         'last_dropout', 
         'weight_decay', 
         'non_linearity',
         'last_non_linearity', 
         'lr', 
         'lr_steps',
         'lr_alpha',
         'task_weights', 
    ]
    return hps



def make_hp_string_col(metrics_df):
    """ Create a column containing a hyperparameter string, resulting from the concatenation of all settings
#     :param pandas df metrics_df: metrics dataframe like what is returned by perf_from_json() 
#     :param list of str hp_cols: list of columns to concatenate into and hyperparamter string
#     :return pandas series with hp string
    """
    
    # create one HP string from variable HPs
    hp_cols=[col for col in metrics_df.columns if col[:3]=='hp_' or col=='model_name']

    col2drop=[]
    for col in hp_cols:
        if metrics_df[col].unique().shape[0] == 1:col2drop.append(col)
        if col == 'hp_string': col2drop.append(col)
            
    remain_hp = list(set(hp_cols).difference(set(col2drop)))
    remain_hp.sort()
    if len(remain_hp) == 0:
        hp_string='single_hp'
    else: 
        print(f'hp string formalism: <{">_<".join([x for x in remain_hp])}>')
    
        hp_string=metrics_df[remain_hp[0]].astype(str).copy()

        if len(remain_hp)>1:
            for hp in remain_hp[1:]:
                hp_string +='_'+metrics_df[hp].astype(str)
            
    return hp_string



# =======================================================================
# =======================================================================
# ==== Hyperparameter selection

def find_best_hyperparam(metrics_df, min_samples=5, fold_va=[0,1,2,3,4], score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ Gets the best hyperparameters (according to mean aggregate) for each performance metrics. Performs the quorum filtering.
#     :param pandas metrics_df: dataframe containing results as provided by perf_from_json()
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param int fold_va: validation folds to look for
#     :param list strings score_type: list of perf metrics to keep (depends on columns in df_res)
#     :return dtype: pandas df containing best HPs per performance metrics
    """
    
    if verbose: print(f"# Hyperparameter selection considered score types: {score_type}")
    
    # aggregate over HPs (does the quorum on class size filtering)
    aggr_df = aggregate_overall(metrics_df, min_samples=min_samples, stats='basic', fold_va=fold_va, score_type=score_type, verbose=verbose)

    # melt the resulting data  
    aggr_dfm = melt_perf(aggr_df, score_type=[s+'_mean' for s in score_type])

    # find out best HPs for each score type
    best_hps = aggr_dfm.iloc[aggr_dfm.groupby(['score_type']).idxmax()['value'].values]
    
    # reorder columns
    columns = [c for c in best_hps.columns if c[:3] == 'hp_']
    columns.append('score_type')
    columns.append('value')
    
    return best_hps[columns]

    


def melt_perf(metrics_df, score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa']):
    """ Melts (or unpivot) the performance dataframe resuting from perf_from_conf(). 
#     :param pandas df_res: dataframe containing results as provided by perf_from_conf()
#     :param list strings perf_metrics: list of perf metrics to keep (depends on columns in df_res)
#     :return dtype: pandas df containing performance in melted format usefull for R ggplot2
    """
    
    for metric in score_type:
        assert metric in metrics_df.columns, f"performance metrics {metric} not found in data frame"
    
    
    hp_cols = [x for x in metrics_df.columns if x[:3]=='hp_' or x=='model_name']
    assert len(hp_cols) > 0, 'No hyperparamters found in dataframe, use hp_* prefix for hyperparameters columns'
    
    # add possible columns in the id col for melting 
    if 'fold_va' in metrics_df.columns: hp_cols.append('fold_va')
    
    for p in score_type: 
        assert p in metrics_df.columns, f'"{p}" column not found in df_res'
    
    dfm = metrics_df.melt(id_vars=hp_cols, value_name='value', var_name='score_type').reset_index()
    dfm = dfm.loc[dfm['score_type'].isin(score_type)]
    dfm['value'] = dfm['value'].astype(float)
    
    
    return dfm



def extract_best_hp_records(perf_metrics, wanted_metrics, fold_va=[0,1,2,3,4]):
    """From a HP search results df , extracts the rows corresponding to the best HP given a score type. The best HP is selected with an aggregate_overall() [mean]
#      :param pandas perf_metrics containing the HP search results return from perf_from_json()
#      :str wanted_metrics: the score type name to use for HP selection
#      :list fold_va: validation folds to look for
    """

    best_hps = find_best_hyperparam(perf_metrics, min_samples=5, score_type=[wanted_metrics], fold_va=fold_va)
    
    hp_cols = [col for col in best_hps.columns if col[:3]=='hp_']
    selection = best_hps.loc[best_hps['score_type']==wanted_metrics+'_mean']
    hp_set = selection.set_index(hp_cols).index[0]
    
    top_perf = perf_metrics.set_index(hp_cols).loc[hp_set,:].reset_index()
    
    return top_perf




def best_hyperparam(dfm):
    """ Gets the best hyperparameters (according to mean aggregate) for each performance metrics from dfm resulting from melt_perf(). 
    ! ! ! Assumes the quorum filtering was performed beforehands. 
#     :param pandas dfm: dataframe containing results as provided by melt_perf()
#     :return dtype: pandas df containing best HPs per performance metrics
    """

    agg_df = all_hyperparam(dfm)
    best_hps = agg_df.iloc[agg_df.groupby(['score_type']).idxmax()['value'].values]
    
    # reorder columns
    columns = [c for c in best_hps.columns if c[:3] == 'hp_']
    columns.append('score_type')
    columns.append('value')
    
    return best_hps[columns]



def all_hyperparam(dfm):
    """ Gives a sorted overview of the performance of all hyperparameter sets """
    dfm = dfm[~dfm['value'].isna()].reset_index() 
    hp_cols = set([x for x in dfm.columns if x[:3]=='hp_'])
    assert len(hp_cols) > 0, 'No hyperparamters found in dataframe, use hp_* prefix for hyperparameters columns'
    
    ## Checking for unused hyperparameter columns: 
    cols_to_remove = set()
    for col in hp_cols: 
        if dfm[col].isnull().all(): 
            cols_to_remove.add(col)
        hp_cols = hp_cols.difference(cols_to_remove)
    hp_cols.add('score_type')
    hp_cols = list(hp_cols)
    agg_df = dfm.groupby(hp_cols).mean().sort_values('value',ascending=False).reset_index()
    for col in cols_to_remove: 
        agg_df[col] = None # this works with non-numerical fields only - use np.nan for numeric 
    return agg_df









# =======================================================================
# =======================================================================
# ==== Performance calculation from y_hat

def perf_from_yhat(y_labels_pred, y_hat, verbose=True, limit=None):
    
    # true_labels: scipy_sparse suscritable, rows=cmpds, columns=tasks , for compounds predicted, same order
    # y_hat: np.ndarray of predicted compounds (in y_hat) 
    # needs y_labels_pred to be mapped to y_hat compound mappings to indices in true labels
    # form y_true and y_pred usign the mapping --> maskout compounds in y_hat not present in y_hat
    
    data = []
    for t in range(y_hat.shape[1]):
        if verbose and t%1000==0:print(t)

        Y_true = y_labels_pred[:,t]
        y_true = Y_true.data
        nnz    = Y_true.nonzero()[0]
        
        y_pred = np.array([])
        if len(nnz) > 0:            
            y_pred = y_hat[nnz,t]
            
        perf_df = all_metrics(y_true, y_pred)
        perf_df['task'] = t
        data.append(perf_df)
        
        if limit is not None and t>limit:break
            
    return pd.concat(data, sort=False).reset_index(drop=True)

    

# Copied form Sparsechem utils.py: 
def all_metrics(y_true, y_score):
    """ For a task, computes all the performance scores such as sparsechem does from Y_hat predictions vectors 
#     :param  np.array y_true: containing true labels: should be positive:1, negative:-1
#     :param  np.array y_score: containing predicitons from y_hat
#     :return pandas df: data frame containing the computed scores

    """
    res_dict = {"num_pos":[np.nan], 
                "num_neg":[np.nan],
                "auc_roc": [np.nan], 
                "auc_pr": [np.nan], 
                "avg_prec_score": [np.nan], 
                "max_f1_score": [np.nan], 
                "kappa": [np.nan]}
    
    y_classes = np.where(y_score > 0.5, 1, -1)     
    
    if len(y_true) <= 1:
        df = pd.DataFrame(res_dict)
        return df
    if (y_true[0] == y_true).all():
        df = pd.DataFrame(res_dict)
        return df
    
    roc_auc_score = sklearn.metrics.roc_auc_score(y_true  = y_true, y_score = y_score)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true = y_true, y_pred = y_classes).ravel()
    
    ## calculating F1 for all cutoffs
    F1_score       = np.zeros(len(precision))
    mask           = precision > 0
    F1_score[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])

    max_f1_score   = F1_score.max()
    auc_pr         = sklearn.metrics.auc(x = recall, y = precision)
    avg_prec_score = sklearn.metrics.average_precision_score(y_true  = y_true, y_score = y_score)
    kappa          = sklearn.metrics.cohen_kappa_score(y_true, y_classes)
    
    n_pos = np.where(y_true>0)[0].shape[0]
    n_neg = np.where(y_true<0)[0].shape[0] 
    
    df = pd.DataFrame({"num_pos": [n_pos], 
                       "num_neg": [n_neg], 
                       "auc_roc": [roc_auc_score], 
                       "auc_pr": [auc_pr], 
                       "avg_prec_score": [avg_prec_score], 
                       "max_f1_score": [max_f1_score], 
                       "kappa": [kappa], 
                       "tn":[tn],
                       "fp":[fp],
                       "fn":[fn],
                       "tp":[tp]})
    return df











# =======================================================================
# =======================================================================
# ==== Performance analysis


# This function calculates the p value for the statistical difference between two ROC AUC curves
# Based on :  http://www.med.mcgill.ca/epidemiology/hanley/software/Hanley_McNeil_Radiology_82.pdf
# Authors : Adam Arany, Jaak Simm (KU Leuven)
def auc_se(auc, num_pos, num_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)

    return np.sqrt((auc*(1-auc) + (num_pos-1)*(q1 - auc**2) + (num_neg-1)*(q2 - auc**2)) / (num_pos*num_neg))

def pvalue(auc1, num_pos1, num_neg1, auc2, num_pos2, num_neg2):
    se1 = auc_se(auc1, num_pos1, num_neg1)
    se2 = auc_se(auc2, num_pos2, num_neg2)
    z   = (auc1 - auc2) / np.sqrt(se1**2 + se2**2)

    return 1.0 - scipy.stats.norm.cdf(z)



def delta_to_baseline(top_baseline, list_top_perfs, fold_va=[0,1,2,3,4]):
    """ Computes the delta-to-baseline of a list of models to compare. The performance data frames are like what perf_from_json returns. A quorum selection filter will take place.
#       :param pandas df top_baseline, all task performance of the baseline model (assumed to correspond to the best HP)    
#       :param list of pandas df top_perf: a list of performance data frames corresponding to models to comapre 
#       :param list of pandas df top_perf: a list of performance data frames corresponding to models to comapre 
#       :param list fold_va: validation folds to look for
#       :return pandas df containing the performance deltas
    """
    deltas = []

    col2keep = ['task', 'fold_va', 'input_assay_id', 'model_name', 'roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa', 'num_pos', 'num_neg']
    top_baseline_scores = top_baseline.drop([x for x in top_baseline.columns if x not in col2keep],axis=1)

    # apply quorum filtering to baseline results
    baseline_valid = quorum_filter(top_baseline_scores, fold_va=fold_va)
    
    for top_perf in list_top_perfs:
        
        top_perf_scores = top_perf.drop([x for x in top_perf.columns if x not in col2keep],axis=1)
        model_name = top_perf['model_name'].iloc[0]
    
        # merge performance based on input_assay_id and num_pos, num_neg (not task identifiers since they could in principle be different) ???
        merged = baseline_valid.merge(top_perf_scores, on=['task','fold_va', 'input_assay_id',  'num_pos', 'num_neg'], suffixes=('', '_'+model_name))
        
    
        # calculate the delta for each score now
        for s in ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa']:
            merged[s+'_delta'] = merged[s+'_'+model_name] - merged[s]

        d = merged[['task', 'fold_va', 'input_assay_id','roc_auc_score_delta', 'auc_pr_delta', 'avg_prec_score_delta', 'max_f1_score_delta','kappa_delta', 'num_pos', 'num_neg']].copy()   
        
        d['model_name'] = model_name
        deltas.append(d)    
    
    return pd.concat(deltas, sort=False)




def delta_to_baseline_from_assay_ids(top_baseline, list_top_perfs, fold_va=[0,1,2,3,4]):
    """ Computes the delta-to-baseline of a list of models to compare. The performance data frames are like what perf_from_json returns. A quorum selection filter will take place.
#       :param pandas df top_baseline, all task performance of the baseline model (assumed to correspond to the best HP) 
#       :param list fold_va: validation folds to look for
#       :param list of pandas df top_perf: a list of performance data frames corresponding to models to comapre 
#       :return pandas df containing the performance deltas
    """
    deltas = []

    col2keep = ['task', 'fold_va', 'input_assay_id', 'model_name', 'roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa', 'num_pos', 'num_neg', 'threshold_value']
    top_baseline_scores = top_baseline.drop([x for x in top_baseline.columns if x not in col2keep],axis=1)
    
    baseline_valid = quorum_filter(top_baseline_scores, fold_va=fold_va)
    baseline_valid['threshold_value'] = baseline_valid['threshold_value'].round(4)
    
    
    for top_perf in list_top_perfs:
        
        top_perf_scores = top_perf.drop([x for x in top_perf.columns if x not in col2keep],axis=1)
        top_perf_scores['threshold_value'] = top_perf_scores['threshold_value'].round(4)
        
        model_name = top_perf['model_name'].iloc[0]
        
        
        # merge performance based on input_assay_id and num_pos, num_neg (not task identifiers since they could in principle be different)
        merged = baseline_valid.merge(top_perf_scores, on=['fold_va', 'input_assay_id',  'threshold_value', 'num_pos', 'num_neg'], suffixes=('', '_'+model_name))
        
        # calculate the delta for each score now
        for s in ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa']:
            merged[s+'_delta'] = merged[s+'_'+model_name] - merged[s]

        d = merged[['task', 'task_'+model_name, 'fold_va', 'input_assay_id','roc_auc_score_delta', 'auc_pr_delta', 'avg_prec_score_delta', 'max_f1_score_delta','kappa_delta', 'num_pos', 'num_neg']].copy()   
        d['model_name'] = model_name
        deltas.append(d)

    return pd.concat(deltas, sort=False)








# =======================================================================
# =======================================================================
# ==== Plotting functions


# function to capture the boxplot specs
def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label']  = labels[i]
        dict1['whislo'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['q1']     = bp['boxes'][i].get_ydata()[1]
        dict1['med']    = bp['medians'][i].get_ydata()[1]
        dict1['mean']   = bp['means'][i].get_ydata()[0]
        dict1['q3']     = bp['boxes'][i].get_ydata()[2]
        dict1['whishi'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


# This function allows to compute the boxplot of performance delta per task size category
# capture the boxes lower whisker, q1, med, q3 and higher lower whisker
# this will allow sharing data necessary for boxplots by preserving the privacy of the full performance data frame
def compute_boxplots(perf_deltas, labels, category_column, score_type=['roc_auc_score_delta', 'auc_pr_delta', 'avg_prec_score_delta', 'max_f1_score_delta', 'kappa_delta']):
    boxes = []
    scored_labels = []
    fig, ax = plt.subplots(figsize=(20,5))
    
    for cat in labels:
        cat_df = perf_deltas.loc[perf_deltas[category_column]==cat]
        
        for score in score_type: 
            b = ax.boxplot(cat_df[score], showmeans=True)
            boxes.append(b)
            scored_labels.append(f'{score}:{cat}')

    plt.close()

    # concatenate the different boxplots specs into a data frame
    boxplots_specs = pd.concat([get_box_plot_data([scored_labels[i]],boxes[i]) for i in range(len(boxes))], ignore_index=True)
    
    label_split = boxplots_specs.label.str.split(':', expand=True)
    
    boxplots_specs['score_type'] = label_split[0]
    boxplots_specs['label'] = label_split[1]
    
    return boxplots_specs#, boxes[0], scored_labels[0]


#  exple of how to reconstruct boxplot from the boxplot specs
def reconstruct_boxplot(boxplot_specs, figsize=(10,10)):
    score_type = boxplot_specs['score_type'].unique()
    num_scores = score_type.shape[0]
    
    fig, ax = plt.subplots( num_scores, 1, figsize=figsize)
    for i, score in enumerate(score_type): 
        boxes= []
        score_df = boxplot_specs.loc[boxplot_specs['score_type']==score].drop('score_type',axis=1)
        for k,v in score_df.to_dict(orient='index').items():
            v['fliers'] = []
            boxes.append(v)

        ax[i].bxp(boxes, showmeans=True, showfliers=True,  patch_artist=True)
        ax[i].set_title(score)
        
    plt.show()
    return 


def reconstruct_boxplot_colored(boxplot_specs, n_partner, n_bins, figsize=(10,10)):
    
    score_type = boxplot_specs['score_type'].unique()
    num_scores = score_type.shape[0]

    
    color_list = list(itertools.chain.from_iterable(itertools.repeat(x, n_bins) for x in sns.color_palette("husl", n_partner)))
    colors = itertools.cycle(color_list)
    
    fig, ax = plt.subplots( num_scores, 1, figsize=figsize, sharex=True)
    for i, score in enumerate(score_type): 
        boxes= []
        score_df = boxplot_specs.loc[boxplot_specs['score_type']==score].drop('score_type',axis=1)
        for k,v in score_df.to_dict(orient='index').items():
            v['fliers'] = []
            boxes.append(v)

        bplot = ax[i].bxp(boxes, showmeans=True, showfliers=True,  patch_artist=True)
        ax[i].set_title(score)
        #ax[i].set_ylim(-0.1, 0.1)
        # fill with colors
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    
    plt.xticks(rotation=90)
    plt.show()
    return colors






def folds_asym_error_plot(metrics_df,  
                          min_samples=5,
                          fold_va=[0,1,2,3,4],
                          score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score', 'kappa'],
                          error_bar_fraction_tasks=0.62,
                          gap_hue=0.5,
                          figsize = (20, 18), 
                          hue_order='auto'):
    """ Validation folds means plotted with matplotlib errorbar and asymetric error bars. The error bar will cover 'error_bar_fraction_tasks' tasks (default, 62%) and will guaranty that the mean point separates the error bar in two parts covering equivalent fractions of the tasks. Peforms the quorum filtering and mean aggregation over folds.
#     :param pandas df metrics_df: metrics dataframe like what is returned by perf_from_json() 
#     :param int min_samples: minimum samples of positive / negative to be present in each fold for a task performance to be considerd (on thee basis of HPs)
#     :param list fold_va: number of folds to be looking for
#     :param float error_bar_fraction_tasks, the fraction of tasks that the error bar will cover (0<=f<=1)
#     :param float gap_hue, x interval in which HP fold performance will be plotted (around X=n_cv)
#     :param list of str score_type: list of score types to consider
#     :param list of str hp_order: list of hyperparamter strings to use ( modeval.make_hp_string_col(metrics_df, hp_cols) lists the HPs)
#     :param str x_group: allows displaying swarms of dots by different variables. The specifed value will be used for X-axis. This is required to be present in melted dataframe [see melt_perf(...)]
#     :param str hue_group: allows ordering of hue swarms of dots by different variables. The specifed value will be used to determine order. This is required to be present in melted dataframe [see melt_perf(...)]
#     :return void (plots the figure) 
    """
    assert error_bar_fraction_tasks > 0 and error_bar_fraction_tasks <=1, "error_bar_fraction_tasks must be < 0 and <= 1"

    # data filtering: filters tasks-HP that do not fullfill the requirement: min_samples from each class in each fold_va fold
    perf_to_consider = quorum_filter(metrics_df, min_samples=5, fold_va=fold_va, verbose=True)
    perf_to_consider['hp'] = make_hp_string_col(perf_to_consider)
    
    if hue_order == 'auto':hue_order = perf_to_consider['hp'].unique()
        

    # example error bar values that vary with x-position
    fig, axes= plt.subplots(nrows=5, figsize=figsize)

    # define color range
    colors = itertools.cycle(sns.color_palette("husl", len(hue_order)))

    

    k=0
    for score_name in score_type:
        for fold in fold_va:
                                               
            # first calculate the means for each fold-HP in specifed order
            means = np.array([perf_to_consider.loc[(perf_to_consider['hp']==x)&(perf_to_consider['fold_va']==fold)][score_name].mean() for x in hue_order])
        
            # get the number of tasks corresponding to the wanted fraction of task p and divide by two to find out upper and lower errors
            # this number will be used to truncate the perf records above and below the mean at the position corresponding to the number of wanted records on each side 
            # (half the numebr of records in wanted fraction)
            n=int(perf_to_consider.loc[(perf_to_consider['fold_va']==fold)].shape[0]/len(hue_order)*error_bar_fraction_tasks/2)
            
            lower_quantiles, higer_quantiles= [],[]
            for x in range(len(hue_order)):

                
                lower_limit = perf_to_consider.loc[(perf_to_consider['hp']==hue_order[x])&
                                                   (perf_to_consider['fold_va']==fold)&
                                                   (perf_to_consider[score_name]<means[x])][score_name].sort_values().iloc[-n] 
                
                higher_limit = perf_to_consider.loc[(perf_to_consider['hp']==hue_order[x])&
                                                    (perf_to_consider['fold_va']==fold)&
                                                    (perf_to_consider[score_name]>means[x])][score_name].sort_values().iloc[n] 
                
                lower_quantiles.append(lower_limit) 
                higer_quantiles.append(higher_limit) 
                
        
            # then determine the y-coordinates of the error bars
            lower_error = means - lower_quantiles
            upper_error = higer_quantiles - means
            asymmetric_error = [lower_error, upper_error]
        
           
            # determine the X-shift between HPs of the same plot
            # we want to plot in interval [-.4 , + .4] relative to each fold X coordinate (i.e. 0,1,2,3,4) 
            # interval is = 0.8 and we want to plot all HP with regular intervals
            shift = gap_hue/len(hue_order)
            
        
            # finally, apply the error plotting with previousely defined limits
            for i in range(len(means)):
                axes[k].errorbar(x=fold+shift*i-(len(means)-1)/2*shift, 
                                 y=means[i], 
                                 yerr=[[asymmetric_error[0][i]],[asymmetric_error[1][i]]], 
                                 fmt='o',
                                 color=next(colors),
                                 elinewidth=0.6,
                                 capsize=2)
                axes[k].set_title(score_name)
            
                if k < len(fold_va) - 1: 
                    axes[k].set_xticklabels([])
                    
                else:
                    axes[k].set_xlabel("validation fold")
        k+=1
    
    plt.show()
    
    return





def pointplot_fold_perf(metrics_df, 
                        score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score', 'kappa'],
                        figsize = (20, 18), 
                        fold_va=[0,1,2,3,4], 
                        x_group='fold_va',
                        x_order='auto',
                        hue_order='auto',
                        hue_group='hp',
                        ):
    """ Seaborn poinplot of fold mean performance with standard dev error bars. X-axis: hyperparamter combinations; Y-axis: mean performance of a fold 
#     :param pandas df metrics_df: metrics dataframe like what is returned by perf_from_json() 
#     :param list of str score_type: list of score types to consider
#     :param list of str hp_order: list of hyperparamter strings to use ( modeval.make_hp_string_col(metrics_df, hp_cols) lists the HPs)
#     :param str x_group: allows displaying swarms of dots by different variables. The specifed value will be used for X-axis. This is required to be present in melted dataframe [see melt_perf(...)]
#     :param str hue_group: allows ordering of hue swarms of dots by different variables. The specifed value will be used to determine order. This is required to be present in melted dataframe [see melt_perf(...)]
#     :return void (plots the figure) 
    """
    assert len(score_type) > 1, "Minimum number of score_types is 2"
    assert x_group != hue_group, "x_group and hue_group need to be different "
    assert (x_group == 'fold_va' and hue_group=='hp') or (x_group == 'hp' and hue_group=='fold_va'), "x_group and hue_group need to be either 'fold_va' or 'hp', interverted"
    
    # filter performance applying the quorumn filter
    perf_to_consider  = quorum_filter(metrics_df, min_samples=5, fold_va=fold_va, verbose=True)
    
    # format for plotting
    perf_to_consider['hp'] = make_hp_string_col(perf_to_consider)
    print(f"# --> {perf_to_consider['hp'].unique().shape[0]} hp combin found")
    
    # set the order of x and hue 
    if x_order=='auto' and x_group=='hp': x_order=np.sort(perf_to_consider['hp'].unique())
    elif x_order=='auto' and x_group=='fold_va': x_order=np.sort(perf_to_consider['fold_va'].unique())
    if hue_order=='auto' and hue_group=='hp': hue_order=np.sort(perf_to_consider['hp'].unique())
    elif hue_order=='auto' and hue_group=='fold_va': hue_order=np.sort(perf_to_consider['fold_va'].unique())
    
    # this will be a subplots stack, each of them shows perf according to one score type
    num_metrics = len(score_type)
    fig, axes = plt.subplots(num_metrics,1, figsize=figsize)

    i=0
    for score_name in score_type:
        
        # do a swarmplot for every score type
        perf_data = perf_to_consider[['hp', 'fold_va',score_name]].copy()
        
        sns.pointplot(x=x_group, 
                      y=score_name, 
                      data=perf_data, 
                      order=x_order,
                      hue=hue_group, 
                      hue_order=hue_order,
                      dodge=0.4, 
                      join=False, 
                      palette='husl', 
                      ci="sd",
                      errwidth=0.7,
                      capsize=0.02,
                      ax=axes[i])
        
        # if last panel, display axis ticks, axis title and a legend
        if i == len(score_type)-1 or len(score_type)==1: 
            if x_group == 'hp': axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)
            elif x_group == 'fold_va': axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
            
        else:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel("")
        
        # set the legend out of the plot area
        axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue_group)   
        
        # control the window range
        #mini=perf_data[score_name].mean() - perf_data[score_name].std()
        #maxi=perf_data[score_name].mean() + perf_data[score_name].std()
        #axes[i].set_ylim(mini-mini/10,maxi+maxi/10)
        #axes[i].set_ylim(0.805,0.825)
        axes[i].set_title(score_name)

        # add a vertical line separation
        for k in range(len(x_order)):
            axes[i].axvline(k+0.5, 0,1, color="grey", alpha=0.5)

        i+=1

    return


def ratioplot(perf_metrics,
              min_samples=5,
              fold_va=[0,1,2,3,4],
              score_type=['roc_auc_score', 'auc_pr','avg_prec_score', 'max_f1_score', 'kappa'], 
              order='auto'):
    """
    
    """
    
    perf_2_consider = quorum_filter(perf_metrics, min_samples=min_samples, fold_va=fold_van_cv)


    # aggregate performance of tasks over the folds
    aggr_perf = aggregate_task_perf(perf_2_consider, fold_va=fold_va, min_samples=min_samples,score_type=score_type , stats='basic')
    aggr_perf['hp'] = make_hp_string_col(aggr_perf)    
    score_type = [x+'_mean' for x in score_type]
    
    if order == 'auto':order=aggr_perf['hp'].unique()

    data = {x:[] for x in score_type}
    for score in score_type:
        
        cutoff_sets = []
        hp_sets = []
        for hp in order:
        
            one_hp = aggr_perf.loc[aggr_perf['hp']==hp]
    
            for cutoff in np.arange(0, 1.01, 0.01)[::-1]:
                r = np.where(one_hp[score]>= cutoff, 1, 0).sum()/one_hp.shape[0]
        
                cutoff_sets.append(cutoff)
                hp_sets.append(hp)
                data[score].append(r)

    df = pd.DataFrame(data)
    df['hp'] = hp_sets
    df['cutoff'] = cutoff_sets
    

    num_metrics = len(score_type)
    fig, ax = plt.subplots(1,num_metrics, figsize=(20, 10))

    for i, score in enumerate(score_type):
        print(i, score)
        sns.lineplot(x="cutoff", y=score, data=df[['hp', "cutoff", score]], hue="hp", hue_order=order, ax=ax[i], palette='husl')
        ax[i].set_xlim(1, 0)
        ax[i].set_xlabel(f"{score} cutoff")
        ax[i].set_ylabel(f"ratio tasks with better than X {score}")
    
    return df

    


def swarmplot_fold_perf(metrics_df, 
                        score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score', 'kappa'],
                        figsize = (20, 18), 
                        fold_va=[0,1,2,3,4], 
                        x_group='fold_va',
                        x_order='auto',
                        hue_order='auto',
                        hue_group='hp',
                        ):
    """ Seaborn swarplot of fold mean performance. X-axis: hyperparamter combinations; Y-axis: mean performance of a fold 
#     :param pandas df metrics_df: metrics dataframe like what is returned by perf_from_json() 
#     :param list of str score_type: list of score types to consider
#     :param list of str hp_order: list of hyperparamter strings to use ( modeval.make_hp_string_col(metrics_df, hp_cols) lists the HPs)
#     :param str x_group: allows displaying swarms of dots by different variables. The specifed value will be used for X-axis. This is required to be present in melted dataframe [see melt_perf(...)]
#     :param str hue_group: allows ordering of hue swarms of dots by different variables. The specifed value will be used to determine order. This is required to be present in melted dataframe [see melt_perf(...)]
#     :return pandas series with hp string  
    """
    assert len(score_type) > 1, "Minimum number of score_types is 2"
    assert x_group != hue_group, "x_group and hue_group need to be different "
    assert (x_group == 'fold_va' and hue_group=='hp') or (x_group == 'hp' and hue_group=='fold_va'), "x_group and hue_group need to be either 'fold_va' or 'hp', interverted"
    
    # aggregate performance of each fold to get a value per fold per HP
    perf_fold  = aggregate_fold_perf(metrics_df, 5, stats='basic', fold_va=fold_va, score_type=score_type)
    
    # format for plotting
    perf_foldm = melt_perf(perf_fold, score_type=[x+'_mean' for x in score_type])
    perf_foldm['hp'] = make_hp_string_col(perf_foldm)
    print(f"# --> {perf_foldm['hp'].unique().shape[0]} hp combin found")
    
    # set the order of x and hue 
    if x_order=='auto' and x_group=='hp': x_order=np.sort(perf_foldm['hp'].unique())
    elif x_order=='auto' and x_group=='fold_va': x_order=np.sort(perf_foldm['fold_va'].unique())
    if hue_order=='auto' and hue_group=='hp': hue_order=np.sort(perf_foldm['hp'].unique())
    elif hue_order=='auto' and hue_group=='fold_va': hue_order=np.sort(perf_foldm['fold_va'].unique())
    
    # this will be a subplots stack, each of them shows perf according to one score type
    num_metrics = len(score_type)
    fig, axes = plt.subplots(num_metrics,1, figsize=figsize)

    i=0
    for score_name in perf_foldm.score_type.unique():
        
        # do a swarmplot for every score type
        perf_data = perf_foldm.loc[perf_foldm['score_type']==score_name]
        sns.swarmplot(x=x_group, 
                      y="value", 
                      data=perf_data, 
                      hue=hue_group, 
                      hue_order=hue_order,
                      palette="husl", 
                      order=x_order, 
                      size=8, linewidth=1, dodge=True, alpha=.85, ax=axes[i])

        # if last panel, display axis ticks, axis title and a legend
        if i == len(score_type)-1 or len(score_type)==1: 
            if x_group == 'hp': axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)
            elif x_group == 'fold_va': axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue_group)   
        else:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel("")
            axes[i].get_legend().remove()
        
        # control the window range
        mini=perf_data['value'].min()
        maxi=perf_data['value'].max()
        axes[i].set_ylim(mini-mini/100,maxi+maxi/100)
        
        axes[i].set_title(score_name)

        # add a vertical line separation
        for k in range(len(x_order)):
            axes[i].axvline(k+0.5, 0,1, color="grey", alpha=0.5)

        i+=1

    return





def match_best_tasks(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', fold_va=[0,1,2,3,4], hp_selection_metric='auc_pr',filename_mask_x=None, filename_mask_y=None, min_samples=5): 

    hp_bests = list()
    for e in ((results_dir_x, t3_mapped_x, filename_mask_x),(results_dir_y, t3_mapped_y, filename_mask_y)):
        df_t3_mapped = pd.read_csv(e[1])
        main_tasks = df_t3_mapped.loc[df_t3_mapped['assay_type']!=aux_assay_type].cont_classification_task_id.dropna().values
        json_df = perf_from_json(e[0],aggregate=False,tasks_for_eval=main_tasks, fold_va=fold_va, filename_mask=e[2]) 
        #json_melted = melt_perf(json_df, score_type=[hp_selection_metric])
        #hp_best = best_hyperparam(json_melted)

        hp_best = find_best_hyperparam(json_df, fold_va=fold_va,min_samples=min_samples,score_type=[hp_selection_metric])
        # keeping only the records associated to the best hyperparameters
        hp_cols = ['hp_'+hp for hp in get_sc_hps()]
        hp_best = pd.merge(json_df[~json_df[hp_selection_metric].isna()],hp_best,how='inner',on=hp_cols)
        hp_best = pd.merge(df_t3_mapped,hp_best,how='inner',left_on='cont_classification_task_id',right_on='task')
        hp_bests.append(hp_best)

    return hp_bests

def statistical_model_comparison_analysis(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', fold_va=[0,1,2,3,4], min_samples=5, filename_mask_x=None, filename_mask_y=None):
    """ Run a statistical significance analysis between two runs, both with hyperparameter optimization
     Best hyperparameters will be selected based on the auc_pr
     For now, only compatible with the json result format . 
     Result is the difference between the second (y) and the first (x) arguments
    :param  str path to the results folder of the 1st run, containing the json files
    :param  str path to the mapped T3 file of the 1st run
    :param  str axis label for the plot, designating the 1st run
    :param  str path to the results folder of the 2nd run, containing the json files
    :param  str path to the mapped T3 file of the 2nd run
    :param  str axis label for the plot, designating the 2nd run
    :param list fold_va: specify folds to be used for valid
    :param int min_samples: statistics calculations will be limited to the tasks which have at least this number of positives and negatives
    :param str filename_mask_x: specify a filename mask to restrict the perf loading to a subset of files for the 1st run results
    :param str filename_mask_y: specify a filename mask to restrict the perf loading to a subset of files for the 2nd run results
    :return None 
    """ 

    hp_bests = match_best_tasks(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', fold_va=fold_va, filename_mask_x=filename_mask_x, filename_mask_y=filename_mask_y, min_samples=min_samples)
    # matching x and y runs
    df_merge = pd.merge(hp_bests[0], hp_bests[1],how='inner',on=['input_assay_id','threshold_value','fold_va'])

    # plotting
    # statistical analysis only valid for aggregated figures over folds - hence groupby
    table = df_merge.groupby(by=['input_assay_id','threshold_value']).mean().reset_index() 
    metric_x = 'roc_auc_score_x'
    metric_y = 'roc_auc_score_y'
    x_lim = 0.5
    y_lim = 0.5
    title = 'AUC ROC'

    res_wide = summarize_diff_statistics(hp_bests[0],hp_bests[1],min_samples=min_samples, fold_va=fold_va)
    table = table[~table['roc_auc_score_y'].isna()]
    table = table[~table['roc_auc_score_x'].isna()]
    res_stat_sign = plot_statisical_significance(table, metric_x, metric_y, label_x, label_y, x_lim, y_lim, title )
    res = pd.concat([res_stat_sign[0], res_wide], axis=1)

    return res 

def summarize_diff_statistics(tasks_x, tasks_y, min_samples=5, fold_va=[0,1,2,3,4]): 
    """ Get the difference of overall aggregated metrics between two task-level dataframe
    Result is the difference between the second and the first argument
#    :param  dataframe containing task-level metrics (reference to compare with)
#    :param  dataframe containing task-level metrics (first term in the difference)
    """
    ress = []
    fields = []
    aggs = ['mean', 'median', 'stdev', 'kurtosis', 'skewness']
    metrics = ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score', 'kappa']
    for a in aggs: 
        for m in metrics: 
            fields.append(m + '_' + a)

    for t in [tasks_x,tasks_y]: 
        # take care of possible missing task weights : will be ignored in groupby so any impute value would do
        t['hp_task_weights'] =  'irrelevant'

        # aggregate_overall goes over the folds and tasks, meaning e.g. the median will be calculated over the folds and tasks
        # it might be preferable to first compute the mean over the folds, then generate the full set of statistics
        t = aggregate_task_perf(t,min_samples=min_samples, fold_va=fold_va, stats='basic')
        t = t.rename(columns={'roc_auc_score_mean': 'roc_auc_score', 'auc_pr_mean': 'auc_pr', 'avg_prec_score_mean': 'avg_prec_score','max_f1_score_mean':'max_f1_score','kappa_mean':'kappa'})
        t.to_csv('t.csv')
        t['fold_va'] = 0  # on previous line, aggregated over the folds, this will trick the aggregate_overall in believing that all folds are there
        ress.append(aggregate_overall(t,min_samples=min_samples, n_cv=1, stats='full')) #### TO DO : this has to be fixed
        
    ress[1].to_csv('ress1.csv')
    res = ress[1][fields]-ress[0][fields]
    return res 

def plot_statisical_significance(table, metric_x, metric_y, label_x, label_y, x_lim, y_lim, title=None):

    ax = plt.subplot(111)
    table['pvalue'] = table.apply(lambda r : pvalue(r[metric_y],r['num_pos_y'],r['num_neg_y'], r[metric_x],r['num_pos_x'],r['num_neg_x']), axis=1)
    table['stat_rev_multi+'] = (table['pvalue']<0.05) & (table[metric_y] > table[metric_x])
    table['stat_rev_multi-'] = (table['pvalue']>0.95) & (table[metric_y] < table[metric_x])

    d = {
        'statistically significantly better tasks (y>x)' : ['{:.4%}'.format(table['stat_rev_multi+'].sum()/len(table))],
        'statistically significantly worse tasks (y<x)' : ['{:.4%}'.format(table['stat_rev_multi-'].sum()/len(table))],
    }
    res = pd.DataFrame(d)

    mp = {(True,False):'blue',(False,False):'lightgray',(False,True):'red'}
    plt.scatter(table[metric_x],table[metric_y],alpha=0.3,c=[mp[e] for e in zip(table['stat_rev_multi+'],table['stat_rev_multi-'])])
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.grid(color='lightgray', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    return res, table






# MTX manipulatiosn such as splitting per fold

def split_folds(M, fold_vector):
    """ Splits global M (i.e. Y or X) into folds Ms
#     :param  scipy.sparse.csr.csr_matrix M
#     :param  np.array fold_vector: containing folds assignment for each row.
#     :return list of size=n_folds where each element is a csr_matrix representing a fold
    """
    assert type(M) == scipy.sparse.csr.csr_matrix, "M needs to be scipy.sparse.csr.csr_matrix"
    assert fold_vector.shape[0] == M.shape[0], "fold_vector must have same shape[0] than M"
    
    folds = [M[fold_vector==f,:] for f in np.unique(fold_vector)]
    return folds


def slice_mtx_rows(M, rows_indices):
    """ Slices out rows of a scipy.sparse.csr_matrix
#     :param  scipy.sparse.csr.csr_matrix M
#     :param  np.array containing integer indices of rows to extract or boolean vector where True is a row to extract
#     :return scipy.sparse.csr.csr_matrix
    """
    assert type(M) == scipy.sparse.csr.csr_matrix, "M needs to be scipy.sparse.csr.csr_matrix"
    return M[row_indices, :]


def slice_mtx_cols(M, col_indices):
    """ Slices out columns of a scipy.sparse.csc_matrix
#     :param  scipy.sparse.csc.csc_matrix M
#     :param  np.array containing integer indices of columns to extract or boolean vector where True is a row to extract
#     :return scipy.sparse.csc.csc_matrix
    """    
    assert type(M) == scipy.sparse.csc.csc_matrix, "M needs to be scipy.sparse.csc.csc_matrix"
    return M[col_indices, :]




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


sns.set_style("whitegrid")


# function template
#def template():
#    """ description
#     :param dtype name: description
#     :param dtype name: description
#     :return dtype: description
#    """
#    return


# TO DO 
# - delta predictions perf between two (or more) models
# - better manage hyperparameters (consistent to the code)


def perf_from_json(
        model_dir_or_file, 
        tasks_for_eval=None, 
        aggregate=False, 
        evaluation_set='va',
        model_name='Y',
        n_cv=5,
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
#     :param int n_cv: specify the number of folds used for cross valid, starting with 1, higher fold_va numbers will be dropped
#     :param str filename_mask: specify a filename mask to restrict the perf loading to a subset of files
#     :return pandas df containing performance and configuration summaries 
    """
    # pandas v0.24 (pd.read_json) is not returning expected results when used with the arguement "tasks_for_eval" 
    assert pd.__version__[:4] =='0.25', 'Pandas version must be 0.25' # should this be placed somewhere else?
    
    if tasks_for_eval is not None: 
        assert type(tasks_for_eval) is np.ndarray, "tasks_for_eval must be an np.array"
        assert tasks_for_eval.ndim == 1, "tasks_for_eval must be np.array with ndim=1"
    
        for x in tasks_for_eval:
            assert int(x) == x, "elements in tasks_for_eval must be integer-like"
            assert x>=0, "elements in tasks_for_eval must be > 0"
            
        assert np.unique(tasks_for_eval).shape[0] == tasks_for_eval.shape[0], "tasks_for_eval must not have duplicates"
        
    assert n_cv != 0, "n_cv minimum value must be 1"
    
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
            assert "results_agg" in data, "Error: cannot find 'results_agg' in data"
            assert evaluation_set in data["results_agg"], f"Error: cannot find '{evaluation_set}' in data results_agg"
            assert tasks_for_eval is None, 'tasks_for_eval has no effect in combination with aggregate=True'

            res_df = pd.read_json(data["results_agg"][evaluation_set], typ="series").to_frame().transpose()
            res_df.columns = [x+'_agg' for x in res_df.columns]
        
        else: 
            assert "results" in data, "Error: cannot find 'results' in data"
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
    cvfolds = list(range(n_cv))
    output_df = output_df.query('fold_va in @cvfolds')
    
    assert output_df.shape[0] > 0, f"No records found for the specified cv fold {n_cv}:{cvfolds}"
    
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



def verify_cv_runs(metrics_df, n_cv=5):
    """ From the metrics dataframe yielded by perf_from_json(), cehcks if each hyperparameter was run n_cv
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int n_cv: number of folds to look for
#     :return void: prints message in stdout
    """    
    hp = [x for x in metrics_df.columns if x[:3] == 'hp_' and not metrics_df[x].isnull().all()]
    hp.append('model_name')
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_ and model_name"
    
    aggr = metrics_df.dropna(axis='columns', how='all').sort_values('fold_va').groupby(hp)['fold_va'].apply(lambda x: ','.join([str(x) for x in x.unique()]))
    
     # if missing ones, print out a warning message
    valid=True
    folds = ",".join([str(x) for x in range(n_cv)])
    if aggr[~aggr.str.contains(folds)].shape[0]>0:
        valid = False
        print("WARNING: missing fold runs")
        print(f"Fold runs found :\n {aggr}")
    
    return valid



def quorum_filter(metrics_df, min_samples=5, n_cv=5, verbose=True):
    """ Filter the metrics data frame of each model (as defined by a HP set) with a quorum rule: minimum N postive samples and N negative sample in each fold 
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int min_samples: minimum class size per fold
#     :param int n_cv: number of folds to look for
#     :param bool verbose: adds verbosity
#     :return pandas df filtered_df: metrics data frame containing , for every given HPs sets, only tasks present in each of the folds 
    """
    if verbose: print(f"# Quorum on class size applied: {min_samples}-{min_samples}")
    
    assert verify_cv_runs(metrics_df, n_cv=n_cv), "Missing cv run, abort."
    assert 'fold_va' in metrics_df.columns, "fold_va must be present in metrics data frame"
    assert 'task' in metrics_df.columns, "task must be present in metrics data frame"
    assert 'num_pos' in metrics_df.columns, "num_pos must be present in metrics data frame"
    assert 'num_neg' in metrics_df.columns, "num_neg must be present in metrics data frame"
    
    index_cols = ['fold_va', 'task', 'model_name'] + [col for col in metrics_df if col[:3] == 'hp_' if not metrics_df[col].isna().all()] 
    df = metrics_df.set_index(keys=index_cols, verify_integrity=True).copy()
    
    # add a dummy column to perform the count
    df['_'] = 1
    
    # first filter rows with at least N positives and N negatives in each fold 
    task_mini = df.loc[(df['num_pos']>=min_samples)&(df['num_neg']>=min_samples)]
    
    # then count the number of folds for each <task,hp> and filter (it must be present in each fold)
    levels = [x for x in range(1, len(index_cols))]
    task_count = task_mini['_'].groupby(level=levels).count()
    selected_tasks = task_count[task_count==n_cv].index.unique()
    
    filtered_res = df.reset_index(level=0).loc[selected_tasks, :].reset_index().drop('_',axis=1)
    
    if verbose: 
        print(f"# --> Total number of tasks   : {metrics_df.task.unique().shape[0]}")
        print(f"# --> Tasks further considerd : {filtered_res.task.unique().shape[0]}")
    
    return filtered_res
    
    





# =======================================================================
# =======================================================================
# ==== aggregation functions: apply only to individual task performances 
# ==== does not apply to sparsechem aggregate performance metrics

def aggregate_fold_perf(metrics_df, min_samples=5, n_cv=5, stats='basic', score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ HP performance aggregation over folds. Includes a quorum filtering step (at least N actives and N inactives in each of the n_cv folds)
    From the metrics dataframe yielded by perf_from_metrics(), does the aggregation over the fold (mean, median, std, skewness, kurtosis) results in one perf per fold.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_metrics() 
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param int n_cv: number of folds to look for
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
    metrics2consider_df = quorum_filter(metrics_df, min_samples=min_samples, n_cv=n_cv)
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
      

    
    
def aggregate_task_perf(metrics_df, min_samples=5, n_cv=5, stats='basic', score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ HP performance aggregation over tasks. Includes a quorum filtering step (at least N actives and N inactives in each of the n_cv folds)
    From the metrics dataframe yielded by perf_from_json(), does the aggregation over the CV (mean, std) results in one perf per task.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param int n_cv: number of folds to look for
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
    metrics2consider_df = quorum_filter(metrics_df, min_samples=min_samples, n_cv=n_cv, verbose=verbose)
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
    

    
    
    
def aggregate_overall(metrics_df, min_samples=5, stats='basic', n_cv=5, score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ HP performance aggregation overall . Includes a quorum filtering step (at least N actives and N inactives in each of the n_cv folds)
    From the metrics dataframe yielded by perf_from_json(), does the aggregation over the CV (mean, std) results in one perf per hyperparameter.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_json() 
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param int n_cv: number of folds to look for
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :param string stats: ['basic', 'full'], if full, calculates skewness of kurtosis
#     :param list score_type: ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'] thelist of metrics to aggregate
#     :param bool verbose
#     :return dtype: pandas df containing performance per hyperparameter setting
    """ 
        
    hp = [x for x in metrics_df.columns if x[:3] == 'hp_' and not metrics_df[x].isna().all()]
    
    assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    
    hp.append('model_name')
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_ and model_name"
    
    
    # keep only tasks verifying the min_sample rule: at least N postives and N negatives in each of the 5 folds
    metrics2consider_df = quorum_filter(metrics_df, min_samples=min_samples, n_cv=n_cv, verbose=verbose)
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
        if metrics_df[col].unique().shape[0] == 1:
            col2drop.append(col)
            
    remain_hp = list(set(hp_cols).difference(set(col2drop)))
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

def find_best_hyperparam(metrics_df, min_samples=5, n_cv=5, score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa'], verbose=True):
    """ Gets the best hyperparameters (according to mean aggregate) for each performance metrics. Performs the quorum filtering.
#     :param pandas metrics_df: dataframe containing results as provided by perf_from_json()
#     :param int min_sample: minimum number of each class (overal) to be present in each fold for aggregation metric
#     :param int n_cv: number of folds to look for
#     :param list strings score_type: list of perf metrics to keep (depends on columns in df_res)
#     :return dtype: pandas df containing best HPs per performance metrics
    """
    
    if verbose: print(f"# Hyperparameter selection considered score types: {score_type}")
    
    # aggregate over HPs (does the quorum on class size filtering)
    aggr_df = aggregate_overall(metrics_df, min_samples=min_samples, stats='basic', n_cv=n_cv, score_type=score_type, verbose=verbose)

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
    
    dfm = metrics_df.melt(id_vars=hp_cols, value_name='value', var_name='score_type')
    dfm = dfm.loc[dfm['score_type'].isin(score_type)]
    dfm['value'] = dfm['value'].astype(float)
    
    
    return dfm



def extract_best_hp_records(perf_metrics, wanted_metrics):
    """From a HP search results df , extracts the rows corresponding to the best HP given a score type. The best HP is selected with an aggregate_overall() [mean]
#      :param pandas perf_metrics containing the HP search results return from perf_from_json()
#      :str wanted_metrics: the score type name to use for HP selection
    """

    best_hps = find_best_hyperparam(perf_metrics, min_samples=5, score_type=[wanted_metrics])
    
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



def delta_to_baseline(top_baseline, list_top_perfs):
    """ Computes the delta-to-baseline of a list of models to compare. The performance data frames are like what perf_from_json returns
#       :param pandas df top_baseline, all task performance of the baseline model (assumed to correspond to the best HP)    
#       :param list of pandas df top_perf: a list of performance data frames corresponding to models to comapre 
#       :return pandas df containing the performance deltas
    """
    deltas = []

    col2keep = ['task', 'fold_va', 'input_assay_id', 'model_name', 'roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa', 'num_pos', 'num_neg']
    top_baseline_scores = top_baseline.drop([x for x in top_baseline.columns if x not in col2keep],axis=1)

    for top_perf in list_top_perfs:
    
        top_perf_scores = top_perf.drop([x for x in top_perf.columns if x not in col2keep],axis=1)
        model_name = top_perf['model_name'].iloc[0]
    
        #print(model_name)
    
        merged = top_baseline_scores.merge(top_perf_scores, on=['task', 'fold_va', 'input_assay_id',  'num_pos', 'num_neg'], suffixes=('', '_'+model_name))
    
        # calculate the delta for each score now
        for s in ['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score','kappa']:
            merged[s+'_delta'] = merged[s+'_'+model_name] - merged[s]

        d = merged[['task', 'fold_va', 'input_assay_id','roc_auc_score_delta', 'auc_pr_delta', 'avg_prec_score_delta', 'max_f1_score_delta','kappa_delta', 'num_pos', 'num_neg']].copy()
        d['model_name'] = model_name
        deltas.append(d)    
    
    return pd.concat(deltas, sort=False)
    

def pointplot_fold_perf(metrics_df, 
                        score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score', 'kappa'],
                        figsize = (20, 18), 
                        n_cv=5, 
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
    
    # filter performance applying the quorumn filter
    perf_to_consider  = quorum_filter(metrics_df, min_samples=5, n_cv=n_cv, verbose=True)
    
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
        mini=perf_data[score_name].mean() - perf_data[score_name].std()
        maxi=perf_data[score_name].mean() + perf_data[score_name].std()
        #axes[i].set_ylim(mini-mini/10,maxi+maxi/10)
        axes[i].set_ylim(0,1)
        axes[i].set_title(score_name)

        # add a vertical line separation
        for k in range(len(x_order)):
            axes[i].axvline(k+0.5, 0,1, color="grey", alpha=0.5)

        i+=1

    return
    
    


def swarmplot_fold_perf(metrics_df, 
                        score_type=['roc_auc_score', 'auc_pr', 'avg_prec_score', 'max_f1_score', 'kappa'],
                        figsize = (20, 18), 
                        n_cv=5, 
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
    perf_fold  = aggregate_fold_perf(metrics_df, 5, stats='basic', n_cv=n_cv, score_type=score_type)
    
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
    




def match_best_tasks(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', n_cv=5, hp_selection_metric='auc_pr'): 

    hp_bests = list()
    for e in ((results_dir_x, t3_mapped_x),(results_dir_y, t3_mapped_y)):
        df_t3_mapped = pd.read_csv(e[1])
        main_tasks = df_t3_mapped.loc[df_t3_mapped['assay_type']!=aux_assay_type].cont_classification_task_id.dropna().values
        json_df = perf_from_json(e[0],aggregate=False,tasks_for_eval=main_tasks, n_cv=n_cv) 
        json_melted = melt_perf(json_df, score_type=[hp_selection_metric])
        hp_best = best_hyperparam(json_melted)
        # keeping only the records associated to the best hyperparameters
        hp_cols = ['hp_'+hp for hp in get_sc_hps()]
        hp_best = pd.merge(json_df[~json_df[hp_selection_metric].isna()],hp_best,how='inner',on=hp_cols)
        hp_best = pd.merge(df_t3_mapped,hp_best,how='inner',left_on='cont_classification_task_id',right_on='task')
        hp_bests.append(hp_best)

    return hp_bests

def statistical_model_comparison_analysis(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', n_cv=5, min_samples=25):
    """ Run a statistical significance analysis between two runs, both with hyperparameter optimization
      Best hyperparameters will be selected based on the auc_pr
      For now, only compatible with the json result format . 
      Result is the difference between the second (y) and the first (x) arguments
#    :param  str path to the results folder of the 1st run, containing the json files
#    :param  str path to the mapped T3 file of the 1st run
#    :param  str axis label for the plot, designating the 1st run
#    :param  str path to the results folder of the 2nd run, containing the json files
#    :param  str path to the mapped T3 file of the 2nd run
#    :param  str axis label for the plot, designating the 2nd run
#    :param int n_cv: specify the number of folds used for cross valid, starting with 0, higher fold_va numbers will be dropped
#    :param int n_cv: statistics calculations will be limited to the tasks which have at least this number of positives and negatives
#    :return None 
    """ 

    hp_bests = match_best_tasks(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', n_cv=n_cv)
    # matching x and y runs
    df_merge = pd.merge(hp_bests[0], hp_bests[1],how='inner',on=['input_assay_id','threshold_value','fold_va'])

    # plotting
    # statistical analysis only valid for aggregated figures over folds - hence groupby
    table = df_merge.groupby(by=['input_assay_id']).mean().reset_index() # supposedly, no threshold_value needed here in the gb-clause
    metric_x = 'roc_auc_score_x'
    metric_y = 'roc_auc_score_y'
    x_lim = 0.5
    y_lim = 0.5
    title = 'AUC ROC'

    res_wide = summarize_diff_statistics(hp_bests[0],hp_bests[1],min_samples=min_samples, n_cv=n_cv)
    table = table[~table['roc_auc_score_y'].isna()]
    table = table[~table['roc_auc_score_x'].isna()]
    res_stat_sign = plot_statisical_significance(table, metric_x, metric_y, label_x, label_y, x_lim, y_lim, title )
    res = pd.concat([res_stat_sign, res_wide], axis=1)

    return res 

def summarize_diff_statistics(tasks_x, tasks_y, min_samples=25, n_cv=5): 
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
        ress.append(aggregate_overall(t,min_samples=min_samples, n_cv=n_cv, stats='full'))
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

    return res






# MTX manipulatiosn such as splitting per fold

def split_folds(M, fold_vector):
    """ Splits global M (i.e. Y or X) into folds Ms
#     :param  scipy.sparse.csr.csr_matrix M
#     :param  np.array fold_vector: containing folds assignment for each row.
#     :return list of size=n_folds where each element is a csr_matrix representing a fold
    """
    assert type(M) == scipy.sparse.csr.csr_matrix, "M needs to be scipy.sparse.csr.csr_matrix"
    assert folds.shape[0] == M.shape[0], "fold_vector must have same shape[0] than M"
    
    folds = [M[fold_vector==f,:] for f in np.unique(folds)]
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




# ==========================================================================
# ==========================================================================
# === old sparsechem utils: not recommanded


def perf_from_metrics(result_dir, tasks_for_eval=None, model_name='Y', 
                      prefix=None, n_cv=5, verify=False, 
                      hidden_sizes=None,
                      last_dropout=None,
                      learning_rate=None,
                      learning_steps=None,
                      weight_decay=None,
                      epochs=None,
                      fold_va=None,
                      fold_te=None,
                      verbose=False):
    """ Collects the performance from the results/*-metrics.csv files. This metrics file contain positive/negtive counts.
    Assumes filename have the default naming convention (i.e. --filename was not used). 
#     :param string result_dir: path to the model folder containing the conf.npy files
#     :param np.array (integers like) tasks_for_eval: tasks to consider for evaluation (default=None)
#     :param bool aggrgate: if True, aggregate results over folds (considering all tasks verifying MIN_SAMPLES)
#     :param string model_name: adds a name in a column to resulting dataframe (default=Y)
#     :param string prefix: used as a filter, if specified only files starting with prefix will be considered
#     :param int n_cv: specify the number of folds used for cross valid, will spit a warning if a setting has not n_cv perf reports
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :param string hidden_sizes: sets a filter on hidden layer size on filename
#     :param string last_dropout: sets a filter on last_dropout  on filename
#     :param string learning_rate: sets a filter on learning_rate on filename
#     :param string learning_steps: sets a filter on learning_steps on filename
#     :param string weight_decay: sets a filter on weight_decay on filename
#     :param string epochs: sets a filter on epochs on filename
#     :param string fold_va: sets a filter on fold_va on filename
#     :param string fold_te: sets a filter on fold_te on filename
#     :param bool verbose: speaks by itself
#     :return dtype: pandas df containing performance metrics files appended to each other and including some HPs
    """    
    
    assert os.path.isdir(result_dir), f"Can't find results directory at {result_dir}"
    
    if prefix is not None: prefix_len = len(prefix)
    
    dataframes = []
    for f in os.listdir(result_dir):
        
        if f[:3]!='sc_':continue
        
        # skip files without prefix if specified 
        if prefix is not None and f[:prefix_len]!=prefix:continue
        
        # skip hyperparameters if requested
        if hidden_sizes is not None and f'_h{hidden_sizes}_' not in f:continue
        if last_dropout is not None and f'_ldo{last_dropout}_' not in f:continue
        if learning_rate is not None and f'_lr{learning_rate}_' not in f:continue
        if learning_steps is not None and f'_lrsteps{learning_steps}_' not in f:continue
        if weight_decay is not None and f'_wd{weight_decay}_' not in f:continue
        if epochs is not None and f'_ep{epochs}_' not in f:continue
        if fold_va is not None and f'_fva{fold_va}_' not in f:continue
        if fold_te is not None and f'_fte{fold_te}-metrics' not in f:continue
        
        
        df = pd.read_csv(os.path.join(result_dir, f))
        
        if tasks_for_eval is not None: df = df.loc[df['task'].isin(tasks_for_eval)].copy()
        
        # capture HP from file name
        # TO DO : capture all possible settings from file name?
        # or request printing out HP settings in metrics dataframe from sparsechem
        df['hp_hidden_sizes']   = re.search('_h(.+?)_', f).group(1)
        df['hp_last_dropout']   = re.search('_ldo(.+?)_', f).group(1)
        df['hp_weight_decay']   = re.search('_wd(.+?)_', f).group(1)
        df['hp_learning_rate']  = re.search('_lr(.+?)_', f).group(1)
        df['hp_learning_steps'] = re.search('_lrsteps(.+?)_', f).group(1)
        df['hp_epochs']         = re.search('_ep(.+?)_', f).group(1)
        df['fold_va']        = re.search('_fva(.+?)_', f).group(1)
        df['fold_va']        = df['fold_va'].astype(np.int)
        df['fold_test']      = re.search('_fte(.+?)-metrics', f).group(1)

        dataframes.append(df)
    
    if verbose:print(f"Loaded {len(dataframes)} metrics files")
    
    metrics_df = pd.DataFrame()

    if len(dataframes)>0: 
        metrics_df = metrics_df.append(dataframes)
        metrics_df['model'] = model_name
    
    # check that every settings have been run over the n_cv 
    if verify:verify_cv_runs(metrics_df, n_cv=n_cv)

    return metrics_df.reset_index(drop=True)



def perf_from_conf(model_dir, tasks_for_eval=None, aggregate=False, model_name='Y'):
    """ Collects the performance from thje models/*-conf.npy files. Useful for HP search because it includes HPs details.
#     :param string model_dir: path to the model folder containing the conf.npy files
#     :param np.array (integers like) tasks_for_eval: tasks to consider for evaluation (default=None)
#     :param bool aggrgate: if True, uses the aggregate results (considering all tasks verifying MIN_SAMPLES)
#     :param string model_name: adds a name in a column to resulting dataframe (default=Y)
#     :return dtype: pandas df containing performance summaries from conf file (including HP: epoch, valid fold, hidden sizes, lr steps ...)
    """
    
    assert os.path.isdir(model_dir), f"Can't find models directory at {model_dir}"
    assert len([x for x in os.listdir(model_dir) if os.path.splitext(x)[1] == '.npy'])>0, f"Did not find *conf.npy in {model_dir}"
    
    if tasks_for_eval is not None and aggregate:
        print("tasks_for_eval will not be considered. Turn off aggregate to consider a subset of tasks")
    
    if aggregate: 
        df_res = perf_from_conf_aggregate(model_dir, model_name=model_name)
    else:
        df_res = perf_from_conf_individual(model_dir, tasks_for_eval, model_name=model_name)
    
    return df_res




def perf_from_conf_aggregate(model_dir, model_name='Y'):
    """ Collects the performance from thje models/*-conf.npy files using the aggregate results (considering min_samples). 
    Useful for HP search because it includes HPs details.
#     :param string model_dir: path to the model folder containing the conf.npy files
#     :param string model_name: adds a name in a column to resulting dataframe (default=Y)
#     :return dtype: pandas df containing performance summaries from conf file (including HP: epoch, valid fold, hidden sizes, lr steps)
    """    
    data = []

    for f in os.listdir(model_dir):
        if f[-4:]!='.npy':continue
        r = np.load(os.path.join(model_dir,f), allow_pickle=True).item()
        auc_roc_va = r["results_agg"]["va"]["roc_auc_score"]
        auc_pr_va  = r["results_agg"]["va"]["auc_pr"]
        
        #kappa      = r["results_agg"]["va"]["kappa"]
        #max_f1     = r["results_agg"]["va"]["max_f1_score"]
        
        epoch_time_tr = r["results_agg"]["tr"]["epoch_time"]
        hidden_sizes  = ",".join([str(x) for x in r["conf"].hidden_sizes])
        learning_rate = r["conf"].lr
        dropout       = r["conf"].last_dropout
        epochs        = r["conf"].epochs
        weight_decay =  r["conf"].weight_decay
        fold_va       = r["conf"].fold_va
        fold_te       = r["conf"].fold_te
        lr_steps      = ",".join([str(x) for x in r["conf"].lr_steps])
        min_samples   = r['conf'].min_samples_auc
        data.append([fold_te, fold_va, epochs, hidden_sizes, dropout, weight_decay, learning_rate, lr_steps, min_samples, auc_roc_va, auc_pr_va, epoch_time_tr])
        
        #data.append([fold_te, fold_va, epochs, hidden_sizes, dropout, weight_decay, learning_rate, lr_steps, min_samples, auc_roc_va, auc_pr_va, max_f1, kappa, epoch_time_tr])
        
    #df_res = pd.DataFrame(data, columns=['fold_te','fold_va', 'hp_epochs', 'hp_hidden_sizes', 'hp_last_dropout', 'hp_weight_decay', 'hp_learning_rate','hp_learning_steps', 'min_samples', 'auc_va_mean', 'auc_pr_va_mean', 'max_f1_va_mean', 'kappa_va_mean', 'train_time_1epochs'])
    
    df_res = pd.DataFrame(data, columns=['fold_te','fold_va', 'hp_epochs', 'hp_hidden_sizes', 'hp_last_dropout', 'hp_weight_decay', 'hp_learning_rate','hp_learning_steps', 'min_samples', 'auc_va_mean', 'auc_pr_va_mean', 'train_time_1epochs'])
    df_res['model'] = model_name
    
    return df_res



def perf_from_conf_individual(model_dir, tasks_for_eval, model_name='Y'):
    """ Collects the performance from thje models/*-conf.npy files using the individual performance numbers. Useful for HP search because it includes HPs details.
#     :param string model_dir: path to the model folder containing the conf.npy files
#     :param np.array like (integers like) tasks_for_eval: tasks to consider for evaluation (user's responsability)
#     :param string model_name: adds a name in a column to resulting dataframe (default=Y)
#     :return dtype: pandas df containing performance summaries from conf file (including HP: epoch, valid fold, hidden sizes, lr steps)
    """    
        
    data = []
    for f in os.listdir(model_dir):
        if f[-4:]!='.npy':continue
        r = np.load(os.path.join(model_dir,f), allow_pickle=True).item()
    
        if tasks_for_eval is None: 
            auc_roc_va = r["results"]["va"]['auc_roc']
            auc_pr_va  = r["results"]["va"]["auc_pr"]
            n_tasks = r["results"]["va"]["auc_pr"].shape[0]
        else: 
            # only pick the wanted tasks 
            auc_roc_va = r["results"]["va"]['auc_roc'][tasks_for_eval]
            auc_pr_va  = r["results"]["va"]["auc_pr"][tasks_for_eval]
            n_tasks = r["results"]["va"]["auc_pr"][tasks_for_eval].shape[0]
    
        hidden_sizes  = ",".join([str(x) for x in r["conf"].hidden_sizes])
        dropout       = r["conf"].last_dropout
        learning_rate = r["conf"].lr
        epochs        = r["conf"].epochs
        weight_decay =  r["conf"].weight_decay
        fold_va       = r["conf"].fold_va
        fold_te       = r["conf"].fold_te
        lr_steps      = ",".join([str(x) for x in r["conf"].lr_steps])
        min_samples   = r['conf'].min_samples_auc
        for i in auc_roc_va.index:
            data.append([i, fold_te, fold_va, epochs, hidden_sizes, dropout, weight_decay, learning_rate, lr_steps, n_tasks, min_samples, auc_roc_va[i], auc_pr_va[i]])
    
    df_res = pd.DataFrame(data, columns=['task','fold_te', 'fold_va', 'hp_epochs', 'hp_hidden_sizes', 'hp_last_dropout', 'hp_weight_decay', 'hp_learning_rate','hp_learning_steps', 'n_tasks_eval', 'min_samples', 'auc_va', 'auc_pr_va'])
    
    df_res['model'] = model_name
    
    return df_res

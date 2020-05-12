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

# function template
def template():
    """ description
#     :param dtype name: description
#     :param dtype name: description
#     :return dtype: description
    """
    
    return

# TO DO : 

# - split Y and X by test set for eval
# - delta predictions perf between two (or more) models

# better manage hyperparameters (consistent to the code)

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


def verify_cv_runs(metrics_df, n_cv=5):
    """ From the metrics dataframe yielded by perf_from_metrics(), cehcks if each hyperparameter was run n_cv
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_metrics() 
#     :param int n_cv: number of folds to look for
#     :return void: prints message in stdout
    """    
    hp = [x for x in metrics_df.columns if x[:3] == 'hp_']
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_"

    aggr = metrics_df.sort_values('fold_va').groupby(hp)['fold_va'].apply(lambda x: ','.join([str(x) for x in x.unique()]))
        
     # if missing ones, print out a warning message
    folds = ",".join([str(x) for x in range(n_cv)])
    if aggr[~aggr.fold_va.str.contains(folds)].shape[0]>0:
        print("WARNING: missing fold runs")
        print(f"Fold runs found :\n {aggr}")
    
    return 

def aggregate_fold_perf(metrics_df, min_samples, n_cv=5,  verify=True):
    """ HP performance aggregation over folds. 
    From the metrics dataframe yielded by perf_from_metrics(), does the aggregation over the fold (mean, std) results in one perf per fold.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_metrics() 
#     :param int min_sample: minimum number of each class (overal) to be considered in mean
#     :param int n_cv: number of folds to look for
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :return dtype: pandas df containing performance per task aggregated over each fold
    """    

    hp = [x for x in metrics_df.columns if x[:3] == 'hp_']
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_"
    assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    assert 'fold_va' in metrics_df.columns, "metrics dataframe must contain fold_va column"
    
    hp.append('fold_va')
    
    if verify:verify_cv_runs(metrics_df, n_cv=n_cv)

    metrics2consider_df = metrics_df.loc[(metrics_df['num_pos']>=min_samples)&(metrics_df['num_neg']>=min_samples)].copy() 

    # drop a few columns which are not relevant in this aggrgation
    cols2drop = [x for x in metrics2consider_df.columns if 'num_pos' in x]
    [cols2drop.append(x) for x in metrics2consider_df.columns if 'num_neg' in x]
    cols2drop.append('task')
    metrics2consider_df.drop(cols2drop, axis=1,inplace=True)    
    
    # do the mean aggregation
    aggr_mean = metrics2consider_df.groupby(hp).mean()
    aggr_mean.columns = [x+'_mean' for x in aggr_mean.columns]

    # do the median aggregation
    aggr_med = metrics2consider_df.groupby(hp).median()
    aggr_med.columns = [x+'_median' for x in aggr_med.columns]
    
    # do the stdev aggregation
    aggr_std = metrics2consider_df.groupby(hp).std()
    aggr_std.columns = [x+'_stdev' for x in aggr_std.columns]

    # do the skew aggregation
    aggr_skew = metrics2consider_df.groupby(hp).skew()
    aggr_skew.columns = [x+'_skewness' for x in aggr_skew.columns]

    # do the kurtosis aggregation
    aggr_kurt = metrics2consider_df.groupby(hp).apply(pd.DataFrame.kurt)
    aggr_kurt.columns = [x+'_kurtosis' for x in aggr_kurt.columns]
    

    return aggr_mean.join(aggr_med).join(aggr_std).join(aggr_skew).join(aggr_kurt).reset_index()
      

def aggregate_task_perf(metrics_df, min_samples, n_cv=5,  verify=True):
    """ HP performance aggregation over tasks. 
    From the metrics dataframe yielded by perf_from_metrics(), does the aggregation over the CV (mean, std) results in one perf per task.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_metrics() 
#     :param int min_sample: minimum number of each class (overal) to be considered in mean
#     :param int n_cv: number of folds to look for
#     :param bool verify: checks for missing folds runs in CV and prints a report if missing jobs
#     :return dtype: pandas df containing performance per task aggregated over CV
    """    

    hp = [x for x in metrics_df.columns if x[:3] == 'hp_']
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_"
    assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    
    hp.append('task')
    hp.append('num_pos')
    hp.append('num_neg')
    
    if verify:verify_cv_runs(metrics_df, n_cv=n_cv)

    metrics2consider_df = metrics_df.loc[(metrics_df['num_pos']>=min_samples)&(metrics_df['num_neg']>=min_samples)]
    
    # do the mean aggregation
    aggr_mean = metrics2consider_df.groupby(hp).mean()
    aggr_mean.columns = [x+'_mean' for x in aggr_mean.columns]

    # do the median aggregation
    aggr_med = metrics2consider_df.groupby(hp).median()
    aggr_med.columns = [x+'_median' for x in aggr_med.columns]
    
    # do the stdev aggregation
    aggr_std = metrics2consider_df.groupby(hp).std()
    aggr_std.columns = [x+'_stdev' for x in aggr_std.columns]

    # do the skew aggregation
    aggr_skew = metrics2consider_df.groupby(hp).skew()
    aggr_skew.columns = [x+'_skewness' for x in aggr_skew.columns]

    # do the kurtosis aggregation
    aggr_kurt = metrics2consider_df.groupby(hp).apply(pd.DataFrame.kurt)
    aggr_kurt.columns = [x+'_kurtosis' for x in aggr_kurt.columns]
    

    return aggr_mean.join(aggr_med).join(aggr_std).join(aggr_skew).reset_index()  .join(aggr_kurt)
    

    
def aggregate_overall(metrics_df, min_samples):
    """ HP performance aggregation overall . 
    From the metrics dataframe yielded by perf_from_metrics(), does the aggregation over the CV (mean, std) results in one perf hyperparameter.
#     :param pandas df metrics_df: metrics dataframe yielded by perf_from_metrics() 
#     :param int min_sample: minimum number of each class (overal) to be considered in mean
#     :return dtype: pandas df containing performance per hyperparameter setting
    """        
    hp = [x for x in metrics_df.columns if x[:3] == 'hp_']
    assert len(hp) > 0, "metrics dataframe must contain hyperparameter columns starting with hp_"
    assert 'num_pos' in metrics_df.columns, "metrics dataframe must contain num_pos column"
    assert 'num_neg' in metrics_df.columns, "metrics dataframe must contain num_neg column"
    
    metrics2consider_df = metrics_df.loc[(metrics_df['num_pos']>=min_samples)&(metrics_df['num_neg']>=min_samples)].copy() 
    
    # drop a few columns which are not relevant in this aggrgation
    cols2drop = [x for x in metrics2consider_df.columns if 'num_pos' in x]
    [cols2drop.append(x) for x in metrics2consider_df.columns if 'num_neg' in x]
    cols2drop.append('task')
    metrics2consider_df.drop(cols2drop, axis=1,inplace=True)
    
    # do the mean aggregation
    aggr_mean = metrics2consider_df.groupby(hp).mean()
    aggr_mean.columns = [x+'_mean' for x in aggr_mean.columns]    
    
    # do the median aggregation
    aggr_med = metrics2consider_df.groupby(hp).median()
    aggr_med.columns = [x+'_median' for x in aggr_med.columns]
    
    # do the stdev aggregation
    aggr_std = metrics2consider_df.groupby(hp).std()
    aggr_std.columns = [x+'_stdev' for x in aggr_std.columns]

    # do the skew aggregation
    aggr_skew = metrics2consider_df.groupby(hp).skew()
    aggr_skew.columns = [x+'_skewness' for x in aggr_skew.columns]

    # do the kurtosis aggregation
    aggr_kurt = metrics2consider_df.groupby(hp).apply(pd.DataFrame.kurt)
    aggr_kurt.columns = [x+'_kurtosis' for x in aggr_kurt.columns]
    
    return aggr_mean.join(aggr_med).join(aggr_std).join(aggr_skew).join(aggr_kurt).reset_index()  
    

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



def perf_from_json(
        model_dir_or_file, 
        tasks_for_eval=None, 
        aggregate=False, 
        evaluation_set='va',
        model_name='Y',
        n_cv=5,
    ):
    """ Collects the performance from thje models/*.json files containing both the model configuration and performance.
      Useful for HP search because it includes HPs details.
#     :param string model_dir: path to the model folder containing the .json files
#     :param np.array (integers like) tasks_for_eval: tasks to consider for evaluation (default=None)
#     :param bool aggrgate: if True, uses the aggregate result from sparsechem (considering all tasks verifying MIN_SAMPLES).
#     :param string model_name: adds a name in a column to resulting dataframe (default=Y)
#     :return pandas df containing performance and configuration summaries 
#     :param int n_cv: specify the number of folds used for cross valid, starting with 0, higher fold_va numbers will be dropped
    """
    res_all = []

    if os.path.isdir(os.path.join(model_dir_or_file)):     
        files = [os.path.join(model_dir_or_file,f) for f in os.listdir(model_dir_or_file) if os.path.isfile(os.path.join(model_dir_or_file,f))]
    elif os.path.isfile(os.path.join(model_dir_or_file)):     
        files = [model_dir_or_file]

    for f in tqdm(files):
        if not f.endswith(".json"):# or not os.path.basename(f).startswith("sc_"):
            print(f"{f} is not a sparsechem json, hence skipped.")
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
 
    output_df = pd.concat(res_all)

    # filter out some folds 
    cvfolds = list(range(n_cv))
    output_df = output_df.query('fold_va in @cvfolds')

    return output_df


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
         'task_weights', # importantly, consider the task weights as hyperparameter
    ]
    return hps


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


def melt_perf(df_res, perf_metrics=['auc_pr_va','auc_va']):
    """ Melts (or unpivot) the performance dataframe resuting from perf_from_conf(). 
#     :param pandas df_res: dataframe containing results as provided by perf_from_conf()
#     :param list strings perf_metrics: list of perf metrics to keep (depends on columns in df_res)
#     :return dtype: pandas df containing performance in melted format usefull for R ggplot2
    """
    
    hp_cols = [x for x in df_res.columns if x[:3]=='hp_']
    assert len(hp_cols) > 0, 'No hyperparamters found in dataframe, use hp_* prefix for hyperparameters columns'
    hp_cols.append('fold_va')
    
    for p in perf_metrics: 
        assert p in df_res.columns, f'"{p}" column not found in df_res'
    
    dfm = df_res.melt(id_vars=hp_cols, value_name='value', var_name='score_type')
    dfm = dfm.loc[dfm['score_type'].isin(perf_metrics)]
    dfm['value'] = dfm['value'].astype(float)
    
    return dfm

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


def best_hyperparam(dfm):
    """ Gets the best hyperparameters for each performance metrics from dfm resulting from melt_perf(). 
#     :param pandas dfm: dataframe containing results as provided by melt_perf()
#     :return dtype: pandas df containing best HPs per performance metrics
    """
    agg_df = all_hyperparam(dfm)
    best_hps = agg_df.iloc[agg_df.groupby(['score_type']).idxmax()['value'].values]    
    return best_hps


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


def match_best_tasks(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', n_cv=5): 

    hp_bests = list()
    hp_selection_metric='auc_pr'
    for e in ((results_dir_x, t3_mapped_x),(results_dir_y, t3_mapped_y)):
        df_t3_mapped = pd.read_csv(e[1])
        main_tasks = df_t3_mapped.loc[df_t3_mapped['assay_type']!=aux_assay_type].cont_classification_task_id.dropna().values
        json_df = perf_from_json(e[0],aggregate=False,tasks_for_eval=main_tasks, n_cv=n_cv) 
        json_melted = melt_perf(json_df, perf_metrics=[hp_selection_metric])
        hp_best = best_hyperparam(json_melted)
        hp_best = hp_best.drop(columns=['fold_va']) # are grouped-by (averaged)
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

    hp_bests = match_best_tasks(results_dir_x, t3_mapped_x, label_x, results_dir_y, t3_mapped_y, label_y, aux_assay_type='Yx', n_cv=5)
    # matching x and y runs
    df_merge = pd.merge(hp_bests[0], hp_bests[1],how='inner',on=['input_assay_id','fold_va'])

    # plotting
    # statistical analysis only valid for aggregated figures over folds - hence groupby
    table = df_merge.groupby(by=['input_assay_id']).mean().reset_index() # supposedly, no threshold_value needed here in the gb-clause
    metric_x = 'roc_auc_score_x'
    metric_y = 'roc_auc_score_y'
    x_lim = 0.5
    y_lim = 0.5
    title = 'AUC ROC'

    res_wide = summarize_diff_statistics(hp_bests[0],hp_bests[1],min_samples=min_samples)
    table = table[~table['roc_auc_score_y'].isna()]
    table = table[~table['roc_auc_score_x'].isna()]
    res_stat_sign = plot_statisical_significance(table, metric_x, metric_y, label_x, label_y, x_lim, y_lim, title )
    res = pd.concat([res_stat_sign, res_wide], axis=1)

    return res 

def summarize_diff_statistics(tasks_x, tasks_y, min_samples=25): 
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
        ress.append(aggregate_overall(t,min_samples=min_samples))
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



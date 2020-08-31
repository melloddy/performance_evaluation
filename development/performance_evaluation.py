import os
import argparse
import torch
import json 
import warnings

import pandas as pd
import numpy as np
import sklearn.metrics
import scipy.sparse as sparse
from scipy.sparse import csc_matrix

from rdkit.Chem import DataStructs, AllChem, MolFromSmiles
from rdkit.Chem.Fingerprints import FingerprintMols

from pathlib import Path

from VennABERS import ScoresToMultiProbs

seed = 42 
np.random.seed(seed)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, default="T10_y.npy")
parser.add_argument("--y_pred_single", help="Yhat prediction output from single-pharma run (./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred) or (<single pharma dir>/y_hat.npy)", type=str, required=True)
parser.add_argument("--y_pred_multi", help="Yhat prediction output from multi-pharma run (./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/pred) or (<multi pharma dir>/y_hat.npy)", type=str, required=True)
parser.add_argument("--folding_test", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str)
parser.add_argument("--folding_train_single", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str)
parser.add_argument("--folding_train_multi", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str)
parser.add_argument("--task_weights", help="CSV file with columns task_id and weight (i.e.  files_4_ml/T9_red.csv)", type=str, default=None)
parser.add_argument("--single_performance_report", help="JSON file with global reported single-pharma performance (i.e. ./Single-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=True)
parser.add_argument("--multi_performance_report", help="JSON file with global reported multi-pharma performance (i.e. ./Multi-pharma-run/substra/medias/subtuple/<pharma_hash>/pred/perf.json)", type=str, default=None, required=True)
parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
parser.add_argument("--task_map_single", help="Taskmap from MELLODDY_tuner output of single run (results/weight_table_T3_mapped.csv)", default = None)
parser.add_argument("--task_map_multi", help="Taskmap from MELLODDY_tuner output of multi run (results/weight_table_T3_mapped.csv)", default = None)
parser.add_argument("--task_map_pred", help="Taskmap from MELLODDY_tuner output of pred run (results/weight_table_T3_mapped.csv)", default = None)
parser.add_argument("--T2_train_single", help="T2 file from training cluster)", default = None, required=True)
parser.add_argument("--T2_train_multi", help="T2 file from training cluster)", default = None, required=True)
parser.add_argument("--T2_pred", help="T2 file from pred cluster)", default = None, required=True)
parser.add_argument("--T4_train_single", help="T4 file from training cluster)", default = None, required=True)
parser.add_argument("--T4_train_multi", help="T4 file from training cluster)", default = None, required=True)
parser.add_argument("--T5_pred", help="T5 file from pred cluster)", default = None, required=True)
parser.add_argument("--T11_pred", help="T11 file from pred cluster)", default = None, required=True)
parser.add_argument("--train_preds_single", help="yhat file from single training cluster)", default = None, required=True)
parser.add_argument("--train_preds_multi", help="yhat file from multi training clusters)", default = None, required=True)
parser.add_argument("--T10_train_single", help="T3 file from multi training cluster)", default = None, required=True)
parser.add_argument("--T10_train_multi", help="T3 file from multi training clusters)", default = None, required=True)
parser.add_argument("--T11_train_single", help="T11 file from train cluster)", default = None, required=True)
parser.add_argument("--T11_train_multi", help="T11 file from train cluster)", default = None, required=True)
parser.add_argument("--T5_train_single", help="T5 file from pred cluster)", default = None, required=True)
parser.add_argument("--T5_train_multi", help="T5 file from pred cluster)", default = None, required=True)
parser.add_argument("--T3_pred", help="complete T3 file from MELLODDY_splits folder", default = None, required=True)

args = parser.parse_args()


def vprint(s=""):
    if args.verbose:
        print(s)     
vprint(args)


print("initial checks and parsing args")

assert pd.__version__[:4] >='0.25', "Pandas version must be >=0.25"

y_pred_single_path = Path(args.y_pred_single)
y_pred_multi_path = Path(args.y_pred_multi)

print(args.task_map_multi)
print(args.task_map_pred)

# if task map pred is not provided, it defaults to the single task map, since they are asusmed to be same
if not(args.task_map_pred):
    task_map_pred = args.task_map_single
else:
    task_map_pred = args.task_map_pred

#load the folding/true data
folding = np.load(Path(args.folding_test))


def calculate_FP(smiles):
    m1 = MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m1,2, nBits=2048)
    bitstring = list(map(int,list(fp.ToBitString())))
    return bitstring

def read_data(path):
    df = pd.read_csv(path) #for testing restrict to first k values
    return df.canonical_smiles


def get_FPs(df):
    """ for a dataframe remove records with invalid smiles and return this dataframe and corresponding fingerprint vector
    """
    fps=[]
    df_after=pd.DataFrame()
    
    smiles_list= df.smiles
    
    for ids, val in enumerate(smiles_list):
        #calculate FPs (and preprocess/remove unwanted chemistry/standardise the molecules)
        try:
            bitstring=calculate_FP(val)
            fps.append(bitstring)
            df_after = df_after.append(df.iloc[ids])
            #threshold the subset of the data using 10um (activity threshold)
            # y.append(1 if df["Standard Value"].iloc[ids] >= 10 else 0)
        except: pass 
    return fps, df_after


def get_tuples(col_id, col, compounds_vec, true_df, pred_df):
# get all compounds and metrics for the current task
    tuples = []
    for row_id, item in col.iteritems():
        # if there is a prediction in the cell
        if item != 0:
            curr_comp = compounds_vec[row_id]
            curr_task = col_id
            
            if true_df.iloc[row_id][col_id] == -1:
                true_label = 0
            else:
                true_label = 1
            
            pred = pred_df.iloc[row_id][col_id]
            
            
            tuples.append((curr_comp, curr_task, pred, true_label))
    return tuples

def lookup_smile(T2_df, comp_id):
    smile = T2_df["smiles"][T2_df["input_compound_id"]==comp_id]
    return smile
                    
"""
# data prep - Fingerprints and similarity matrix
"""

def get_datastructs_from_smiles(df):
    fps, df_after = get_FPs(df)
    fps = np.array(fps,dtype=np.uint8)
    
    bitstrings = ["".join(fps[idx].astype(str)) for idx in range(len(list(fps)))]
    structs = [DataStructs.cDataStructs.CreateFromBitString(i) for i in bitstrings]
    return structs, df_after

def mask_matrix(true_data, pred_data):
    ''' 
    gets 2 matrices of same shape <true_data> and <pred_data> 
    and masks <pred_data> according to <true_data>
    '''
    assert true_data.shape == pred_data.shape, "Shapes of both matrices need to be same"
    true_data = pd.DataFrame(true_data)
    pred_data = pd.DataFrame(pred_data)
    pred_data = pred_data.mask(true_data == 0, 0)
    return pred_data

def get_tasks_inner_join(task_map_pred,                          
                         folding,
                         args,
                         validation=True):
    """
    This method preprocesses the prediction files from the SP and MP and the true label matrix.
    It filters out all tasks that the three files do not have in common, 
    yielding only the inner join tasks of all three.
    
    In:
        - single multi and pred raw data paths and repsective task map paths
    Do:
        - get inner join taks (tasks, all clusters have in common) of single, multi and prediction cluster
    Out:
        - return the raw data tables with only the inner join tasks
    """
    
    vprint("Start retrieving common tasks from all clusters...")
    
    single_pred_path   = args.y_pred_single
    multi_pred_path    = args.y_pred_multi
    true_path          = args.y_true_all
    task_map_single    = args.task_map_single
    task_map_multi     = args.task_map_multi
    task_weights       = args.task_weights

    
    ## filtering out validation fold
    fold_va = 0
    
    # get both tasks and pred 
    # load the data
    true_data = np.load(true_path, allow_pickle = True)
    true_data = true_data.tolist()    
    true_data = true_data.tocsr()

    single_pred = np.load(single_pred_path, allow_pickle = True)
    multi_pred = np.load(multi_pred_path, allow_pickle = True)


    print(single_pred)

    print(true_data.shape)
    print(single_pred.shape)
    print(multi_pred.shape)
    
    # get compound ids
    """
    input_compound_id 
    ->[über results_tmp/descriptors/mapping_table_T5,csv] 
    ->  descriptor_vector_id -> [über results/T11.csv] 
    -> cont_descriptor_vector_id == "sparse matrix row id"
    """
    T5 = pd.read_csv(args.T5_pred)
    T11 = pd.read_csv(args.T11_pred)
      
    if validation: 
        true_data    = true_data[folding == fold_va] #only keeping validation fold
        single_pred  = single_pred[folding == fold_va] #only keeping validation fold
        multi_pred   = multi_pred[folding == fold_va] #only keeping validation fold
        T11          = T11[folding == fold_va]
    else:
        true_data    = true_data[folding != fold_va] #keeping all but validation fold
        single_pred  = single_pred[folding != fold_va] #keeping all but validation fold
        multi_pred   = multi_pred[folding != fold_va] #keeping all but validation fold
        T11          = T11[folding != fold_va]
    
    T115 =  T11.merge(T5, on = "descriptor_vector_id")   
    compounds_vec = []
    
    # for each unique cont_descriptor_vector_id get all compounds  from T115
    for cont_vec_id  in T115.cont_descriptor_vector_id.unique():
        compounds_vec.append(T115['input_compound_id'][T115['cont_descriptor_vector_id'] == cont_vec_id].tolist())
    
    true_data = true_data.todense()
    
    true_data = pd.DataFrame(true_data)
    single_pred = pd.DataFrame(single_pred)
    multi_pred = pd.DataFrame(multi_pred)
    
    '''
    #read task maps
    '''
    SP_tasks = pd.read_csv(task_map_single)
    MP_tasks = pd.read_csv(task_map_multi)
    Pred_tasks = pd.read_csv(task_map_pred)
    tw = pd.read_csv(task_weights)
    
    SP_tasks_rel = SP_tasks[["classification_task_id", "cont_classification_task_id"]] 
    MP_tasks_rel = MP_tasks[["classification_task_id", "cont_classification_task_id"]] 
    Pred_tasks_rel = Pred_tasks[["classification_task_id", "cont_classification_task_id"]] 
    
    global_index = pd.DataFrame({"id": range(MP_tasks.shape[0] + 1)})

    '''
    ## match single map
    ## match multi map
    ## match pred map
    '''
    df_s = global_index.merge(SP_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')
    df_m = global_index.merge(MP_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')        
    df_p = global_index.merge(Pred_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')
    

    '''    
    # get indices to be dropped 
    '''
    to_keep = []
    for i in range(len(df_s)):
        if(
            not(pd.isna(df_s.cont_classification_task_id[i])) and
            not(pd.isna(df_m.cont_classification_task_id[i])) and
            not(pd.isna(df_p.cont_classification_task_id[i]))
        ):
            to_keep.append(True)
        else:
            to_keep.append(False)

    print("inner join sum: " +str(sum(to_keep)))

    '''
    # get the to be dropped cont_classification_task_ids
    # remove na's to only get indices to be dropped
    '''
    
    df_keep = df_s.cont_classification_task_id.iloc[to_keep].dropna()
    
    '''
    #keep these indices from SP_tasks, MP_tasks, Pred_tasks
    # the indices are the actual task ids from each table we need to keep
    '''
    
    vprint("SP_tasks")
    vprint(SP_tasks)
    
    # get the inner tasks indices and their names
    def get_inner_task_indices_names(data, keep):
        return data[data["classification_task_id"].isin(keep.index)], \
            list(data["classification_task_id"][data["classification_task_id"].isin(keep.index)])
    
    SP_tasks_inner, SP_tasks_inner_names = get_inner_task_indices_names(SP_tasks, df_keep)    
    MP_tasks_inner, MP_tasks_inner_names = get_inner_task_indices_names(MP_tasks, df_keep)    
    Pred_tasks_inner, Pred_tasks_inner_names = get_inner_task_indices_names(Pred_tasks, df_keep)
    
    # only keep inner tasks and add task anmes as column names
    single_pred_inner = single_pred.iloc[:,SP_tasks_inner["cont_classification_task_id"]]
    single_pred_inner.columns = SP_tasks_inner_names

    multi_pred_inner = multi_pred.iloc[:,MP_tasks_inner["cont_classification_task_id"]]
    multi_pred_inner.columns = MP_tasks_inner_names
    
    true_data_inner = true_data.iloc[:,Pred_tasks_inner["cont_classification_task_id"]]
    true_data_inner.columns = Pred_tasks_inner_names
    
    tw_inner = tw.iloc[Pred_tasks_inner["cont_classification_task_id"]]
    
    # mask pred files
    single_pred_inner = mask_matrix(true_data_inner, single_pred_inner)
    multi_pred_inner = mask_matrix(true_data_inner, multi_pred_inner)           
                
    
    return (single_pred_inner, multi_pred_inner, true_data_inner, tw_inner , compounds_vec)


vprint("getting inner join of tasks and compounds")
# validation fold
single_pred_inner, multi_pred_inner, true_data_inner, tw_inner, compounds_vec = \
    get_tasks_inner_join(task_map_pred, folding, args, validation = True)

pd.DataFrame(single_pred_inner).to_csv("single_pred_inner.csv")
pd.DataFrame(multi_pred_inner).to_csv("multi_pred_inner.csv")
pd.DataFrame(true_data_inner).to_csv("true_data_inner.csv")
pd.DataFrame(compounds_vec).to_csv("compounds_vec.csv")
pd.DataFrame(pd.read_csv(task_map_pred)).to_csv("task_map_pred.csv")

# prop folds
single_pred_inner_prop, multi_pred_inner_prop, true_data_inner_prop, tw_inner_prop, compounds_vec_prop = \
    get_tasks_inner_join(task_map_pred, folding, args, validation = False)

#%%

# decide if SP or MP run based on file input

vprint("Path management")
if y_pred_single_path.stem == "pred":
    pred = True
else:
    pred = False
    assert (args.task_map_single != None and args.task_map_multi != None), "When providing .npy Yhat input, also task mapping is required (task_map_single + task_map_multi)"
    task_map_single = Path(args.task_map_single).resolve()
    task_map_multi = Path(args.task_map_multi).resolve()
    task_map_pred = Path(task_map_pred).resolve()
    
    SP_tasks = pd.read_csv(task_map_single)
    MP_tasks = pd.read_csv(task_map_multi)
    PRED_tasks = pd.read_csv(task_map_pred)


if args.filename is not None:
    name = args.filename
else:
    name  = f"{os.path.basename(args.y_true_all)}_{os.path.basename(args.y_pred_single)}_{os.path.basename(args.y_pred_multi)}_{os.path.basename(args.folding_test)}"
    if args.task_weights is not None:
        name += f"_{os.path.basename(args.task_weights)}"
vprint(f"Run name is '{name}'.")
assert not os.path.exists(name), f"{name} already exists... exiting"
os.makedirs(name)


vprint("Loading true labels matrix")
y_true_all = np.load(args.y_true_all, allow_pickle=True).item()
y_true_all = y_true_all.tocsc()



# y_true = y_true_all[folding == fold_va]
y_true = true_data_inner


## Loading task weights (ported from WP2 sparse chem pred.py code)
if args.task_weights is not None:
    tw_df = tw_inner

    assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
    assert tw_df.shape[1] == 2, "task weight file (CSV) must only have 2 columns"
    assert "weight" in tw_df.columns, "weight is missing in task weights CVS file"

    assert y_true.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
    assert (0 <= tw_df.weight).all(), "task weights must not be negative"
    assert (tw_df.weight <= 1).all(), "task weights must not be larger than 1.0"

    assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
    assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
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

def write_global_report(args_name,global_performances,single_multi):
    global name
    perf_df = pd.DataFrame([global_performances],columns=\
        ['aucpr_mean','aucroc_mean','maxf1_mean', 'kappa_mean', 'tn', 'fp', 'fn', 'tp'])
    perf_df.to_csv(name + '/' + single_multi + "_" + os.path.basename(args_name) + '_global_performances.csv')
    
    return perf_df

def write_local_report(args_name,local_performances, single_multi):
    vprint("local_performances: " + str(local_performances.shape))

    # write local report
    local_performances.to_csv(name + '/' + single_multi + "_" + os.path.basename(args_name) + '_local_performances.csv')
    
    # write local report aggregated by assay type, ignore task id
    df = local_performances.loc[:,'assay type':].groupby('assay type').mean()
    df.to_csv(name + '/' + single_multi + "_assay_type" + "_" + os.path.basename(args_name) + '_local_performances.csv')
    return

def mask_y_hat(true_data, pred_data):
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
 
    global fold_va
    global SP_tasks
    global MP_tasks
    global PRED_tasks
    
    #debugging
    vprint(true_data.shape)
    vprint(pred_data.shape)    

    vprint("after validation: " + str(true_data.shape))
    vprint("after validation: " + str(pred_data.shape))
        
    """ remove tasks 
    filter out tasks, that we did not predict on (the tasks from SP y hat), 
    since the overall model predicts on all tasks it saw during training, 
    not only the ones relevant for prediction.
    """
    if true_data.shape[1] != pred_data.shape[1]: # remove tasks if number is not same
        
        """ read data required
        
        file: results/weight_table_T3_mapped.csv, for the single and the multi pharma data
        contains mappings of the taks, so we can filter the tasks out for the MP y_hat file, since we ust not compare tasks that are not in SP y hat
        """
        SP_tasks_rel = SP_tasks[["classification_task_id", "cont_classification_task_id"]] 
        
        MP_tasks = pd.read_csv(task_map_multi)
        MP_tasks_rel = MP_tasks[["classification_task_id", "cont_classification_task_id"]] 
        
        #create overall df
        global_index = pd.DataFrame({"id": range(MP_tasks.shape[0] + 1)})
        
        """
        # we drop all MP row indices from SP and MP index table
        # all remaining relative position-indices (not "id", nor the pandas index, but only position in the table) in SP are the ones we need to keep
        """
        
        # cont class id as first column
        # extend by essay_type
        
        # merge global index with task indices from MP and SP task table, which are subsets of the global index
        MP_df = global_index.merge(MP_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')        
        SP_df = global_index.merge(SP_tasks_rel, left_on = "id", right_on = "classification_task_id", how='left')

        vprint(MP_df)
        #get filled indices from MP and keep only these in SP and MP 
        keep_MPs = [not i for i in MP_df["cont_classification_task_id"].isnull()]
        
        # vprint((keep_MPs))
        
        # drop tasks missing in MP (to make them same shaped)
        MP_df = MP_df[keep_MPs]
        SP_df = SP_df[keep_MPs]

        
        vprint("MP_df before second: " + str(MP_df.shape))
        vprint("SP_df before second: " + str(SP_df.shape))
        vprint("pred before second: " + str(pred_data.shape))
        # vprint(MP_df.shape)
        
        
        """ fix step also remove taks in the other direction
        """        
        # get the final indices to be kept in y hat prediction matrix
        # drop remaining
        task_ids_keep = [not i for i in SP_df["cont_classification_task_id"].isnull()]
        # vprint(task_ids_keep)
        vprint("PRED")
        vprint(pred_data)
        vprint("XXXX  "+ str(pred_data.shape))
        pred_data = pred_data[:, task_ids_keep]

        # debugging
        vprint("Shape after task removal:" + str(pred_data.shape))

    
    """ masking y hat
    
    mask the pred matrix the same as true label matrix, 
    i.e. set all values in pred zero, that are zero (not predicted) in true label matrix
    """
    
    pred_data = pd.DataFrame(pred_data).mask(pd.DataFrame(true_data)==0, 0)
    pred_data = np.array(pred_data)
    assert true_data.shape == pred_data.shape, f"True shape {true_data.shape} and Pred shape {pred_data.shape} need to be identical"
    return pred_data



## run performance code for single- or multi-pharma run
def per_run_performance(y_pred_arg, performance_report, single_multi):
    global y_true
    global tw_df
    global pred
    global args
    global SP_tasks
  
    global single_pred_inner
    
    vprint("Calculating per run performance...")
    
    
    if single_multi == "single":
        if pred:
            y_pred = torch.load(args.y_pred_single)
        else:
            y_pred = sparse.csc_matrix(y_pred_arg)
    else:
        if pred:
            y_pred = torch.load(args.y_pred_single)
        else:
            y_pred = sparse.csc_matrix(y_pred_arg)
    
    
    y_pred_arg = args.y_pred_multi
    y_true = csc_matrix(y_true)
    
    ## checks to make sure y_true and y_pred match
    assert y_true.shape == y_pred.shape, f"y_true shape do not match y_pred ({y_true.shape} & {y_pred.shape})"
    vprint(type(y_true))
    vprint(type(y_pred))
    assert y_true.nnz == y_pred.nnz, "y_true number of nonzero values do not match y_pred"
    assert (y_true.indptr == y_pred.indptr).all(), "y_true indptr do not match y_pred"
    assert (y_true.indices == y_pred.indices).all(), "y_true indices do not match y_pred"
    
    task_id = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
    assay_type = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
    aucpr   = np.full(y_true.shape[1], np.nan)
    aucroc  = np.full(y_true.shape[1], np.nan)
    maxf1   = np.full(y_true.shape[1], np.nan)
    kappa   = np.full(y_true.shape[1], np.nan)
    tn      = np.full(y_true.shape[1], np.nan)
    fp      = np.full(y_true.shape[1], np.nan)
    fn      = np.full(y_true.shape[1], np.nan)
    tp      = np.full(y_true.shape[1], np.nan)
    vennabers = np.full(y_true.shape[1], np.nan)
    vennabers_actives = np.full(y_true.shape[1], np.nan)
    vennabers_inactives = np.full(y_true.shape[1], np.nan)
    
    num_pos = (y_true == +1).sum(0)
    num_neg = (y_true == -1).sum(0)
    cols55  = np.array((num_pos >= 5) & (num_neg >= 5)).flatten()

    """ iterate over tasks
    """
    
    for col in range(y_true.shape[1]):
        y_true_col = y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1
        y_pred_col = y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]]
        
        pts = np.vstack((y_pred_col, y_true_col)).T # points for Venn-ABERS
        
        if y_true_col.shape[0] <= 1:
            ## not enough data for current column, skipping
            continue
        if (y_true_col[0] == y_true_col).all():
            continue
                
        task_id[col] = single_pred_inner.columns[col]
        
        assay_type[col] = SP_tasks["assay_type"][SP_tasks["cont_classification_task_id"]==col].iloc[0]
        y_classes   = np.where(y_pred_col > 0.5, 1, 0)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true_col, probas_pred = y_pred_col)
        aucpr[col]  = sklearn.metrics.auc(x = recall, y = precision)
        aucroc[col] = sklearn.metrics.roc_auc_score(y_true  = y_true_col, y_score = y_pred_col)
        maxf1[col]  = find_max_f1(precision, recall)
        kappa[col]  = sklearn.metrics.cohen_kappa_score(y_true_col, y_classes)
        tn[col], fp[col], fn[col], tp[col] = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()

        # vennabers[col], vennabers_actives[col], vennabers_inactives[col] = get_VA_margin_mean_cross(pts) # pass the folding

        
    ##local performance:
    local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],aucpr[cols55],aucroc[cols55],maxf1[cols55],\
                        kappa[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55], vennabers[cols55], vennabers_actives[cols55], vennabers_inactives[cols55]]).T,\
                        columns=['task id', 'assay type', 'aucpr','aucroc','maxf1','kappa','tn','fp','fn','tp', 'vennabers', 'vennabers_actives', 'vennabers_inactives'])
    
    ##correct the datatypes for numeric columns
    vprint(local_performance)
    for c in local_performance.iloc[:,2:].columns:
        local_performance.loc[:,c] = local_performance.loc[:,c].astype(float)
    write_local_report(y_pred_arg,local_performance, single_multi)
                        
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
    
    if pred:
        global_pre_calculated_performance = global_perf_from_json(performance_report)
        #only assert pre-calculated performance if not weight averaging for compatability
        if not args.task_weights:
            assert global_pre_calculated_performance == aucpr_mean, f"reported performance in {performance_report} ({global_pre_calculated_performance}) does not match calculated performance for {y_pred_arg} ({aucpr_mean})"
    
    global_performance = write_global_report(y_pred_arg,[aucpr_mean,aucroc_mean,maxf1_mean,kappa_mean, tn_sum, fp_sum, fn_sum, tp_sum], single_multi)
    return [local_performance,global_performance]


def calculate_deltas(single_results, multi_results):
    for idx, delta_comparison in enumerate(['local_deltas.csv','global_deltas.csv']):

        assert single_results[idx].shape[0] == multi_results[idx].shape[0], "the number of tasks are not equal between the single- and multi-pharma runs"
        assert single_results[idx].shape[1] == multi_results[idx].shape[1], "the number of reported metrics are not equal between the single- and multi-pharma runs"
        
        # add assay type only if local
        if(delta_comparison == 'local_deltas.csv'):
            at = multi_results[idx]["assay type"]
            delta = (multi_results[idx].loc[:, "aucpr":]-single_results[idx].loc[:, "aucpr":])
            tdf = pd.concat([at, delta], axis = 1)
            tdf.to_csv(name + '/' + delta_comparison)
            
            # aggregate on assay type level
            tdf.groupby("assay type").mean().to_csv(name + '/' + 'assay_type' + delta_comparison)
        elif (delta_comparison == 'global_deltas.csv'):
            (multi_results[idx]-single_results[idx]).to_csv(name + '/' + delta_comparison)


def get_tuples_form_matrix(df):
    """
    returns tuples (row_index, col_index, value) for all cells from a df not zero
    """
    df_sparse = csc_matrix(df) 
    row_ids, col_ids, data = sparse.find(df_sparse)
    row_ids_comps = pd.Series(df.index[row_ids])
    col_ids_comps = pd.Series(df.columns[col_ids])
    tups = pd.concat([row_ids_comps, col_ids_comps, pd.Series(data)], axis = 1)
    tups.columns = ["compound_id", "task_id", "value"]
    
    return tups
    
    

def add_task_comp_ids(data_df, task_map, T11, T5):
    """
    # add task ids and compound ids to preds and actual labels (T10)
    """    
    #get single and multi task ids
    single_tasks = task_map["classification_task_id"][task_map["cont_classification_task_id"].notna()]
    
    # set column names from tasklist
    data_df.columns  = list(single_tasks)  
    
    # for train_preds_single/multi  get get input compound id from T5 via descriptor_vector id from T11
    x = data_df.merge(T11[["cont_descriptor_vector_id", "descriptor_vector_id"]], left_on = data_df.index ,right_on ="cont_descriptor_vector_id").iloc[:,-2:]
    compounds_vec = x.merge(T5.drop_duplicates(subset=["descriptor_vector_id"], keep=False), on="descriptor_vector_id", how="left")
    
    # set compound ids to be the index
    data_df.index = list(compounds_vec["input_compound_id"])
    data_df = data_df[data_df.index.notna()]
    data_df.index = data_df.index.astype(int)
    
    return data_df

def get_cal_tuples_from_training(pred_df, task_map, T5, T10, T11, 
                                  folding_train
                                 ):
    """
    # # prepare to fit VA on all training data and predict on all test points at once
    """
    # load the data
    T10 = pd.DataFrame(np.load(T10, allow_pickle=True).tolist().todense())        
    T10 = pd.DataFrame.sparse.from_spmatrix(csc_matrix(T10))            # convert to sparse
    pred_df  = pd.DataFrame(np.load(pred_df))
    pred_df  = pred_df.mask(T10 == 0, 0)                                # mask pred with true labels
    pred_df  = pd.DataFrame.sparse.from_spmatrix(csc_matrix(pred_df))    # convert to sparse
    
    #keep all but validation fold
    # pred_df = pred_df[folding_train !=0]
    # T10 = T10[folding_train !=0]
    
    task_map = pd.read_csv(task_map)
    T11 = pd.read_csv(T11)
    T5  = pd.read_csv(T5)

    # mask predictions and add task and compound ids
    pred_df  = add_task_comp_ids(pred_df, task_map, T11, T5)    # pred matrix
    T10      = add_task_comp_ids(T10, task_map, T11, T5)        # true matrix
    
    # convert the matrices into tuples for easier conversion into VA points    
    pred_tuples = get_tuples_form_matrix(pred_df)
    true_tuples = get_tuples_form_matrix(T10)
    
    assert pred_tuples.shape == true_tuples.shape, "number of tasks-compound tuples from pred and true do not match"
    cal = pred_tuples.merge(true_tuples, on = ["compound_id", "task_id"])
    
    return cal   
   
        
def write_distances_report(true_data_prop, 
                           true_data,
                           pred_data_prop,  
                           pred_data,
                           compounds_vec_prop,
                           compounds_vec,
                           args,
                           performance_results,
                           single_multi,
                           k):

    ''' Get training data in same shape with only the required tasks and all relating compounds
    
    '''    
    calibrate_taskwise = True
    
    global name
    
    training_tasks = true_data_inner_prop.columns    
    pd.DataFrame().to_csv(name + '/' + single_multi +  "_tmp_distances.csv")
    
    
    num_pos = (true_data == +1).sum(0)
    num_neg = (true_data == -1).sum(0)
    cols55  = np.array((num_pos >= 5) & (num_neg >= 5)).flatten()
    
    #remove non-quorum tasks from df
    true_data_prop = true_data_prop.loc[:,list(cols55)]
    pred_data_prop = pred_data_prop.loc[:,list(cols55)]
    pred_data = pred_data.loc[:,list(cols55)]
    true_data = true_data.loc[:,list(cols55)]
    
    
    # load T2/4 files for lookup
    if single_multi == "single":
        T2_train = pd.read_csv(args.T2_train_single)
        T4_train = pd.read_csv(args.T4_train_single)
    else:
        T2_train = pd.read_csv(args.T2_train_multi)
        T4_train = pd.read_csv(args.T4_train_multi)
        
    T2_pred = pd.read_csv(args.T2_pred)
    
    # take all compounds for these tasks from T4
    training_compounds = T4_train[T4_train['classification_task_id'].isin(training_tasks)]
    
    #add smiles to training_compounds
    training_compounds = training_compounds.merge(T2_train, on = "input_compound_id")  
    overall = pd.DataFrame() # contains all results (comp, task, VA, distances etc)     
    
      
    """
    get calibration points for VA
    """    
    if single_multi == "single":
        folding_train = np.load(args.folding_train_single)
        cal_tuples= get_cal_tuples_from_training(args.train_preds_single, args.task_map_single, args.T5_train_single, args.T10_train_single, args.T11_train_single, 
                                                  folding_train
                                                 )
    else:
        folding_train = np.load(args.folding_train_multi)
        cal_tuples= get_cal_tuples_from_training(args.train_preds_multi, args.task_map_multi, args.T5_train_multi, args.T10_train_multi, args.T11_train_multi, 
                                                 folding_train
                                                 )
    """ keep only validation fold
    """
    
    training_compounds = cal_tuples.merge(T2_train, left_on = "compound_id" , right_on = "input_compound_id")
    training_compounds = training_compounds.rename({"task_id":"classification_task_id", "value_y":"class_label"}, axis = 1)
    training_compounds = training_compounds.drop(["compound_id", 'value_x'], axis =1)
    training_compounds["class_label"] = [0 if i==-1.0 else 1 for i in training_compounds["class_label"]]

    cal_tuples.columns = ["compound_id", "task_id","pred","true"]
    # convert -1 labels to 0
    cal_tuples["true"][cal_tuples["true"] == -1] = 0
    
    
    if calibrate_taskwise == False:
        """
        prep if calculation of VA based on all tasks instead task wise
        """
        pred_data_tmp       = pd.concat([pred_data, pd.Series(compounds_vec)], axis =1)                     # merge compounds 
        pred_data_tmp       = pd.DataFrame([i[1] for i in pred_data_tmp.iterrows() if len(i[1][0]) ==1])    # remove all rows where number of comps is >1
        pred_data_tmp.index = [i[0] for i in pred_data_tmp[0]]                                              # set index to be compounds
        pred_data_tmp       = pred_data_tmp.iloc[:,:-1]                                                     # drop last column containing compounds
    
        pred_data_prop_tmp       = pd.concat([pred_data_prop, pd.Series(compounds_vec_prop)], axis =1)
        pred_data_prop_tmp       = pd.DataFrame([i[1] for i in pred_data_prop_tmp.iterrows() if len(i[1][0]) ==1])
        pred_data_prop_tmp.index = [i[0] for i in pred_data_prop_tmp[0]]
        pred_data_prop_tmp       = pred_data_prop_tmp.iloc[:,:-1]
        
        # get tuples
        pred_data_tmp_tuples        = get_tuples_form_matrix(pred_data_tmp)
        pred_data_prop_tmp_tuples   = get_tuples_form_matrix(pred_data_prop_tmp)
        
        # stack tuples of all folds ~ entire prediction set/cluster
        pred_tuples_all = pd.concat([pred_data_tmp_tuples ,pred_data_prop_tmp_tuples], axis = 0)
        
        # get only the prediciton columns
        test = pred_tuples_all.iloc[:, 2]
    
        cal_pts = [(i[2], i[3]) for i in np.array(cal_tuples)]
        test = list(test)
        
        pred_tuples_all["p0"], pred_tuples_all["p1"] = ScoresToMultiProbs(cal_pts, test)
        pred_tuples_all["VA_loss"] = pred_tuples_all["p1"] / 1- pred_tuples_all["p0"] + pred_tuples_all["p1"]
        pred_tuples_all["VA_margin"] = pred_tuples_all["p1"]-pred_tuples_all["p0"]
        
    
    #calibrate VA and get preds
    col_counter = 0
    
    for col_id, col in true_data.iteritems():   
        col_counter +=1
        vprint("Distance calculation for task " + str(col_counter) + "/" + str(true_data.shape[1]) + "...")
        # get current compound-task tuple (can be several compounds per row)
        aucpr = []
        
        test_smiles_list = []
        
        va_list_p0 = []
        va_list_p1 = []
        
        nn_list_number = []
        
        nn_smiles_1 = []
        nn_sims_1 = []
        nn_labels_1 = []
        
        nn_smiles_0 = []
        nn_sims_0 = []
        nn_labels_0 = []
                    
        nn_smiles_n = []
        nn_sims_n = []
        nn_labels_n = []
        
        
        tuples_prop = get_tuples(col_id, true_data_prop[col_id],compounds_vec_prop, true_data_prop, pred_data_prop)
        tuples      = get_tuples(col_id, col,compounds_vec, true_data, pred_data)   
        
        
        """ commented out
        """
        tuples = pd.DataFrame(tuples + tuples_prop) # compute distance_report for all test folds
        # tuples = pd.DataFrame(tuples) #  compute distance_report only for test validation fold
        
        #drop all rows with more than one compound id
        tuples = pd.DataFrame([i[1] for i in tuples.iterrows() if len(i[1][0]) ==1])
    
        # convert compound id form list to int
        tuples[0] = [i[0] for i in tuples[0]]
        tuples  = [(t[0], t[1], t[2], t[3]) for tid, t in tuples.iterrows()]        
            
        """
        for label 0 and label 1
        # get all compounds from training_compounds for the current task 
        # calculate the fp items
        # get bulk tanimoto similarities
        """
        training_compounds_task = training_compounds[training_compounds["classification_task_id"] == col_id]
        
        training_compounds_task_0 = training_compounds_task[training_compounds_task["class_label"] == 0]
        training_compounds_task_1 = training_compounds_task[training_compounds_task["class_label"] == 1]
        
        temp_mols_0 = [MolFromSmiles(x) for x in training_compounds_task_0.smiles]   
        temp_mols_1 = [MolFromSmiles(x) for x in training_compounds_task_1.smiles]   

        fps_struct_0 = [FingerprintMols.FingerprintMol(x) for x in temp_mols_0]
        fps_struct_1 = [FingerprintMols.FingerprintMol(x) for x in temp_mols_1]
        
        df_0 = training_compounds_task_0.copy()
        df_1 = training_compounds_task_1.copy()

        
        """ perform VA task wise calculation
        """
        cal_pts = [(i[2], i[3]) for i in np.array(cal_tuples[cal_tuples.task_id == col_id])]
        
        # calibrate VA and get predictions
        p0, p1 = ScoresToMultiProbs(cal_pts, [t[2] for t in tuples])       
       
        va_list_p0 = p0
        va_list_p1 = p1
        
        for idx, t in enumerate(tuples):
            
            # for each compound in validation set of current task, VA is fitted on all other folds and the score appended
            other_t = list(tuples)
            other_t.remove(t)
            test_comp  = t[0]
            
            """
            calculate similarities
            """            
            # look the compound up in T2
            test_smiles = T2_pred["smiles"][T2_pred["input_compound_id"] == test_comp].iloc[0]
            
            # test_fp = calculateFP(test_smiles)
            """ try alternative fp calc
            """
            test_mol = MolFromSmiles(test_smiles)
            test_struct = FingerprintMols.FingerprintMol(test_mol)
            test_sims_0 = DataStructs.BulkTanimotoSimilarity(test_struct,fps_struct_0)
            test_sims_1 = DataStructs.BulkTanimotoSimilarity(test_struct,fps_struct_1)
            
            df_0["test_sims_0"] = test_sims_0
            df_1["test_sims_1"] = test_sims_1
            
            # bulk tanimoto
            df_0 = df_0.sort_values(by=["test_sims_0"], ascending = False)
            df_1 = df_1.sort_values(by=["test_sims_1"], ascending = False)
            
            aucpr.append(performance_results[0].aucpr[performance_results[0]["task id"] == str(col_id)].iloc[0])
            test_smiles_list.append(test_smiles)
            
            nn_list_number.append(list(np.arange(1, k+1)))
            nn_smiles_0.append(list(df_0["smiles"][:k]))
            nn_smiles_1.append(list(df_1["smiles"][:k]))

            nn_sims_0.append(list(df_0["test_sims_0"][:k]))             
            nn_sims_1.append(list(df_1["test_sims_1"][:k]))
            
            nn_labels_0.append(list(df_0["class_label"][:k]))
            nn_labels_1.append(list(df_1["class_label"][:k]))
            
            #add label-independent NN
            if df_1["test_sims_1"].iloc[0] > df_0["test_sims_0"].iloc[0]:
                max_df = df_1
                nn_sims_n.append(max_df["test_sims_1"].iloc[0])
            else:
                max_df = df_0
                nn_sims_n.append(max_df["test_sims_0"].iloc[0])
            nn_smiles_n.append(max_df["smiles"].iloc[0])
            nn_labels_n.append(max_df["class_label"].iloc[0])
            
        tuples = pd.DataFrame(tuples, columns = ['compound','task', 'prediction', 'true_label'])
        
        nn_columns  =   [nn_smiles_0, nn_smiles_1, nn_sims_0, nn_sims_1, nn_labels_0, nn_labels_1]   
        
        tuples["comp_smiles"]=test_smiles_list
        
        
        if calibrate_taskwise == False:
            """ if calibrate on all tasks
            """
            tuples["p0"] = list(pred_tuples_all[pred_tuples_all["task_id"] == col_id]["p0"])
            tuples["p1"] = list(pred_tuples_all[pred_tuples_all.task_id == col_id]["p1"])
            tuples["VA_loss"] = list(pred_tuples_all[pred_tuples_all.task_id == col_id]["VA_loss"])
            tuples["VA_margin"] = list(pred_tuples_all[pred_tuples_all.task_id == col_id]["VA_margin"])
            
        elif (calibrate_taskwise == True):
            """ if calibrate task-wise
            """
            T3_pred = pd.read_csv(args.T3_pred)
            
            tuples["assay_type"] = list(tuples.merge(T3_pred, left_on = "task", right_on = "classification_task_id")["assay_type"])
            tuples["p0"] = va_list_p0
            tuples["p1"] = va_list_p1
            tuples["V_loss"] = tuples["p1"]/(1-tuples["p0"]+tuples["p1"])
            tuples["VA_margin"] = np.array(va_list_p1) - np.array(va_list_p0) 
            
            tuples["aucpr"] = aucpr
            tuples["k_no"] = nn_list_number
            tuples["smiles_nearest"] = nn_smiles_n
            tuples["sims_nearest"] = nn_sims_n
            tuples["class_labels_nearest"] = nn_labels_n
            
            tmp_smiles_0 = pd.DataFrame(nn_smiles_0, columns = ["smile_neg_1", "smile_neg_2", "smile_neg_3", "smile_neg_4", "smile_neg_5"])
            tmp_sims_0 = pd.DataFrame(nn_sims_0, columns = ["sim_neg_1", "sim_neg_2", "sim_neg_3", "sim_neg_4", "sim_neg_5"])
            tmp_labels_0 = pd.DataFrame(nn_labels_0, columns = ["label_neg_1", "label_neg_2", "label_neg_3", "label_neg_4", "label_neg_5"])
            tmp_smiles_1 = pd.DataFrame(nn_smiles_1, columns = ["smile_pos_1", "smile_pos_2", "smile_pos_3", "smile_pos_4", "smile_pos_5"])
            tmp_sims_1 = pd.DataFrame(nn_sims_1, columns = ["sim_pos_1", "sim_pos_2", "sim_pos_3", "sim_pos_4", "sim_pos_5"])
            tmp_labels_1 = pd.DataFrame(nn_labels_1, columns = ["label_pos_1", "label_pos_2", "label_pos_3", "label_pos_4", "label_pos_5"])

        tmp_stack = pd.DataFrame()
        for i in range(tmp_smiles_0.shape[1]):
            tmp_stack = pd.concat([tmp_stack,
                                   tmp_smiles_0.iloc[:,i],
                                   tmp_sims_0.iloc[:,i],
                                   tmp_labels_0.iloc[:,i],
                                   tmp_smiles_1.iloc[:,i],
                                   tmp_sims_1.iloc[:,i],
                                   tmp_labels_1.iloc[:,i]
                                   ], axis = 1)
        tuples = pd.concat([tuples, tmp_stack], axis = 1)
        
        #add va_list to tuples and add tuples to overall list
        overall = overall.append(tuples) # contains the tuple list for each task id      
        tuples.to_csv(name + '/' + single_multi +  "_tmp_distances.csv", mode="a", header = False)
    overall.to_csv(name + '/' + single_multi +  "_distances.csv")
    vprint("Distance calculation done ...")
    return overall



vprint(f"Calculating '{args.y_pred_single}' performance.")


if pred:
    single_partner_results=per_run_performance(np.array(single_pred_inner),args.single_performance_report, "single")
else:    
    masked_single = mask_y_hat(np.array(true_data_inner), np.array(single_pred_inner))
    single_partner_results=per_run_performance(masked_single,args.single_performance_report, "single")
    distances_single = write_distances_report(
                        true_data_inner_prop, 
                        true_data_inner, 
                        single_pred_inner_prop,
                        single_pred_inner,
                        compounds_vec_prop,
                        compounds_vec,
                        args,
                        single_partner_results,
                        "single",
                        5)
    
vprint(f"Calculating '{args.y_pred_multi}' performance.")

if pred:
    multi_partner_results=per_run_performance(np.array(multi_pred_inner),args.multi_performance_report, "multi")
else:
    masked_multi = mask_y_hat(np.array(true_data_inner), np.array(multi_pred_inner))
    multi_partner_results=per_run_performance(masked_multi,args.multi_performance_report, "multi")
    distances_multi = write_distances_report(
                        true_data_inner_prop, 
                        true_data_inner, 
                        multi_pred_inner_prop,
                        multi_pred_inner,
                        compounds_vec_prop,
                        compounds_vec,
                        args,
                        multi_partner_results,
                        "multi",
                        5)
  
vprint(f"Calculating delta between '{args.y_pred_single}' & '{args.y_pred_multi}' performances.")
calculate_deltas(single_partner_results,multi_partner_results)

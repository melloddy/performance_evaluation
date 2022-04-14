import os
import sys
import pandas as pd
import numpy as np
import glob
import json
import ast
import seaborn as sns
import tarfile
from shutil import copyfile as cp
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Select Optimal Models"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--main_metric_label",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cp_list_file",
        type=str,
    )
    args = parser.parse_args()
    return args


def get_overlapping_ranks(ID_PH1, ID_PH2, ID_PH3):
    
    l_files = glob.glob('cp'+ID_PH1+'*')
    l_ranks_PH1 = [file.split('_')[1] for file in l_files]    
    l_files = glob.glob('cp'+ID_PH2+'*')
    l_ranks_PH2 = [file.split('_')[1] for file in l_files]    
    l_files = glob.glob('cp'+ID_PH3+'*')
    l_ranks_PH3 = [file.split('_')[1] for file in l_files]

    l_ranks = list(set(l_ranks_PH1) & set(l_ranks_PH2) & set(l_ranks_PH3))
    max_rank = np.array(l_ranks, dtype='int').max()
    
    return l_ranks, max_rank


def untar_rename_get_main_metric(file_tar, main_metric_label):
    
    dir_tar = file_tar[:-7]
    
    # partial un-tar
    with tarfile.open(file_tar, 'r:gz') as tar:
        l_file_extract = [file for file in tar.getnames() if 'perf' in file]
        l_file_extract.append(dir_tar+'/metadata.json')
        for file_extract in l_file_extract:
            tar.extract(file_extract)

    # rename perf files
    with open(f'{dir_tar}/metadata.json') as json_data:
        metadata = json.load(json_data)
        metrics = metadata["metrics"]
    for m in metrics:
        metric = metadata["metrics"][m]["metadata"]["metrics_name"]
        cp(f"{dir_tar}/perf/%s-perf.json" %m, f"{dir_tar}/perf/%s-perf.json" %metric)
        
    # extract main metric
    with open(f"{dir_tar}/perf/"+main_metric_label+"-perf.json") as json_data:
        data = json.load(json_data)
        main_metric = data["all"]
        
    return main_metric


args = vars(arg_parser())
dataset = args["dataset"]
main_metric_label = args["main_metric_label"]
cp_list_file = args["cp_list_file"]

# read in CP list
df = pd.read_csv(cp_list_file, sep=';')

# pivot dataframe
df_pivot = df.pivot(index='Model Index', columns='Phase', values='CP Tag')
df_pivot = df_pivot.rename_axis(None, axis=1).reset_index()

# iterate over model sets
l_max_rank = []
l_opt_rank = []
l_opt_perf_phase1 = []
l_opt_perf_phase2 = []
l_model_file_phase1 = []
l_model_file_phase2 = []
l_model_file_phase3 = []

for _, row in df_pivot.iterrows():
    
    model_index = row['Model Index']
    ID_PH1 = row['PH1'][2:]
    ID_PH2 = row['PH2'][2:]
    ID_PH3 = row['PH3'][2:]
    
    print("Working on Model set #%s (CPs: %s %s %s)" % (model_index, ID_PH1, ID_PH2, ID_PH3))
    
    # get model list availabel across all phases
    l_ranks, max_rank = get_overlapping_ranks(ID_PH1, ID_PH2, ID_PH3)

    # extract optimal epoch based on phase1 results
    l_perf = []
    l_files_phase1 = []
    for rank in l_ranks:
        file_tar_PH1 = glob.glob('cp'+ID_PH1+'_'+rank+'_*')[0]
        perf_PH1 = untar_rename_get_main_metric(file_tar_PH1, main_metric_label)
        l_perf.append(perf_PH1)
        l_files_phase1.append(file_tar_PH1)

    idx_max = np.array(l_perf).argmax()
    opt_rank = l_ranks[idx_max]
    opt_perf_phase1 = l_perf[idx_max]
    opt_file_phase1 = l_files_phase1[idx_max]

    # extract corresponding phase2 performance 
    file_tar_PH2 = glob.glob('cp'+ID_PH2+'_'+opt_rank+'_*')[0]
    opt_perf_phase2 = untar_rename_get_main_metric(file_tar_PH2, main_metric_label)
    file_tar_PH3 = glob.glob('cp'+ID_PH3+'_'+opt_rank+'_*')[0]
    
    print("\t rank max: %s, rank opt: %s" % (max_rank, opt_rank))
    print("\t perf phase1: %s, perf phase2: %s" % (opt_perf_phase1, opt_perf_phase2))
    
    l_max_rank.append(max_rank)
    l_opt_rank.append(opt_rank)
    l_opt_perf_phase1.append(opt_perf_phase1)
    l_opt_perf_phase2.append(opt_perf_phase2)
    l_model_file_phase1.append(opt_file_phase1)
    l_model_file_phase2.append(file_tar_PH2)
    l_model_file_phase3.append(file_tar_PH3)


df_out = df_pivot.copy()
df_out['rank max'] = l_max_rank
df_out['rank opt'] = l_opt_rank
df_out['perf PH1'] = l_opt_perf_phase1
df_out['perf PH2'] = l_opt_perf_phase2
df_out['file PH1'] = l_model_file_phase1
df_out['file PH2'] = l_model_file_phase2
df_out['file PH3'] = l_model_file_phase3
df_out.to_csv('perf_all_'+dataset+'.csv', index = False)

df_out_opt = df_out.iloc[df_out['perf PH1'].idxmax()].to_frame().T
df_out_opt.to_csv('perf_opt_'+dataset+'.csv', index = False)


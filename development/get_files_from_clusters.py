#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:34:08 2020

@author: marc.frey@astrazeneca.com
"""
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
import os

from datetime import datetime

def init_arg_parser():
    parser = argparse.ArgumentParser(description='Split T2 clustered file into one file per cluster')
    parser.add_argument('-T2', '--T2_file', type=str, help='path of the clustered T2 file (output from cluster_MELLODDY_xxxx.py))', required=True)
    parser.add_argument('-T3', '--T3_file', type=str, help='path to T3 file', required=True)
    parser.add_argument('-T4', '--T4_file', type=str, help='path to T4 file', required=True)
    args = parser.parse_args()

    return args


""" Path handling
"""
args = vars(init_arg_parser())
T2_path = args["T2_file"]
T3_path = args["T3_file"]
T4_path = args["T4_file"]



T2_path = Path(T2_path)
T3_path = Path(T3_path)
T4_path = Path(T4_path)

T2_parent = T2_path.parent

os.chdir(T2_parent)

""" T2 file read
"""
#read T2_clustered file
T2 = pd.read_csv(T2_path)

# sort data by clusters
T2.sort_values(by = ["Cluster_PC1_PC2"], inplace = True)

# get unique clusters
clusters =  T2["Cluster_PC1_PC2"].unique()



""" T4 file read
"""

# read file
T4 = pd.read_csv(T4_path)

#rename column for join
T2.rename(columns={"compound_id":"input_compound_id"}, inplace = True)
T4 = T4.merge(T2, on = "input_compound_id")



""" T3 file read
"""

# get distinct classification task ids for all compound ids in cluster c from T4
# filter T3 for these tasks


T3 = pd.read_csv(T3_path)
#T3 = T3.merge(T4, on = "classification_task_id")

""" Data Frame Compression
https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
"""

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

T2 = reduce_mem_usage(T2)
T4 = reduce_mem_usage(T4)
T3 = reduce_mem_usage(T3)

    
""" writing the files 0-9 into split folders
"""
# iterate over clusters, create folder for each
# get matching rows and write to csv
#%%
t1 = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

for c in clusters:
    
    # folder = Path("split_0" + str(c) + "__" + str(t1))
    folder = Path("split_0" + str(c))
    folder.mkdir(parents= True)

    # filter and write T2 file    
    split_set2 = T2[T2["Cluster_PC1_PC2"] == c].iloc[:,:-1] 
    split_set2.rename(columns={"canonical_smiles":"smiles"}, inplace = True)
    split_set2.to_csv(str(T2_parent) + "/" + str(folder)+ "/"+ str(T2_path.stem) + "_split_0" + str(c) + ".csv" , index = False)

    # filter and write T4 file
    split_set4 = T4[T4["Cluster_PC1_PC2"] == c].iloc[:,:-2]
    split_set4.to_csv(str(T2_parent) + "/" + str(folder)+ "/" + str(T4_path.stem) + "_split_0" + str(c) + ".csv" , index = False)

    # filter and write T3 file
    
    #find all classification tasks relevant to the current cluter
    relevant_class_ids = split_set4["classification_task_id"].unique()
    
    # split_set3 = T3[T3["Cluster_PC1_PC2"] == c].iloc[:,:8]
    split_set3 = T3[T3["classification_task_id"].isin(relevant_class_ids)] # get only rows that contain one of the relevant classification task ids
    split_set3.to_csv(str(T2_parent) + "/" + str(folder)+ "/" + str(T3_path.stem) + "_split_0" + str(c) + ".csv" , index = False)


""" create 11th folder and files containing all data
"""

# folder = Path("split_" + str(10)+ "__" + str(t1))
folder = Path("split_" + str(10))
folder.mkdir(parents= True)

T2.rename(columns={"canonical_smiles":"smiles"}, inplace = True)
T2.iloc[:,:-1] .to_csv(str(T2_parent) + "/" + str(folder)+ "/" +str(T2_path.stem) + "_split_" + str(10) + ".csv" , index = False)    
T4.iloc[:,:-2].to_csv(str(T2_parent) + "/" + str(folder)+ "/" +str(T4_path.stem) + "_split_" + str(10) + ".csv" , index = False)
T3.to_csv(str(T2_parent) + "/" + str(folder)+ "/" +str(T3_path.stem) + "_split_" + str(10) + ".csv" , index = False)
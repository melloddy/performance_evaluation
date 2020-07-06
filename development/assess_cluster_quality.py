#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 06:14:09 2020

@author: kfrl011
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def init_arg_parser():
    parser = argparse.ArgumentParser(description='plot cluster distributions of LSH and labels')
    parser.add_argument("-b", "--base_dir", type=str, help = "base directory", required = True)
    parser.add_argument("-c", "--clusters", nargs='+', help ="cluster number to create histograms for", required = True)
    args = parser.parse_args()

    return args

args = vars(init_arg_parser())

base_dir =  args["base_dir"]
clusters = args["clusters"]

print(clusters) 

for c in clusters:
    
    # T11 = args["folding"]
    # T4 = args["task_class"]
    
    # dir_to = Path(T4).parent
    # filename_to = Path(T4).stem
    # path_to = str(dir_to) + "/" + str(filename_to)
    
    if int(c) <= 9:
        split = "split_0" + str(c)
    else:
        split = "split_" + str(c)
    
    dir_to = base_dir + "/" + split
    
    T11 = dir_to + "/data_prep/split_" + str(c) + "_CH/files_4_ml/T11_fold_vector.npy"
    T4 = dir_to + "/CH_T4_" + split + ".csv"
    
    
    import pandas as pd
    
    # LSH distribution
    LSH = np.load(T11)
    plt.hist(LSH)
    plt.savefig(dir_to + "/" + split + "_LSH_hist.png")
    plt.cla()
    
    # class labels
    y = pd.read_csv(T4)["class_label"]
    plt.hist(y)
    plt.savefig(dir_to + "/" + split + "_class_hist.png")
    plt.cla()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 07:44:02 2020

@author: kfrl011
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
from pathlib import Path

"""optional seaborn style  setup
"""

flatui = ["#661D98", "#2CBDFE"]

sns.set(font='Franklin Gothic Book',
        rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'dimgrey',
 'xtick.bottom': False,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})


sns.set_palette(flatui, 2)

def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run K-Means Clustering with PCA')
    parser.add_argument('-s', '--single_performance', type=str, help='path of multi_y_hat.npy_local_performances.csv', required=True)
    parser.add_argument('-m', '--multi_performance', type=str, help='number of single_y_hat.npy_local_performances.csv', required = True)
    args = parser.parse_args()

    return args


def plot_assay_type_boxplots(single, multi):
    assay_types = single["assay type"].unique()
    
    # metrics = data.columns[2:]
    metrics = ['aucpr', 'aucroc', 'maxf1', 'kappa','vennabers']
    
    for m in metrics:
        data = {}
        for at in assay_types:
            data[at] = single[single["assay type"] == at][str(m)]
    
        fig, ax = plt.subplots()
        ax.boxplot(data.values())
        ax.set_xticklabels(data.keys())
        ax.set_ylabel(str(m))
        ax.set_title("Assay comparison of " + str(m))
        plt.cla()

def plot_assay_type_boxplots_multi(single, multi):
    assay_types = single["assay type"].unique()
    
    # metrics = data.columns[2:]
    metrics = ['aucpr', 'aucroc', 'maxf1', 'kappa','vennabers']
    
    for m in metrics:
        data = pd.DataFrame({"assay type": single["assay type"],
                                  "Single": single[str(m)],
                                  "Multi": multi[str(m)]})
        
        dd=pd.melt(data,id_vars=['assay type'],value_vars=['Single','Multi'],var_name='model')
        sns.boxplot(x='assay type',y='value',data=dd,hue='model')
        plt.ylabel(str(m))
        plt.title("Assay comparison of " + str(m))
        plt.cla()

def plot_metrics(single, multi, x = "assay_type", plottype = "scatter", to_dir = None):
    to_dir = str(to_dir) + "/performance_plots"
    Path(to_dir).mkdir(exist_ok = True)
    
    assay_type = single["assay type"].unique()
    metrics = ['aucpr', 'aucroc', 'maxf1', 'kappa','vennabers']
    task_size = multi["tp"] + multi["fp"] + multi["tn"] + multi["fn"]
    
    x_axes_dict = {"assay_type": single["assay type"], 
              "task_size": task_size}
    
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    for m in metrics:
        data = pd.DataFrame({str(x): x_axes_dict[str(x)],
                                  "Single": single[str(m)],
                                  "Multi": multi[str(m)]})
        dd=pd.melt(data,id_vars=[str(x)],value_vars=['Single','Multi'],var_name='model')

        if plottype == "boxplot":
            sns.boxplot(x = str(x),y='value',data=dd,hue='model')
        elif plottype == "scatter":            
            # sns.scatterplot(x_axes_dict[str(x)], single[m]) 
            # sns.scatterplot(x = x_axes_dict[str(x)], y = 'value', data=dd, hue='model') 
            sns.scatterplot(x = str(x), y = 'value', data=dd, hue='model', alpha = 0.2) 
            
        plt.xlabel(str(x))
        plt.ylabel(str(m))
        plt.title("Comparison of " + str(m))
        plt.savefig(str(to_dir) + "/" + m +"_vs_" + x + "_" + plottype + ".png")
        plt.cla()
    sns.distplot(task_size)
    plt.xlabel(str(x))
    plt.ylabel("no of tasks")
    plt.title("Task size distribution")
    plt.savefig(str(to_dir) + "/task_size_distribution.png")
    plt.cla()
        
# def plot_metric_vs_task_size(single, multi, x = "assay_type", plottype = "scatter"):
#     assay_type = single["assay type"].unique()
#     metrics = ['aucpr', 'aucroc', 'maxf1', 'kappa','vennabers']
#     task_size = multi["tp"] + multi["fp"] + multi["tn"] + multi["fn"]
    
#     x_axes_dict = {"assay_type": single["assay type"], 
#               "task_size": task_size}
    
#     rows = 2
#     cols = 3
#     fig, axes = plt.subplots(nrows=rows, ncols=cols)


            
#     for m in metrics:
#         data = pd.DataFrame({str(x): x_axes_dict[str(x)],
#                                   "Single": single[str(m)],
#                                   "Multi": multi[str(m)]})
#         dd=pd.melt(data,id_vars=[str(x)],value_vars=['Single','Multi'],var_name='model')

#         if plottype == "boxplot":
#             for i, var in enumerate(metrics):  
#                 row = i//cols
#                 pos = i % rows
#                 sns.boxplot(x = str(x),y='value',data=dd,hue='model',ax=axes[row][pos])
#         elif plottype == "scatter":            
#             for i, var in enumerate(metrics):  
#                 row = i//cols
#                 pos = i % rows
#                 sns.scatterplot(x = str(x), y = 'value', data=dd, hue='model', alpha = 0.3,ax=axes[row][pos]) 
#         plt.xlabel(str(x))
#         plt.ylabel(str(m))
#         plt.title("Comparison of " + str(m))
#         #plt.show()

args = vars(init_arg_parser())
single_path = args["single_performance"]
multi_path = args["multi_performance"]

to_dir = Path(single_path).parent

multi = pd.read_csv(multi_path, index_col = None).loc[:,"task id":]
single = pd.read_csv(single_path)      
# plot_assay_type_boxplots(single, multi)
# plot_assay_type_boxplots_multi(single, multi)
plot_metrics(single, multi, "task_size", to_dir = to_dir)
plot_metrics(single, multi, "assay_type", "boxplot", to_dir = to_dir)

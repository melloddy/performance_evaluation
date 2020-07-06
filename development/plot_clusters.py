# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:32:23 2020

@author: marc.frey@astrazeneca.com
"""
import argparse


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors,Crippen

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run K-Means Clustering with PCA')
    parser.add_argument('-p', '--pca_path', type=str, help='path of the normalized pca csv file (e.g. Output of cluster_MELLODDY_PCA_KM)', required=True)
    args = parser.parse_args()
    return args    
    
def plotClustersFromPcaCSV(norm_pca_csv):
    
    parent = Path(norm_pca_csv).parent
    
    #read pca data from csv plot clusters and output png to same directory
    descriptors_pca = pd.read_csv(norm_pca_csv)
    plotClustersFromPcaTable(descriptors_pca, path_to = norm_pca_csv)
    
def plotClustersFromPcaTable(descriptors_pca, path_to):
    
    t1 = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    t2 = datetime.now()
    
    path = Path(path_to)
    parent = path.parent
    
    plt.rcParams['axes.linewidth'] = 1.5
    plt.figure(figsize=(10,10))
        
    color_code={ 0:      '#FF00FF',\
                  1.0:   '#0000FF',\
                  2.0:   '#00FFFF',\
                  3.0:   '#008000',\
                  4.0:   '#273d48',\
                  5.0:   '#FFFF00',\
                  6.0:   '#800000',\
                  7.0:   '#FFA500',\
                  8.0:   '#FF0000',\
                  9.0:   '#000000',
                  }

        # 0 pink
        # 1 dark blue
        # 2 aqua
        # 3 green
        # 4 blue-grey
        # 5 yellow
        # 6 brown-red
        # 7 orange
        # 8 red orange
        # 9 black
        
        
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)

    for i in descriptors_pca.index[:]: 
        
        plt.plot(descriptors_pca.loc[i].at['PC1_normalized'],descriptors_pca.loc[i].at['PC2_normalized'],
                    c=color_code[descriptors_pca.loc[i].at['Cluster_PC1_PC2']],
                    marker='o',markersize=5, 
                    #label = set(color_code[descriptors_pca.loc[i].at['Cluster_PC1_PC2']]),
                    # markeredgecolor='k',
                    alpha=0.5)

    circle = plt.Circle((0,0), 1, color='gray', fill=False,clip_on=True,linewidth=1.5,linestyle='--')
    plt.tight_layout()    

    plt.xlabel ('PC1',fontsize=14,fontweight='bold')
    plt.ylabel ('PC2',fontsize=14,fontweight='bold')
    plt.tick_params ('both',width=2,labelsize=12)
    
    plt.savefig(str(parent) + "/" + str(path.stem) + "_clusters_PCA_kmeans_" +str(t1) +".png", dpi=300)
    
 
    
args = vars(init_arg_parser())
path = args["pca_path"]


plotClustersFromPcaCSV(path)

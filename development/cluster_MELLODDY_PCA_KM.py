# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:48:59 2020

@author: kfrl011
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

import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
# from rdkit import AllChem


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run K-Means Clustering with PCA')
    parser.add_argument('-p', '--t2_path', type=str, help='path of the T2 input file', required=True)
    parser.add_argument('-c', '--no_comps', type=int, help='number of PCs used for PCA', default = 2, choices=[2, 3])
    args = parser.parse_args()

    return args


def calculateFP(smiles):
    m1 = Chem.MolFromSmiles(smiles)
    #m1 = preprocessMolecule(m1) # not required for example code  -- 05/06 Lewis & Marc
    fp = AllChem.GetMorganFingerprintAsBitVect(m1,2, nBits=2048)
    bitstring = list(map(int,list(fp.ToBitString())))
    return (bitstring, m1)

def read_data(p):
    # df = pd.read_csv(p) 
    df = pd.read_csv(p, nrows = 100) 
    return df


def getFPs(row):
    X=[]
    y=[]
    comps = []
    # for ids, val in enumerate(df):
    
        #calculate FPs (and preprocess/remove unwanted chemistry/standardise the molecules)
    try:
        bitstring, m =calculateFP(row.iloc[1])
        X = bitstring
        comps = row.iloc[0]
        #threshold the subset of the data using 10um (activity threshold)
        #y.append(1 if df["Standard Value"].iloc[ids] >= 10 else 0)
    except TypeError: pass 
    #assert len(X) == len(y), "X and y must be of same size"
    return np.array(X), comps

def getMolsFromSmiles(df):
    """
    Params:     df : pandas df, containing compound_id and smiles.
    Returns:    df : pandas df, containing compound_id and molecule. Dropping invalid ones.
    """
    comp_ids = []
    molecules = []
    
    for idx, val in enumerate(df.iloc[:,1]):
        try:
            mol = Chem.MolFromSmiles(val)
            molecules.append(mol)
            comp_ids.append(df.iloc[idx,0])
        except:
            "pass"
    df = pd.DataFrame({"compound_id": comp_ids, "molecule":molecules})
    return df


def getPropsFromMol(mol):
    
    MolWt = Descriptors.ExactMolWt(mol) #Mol weight
    TPSA  = Chem.rdMolDescriptors.CalcTPSA(mol) #Topological Polar Surface Area
    nRotB = Descriptors.NumRotatableBonds (mol) #Number of rotable bonds
    HBD   = Descriptors.NumHDonors(mol) #Number of H bond donors
    HBA   = Descriptors.NumHAcceptors(mol) #Number of H bond acceptors
    LogP  = Descriptors.MolLogP(mol) #LogP

    return (MolWt, TPSA, nRotB, HBD, HBA, LogP)


def getPropsFromMols(mols_df):
    # comp_ids = pd.Series("compound_id")
    props = pd.DataFrame()
    comp_ids = []
    
    # get properties for each molecule
    for idx, mol in enumerate(mols_df["molecule"]):
        try:
            props = props.append(pd.DataFrame(list(getPropsFromMol(mol))).T, ignore_index=True)
            comp_ids.append(mols_df.iloc[idx,0])
        except:
            pass
    
    # prepare compounds df
    comp_ids = pd.DataFrame(comp_ids)
    comp_ids.reset_index(drop=True, inplace=True)
    comp_ids.rename(columns={0:"compound_id"}, inplace = True)
    
    # prepare props df
    props.reset_index(drop=True, inplace=True)
    props.rename(columns={0:'MolWt',1:'TPSA',2:'nRotB',3:'HBD',4:'HBA',5:'LogP'}, inplace = True)
    
    # merge compounds and properties 
    props_df = pd.concat([comp_ids, props], axis=1)
    
    return props_df
    
def getPCAs(props_df):
    descriptors = props_df.loc[:, ['MolWt', 'TPSA', 'nRotB', 'HBD','HBA', 'LogP']].values
    descriptors_std = StandardScaler().fit_transform(descriptors)
    pca = PCA()
    descriptors_2d = pca.fit_transform(descriptors_std)
    descriptors_pca= pd.DataFrame(descriptors_2d)
    descriptors_pca.index = props_df.index
    descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]

    # print(pca.explained_variance_ratio_) 
    # print(sum(pca.explained_variance_ratio_))

    return descriptors_pca


def plotPCA(desc_pca):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    ax.plot(desc_pca['PC1'],desc_pca['PC2'],'o',color='k')
    ax.set_title ('Principal Component Analysis',fontsize=16,fontweight='bold',family='sans-serif')
    ax.set_xlabel ('PC1',fontsize=14,fontweight='bold')
    ax.set_ylabel ('PC2',fontsize=14,fontweight='bold')
    
    plt.tick_params ('both',width=2,labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
def plotPCAnorm(descriptors_pca):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    ax.plot(descriptors_pca['PC1_normalized'],descriptors_pca['PC2_normalized'],'o',color='k')
    ax.set_title ('Principal Component Analysis',fontsize=16,fontweight='bold',family='sans-serif')
    ax.set_xlabel ('PC1',fontsize=14,fontweight='bold')
    ax.set_ylabel ('PC2',fontsize=14,fontweight='bold')
    
    plt.tick_params ('both',width=2,labelsize=12)
    
    plt.tight_layout()
    plt.show()  
    
    
def normalizePCAs(descriptors_pca):
    
    # This normalization will be performed just for PC1 and PC2, but can be done for all the components.
    scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
    scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))
    scale3 = 1.0/(max(descriptors_pca['PC3']) - min(descriptors_pca['PC3']))
    
    try:
        # And we add the new values to our PCA table
        descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
        descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]
        descriptors_pca['PC3_normalized']=[i*scale2 for i in descriptors_pca['PC3']]
    except:
        pass
    
    return descriptors_pca

def normalizedPCAs2csv(norm_PCA):
    global no_pcs
    
    if no_pcs == 2:
        pcas = norm_PCA.iloc[:,-3:]
    else:
        pcas = norm_PCA.iloc[:,-4:]
        
    pcas.to_csv(str(parent) + "/" +str(path.stem) + "___normalized_" + str(no_pcs) + "_PCs___" + str(t1) + ".csv", index = False)


def plotClusterHistogram(cluster_csv_path):
    """
    Parameters
    ----------
    cluster_csv_path : path_to_csv
        csv structured as "T2_clustered_kmeans" output, as resulting from running clustering [compound_id, Cluster_PC1_PC2]

    Returns
    -------
    None. Saves fig. to directory the source csv is located in.

    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime    
    
    t1 = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    CB91_Purple = '#9D2EC5'
    CB91_Purple = '#00FFFF'
    
    cluster_csv_path = Path(cluster_csv_path)
    parent_dir = cluster_csv_path.parent
    data = pd.read_csv(cluster_csv_path)
    
    plt.box(False)
    plt.hist(data["Cluster_PC1_PC2"], rwidth = 0.8, bins = 10, color = CB91_Purple)
    plt.title("Sizes of clusters")
    plt.xlabel("Cluster No.")
    plt.ylabel("No. of samples")
    plt.savefig(str(parent_dir) + "/" + str(cluster_csv_path.stem) + "___cluster_size_histogram___" +str(t1) +".png")
    plt.show()
    
    
#read console parameters
args = vars(init_arg_parser())
path = args["t2_path"]
no_pcs = args["no_comps"]

assert no_pcs in [2, 3], "-c please provide either 2 or 3 as number of Principal Components. Exiting..."

# path = "//projects/mai/Marc/MELLODDY_Github/data_prep/chembl25_github/chembl25_T2.csv"   
# path = "//samba.scp.astrazeneca.net/mai/Marc/MELLODDY_Github/data_prep/chembl25_github/chembl25_T2.csv"   
# path = "/projects/mai/Marc/T2_standardized_ChEMBL.csv"

from pathlib import Path
path = Path(path)
parent = path.parent


# read fc specified file
df = read_data(path)


# generate molecules
mols_df = getMolsFromSmiles(df)
# calculate properties for all molecules
props_df = getPropsFromMols(mols_df)
# apply pca and normalize results 
norm_PCA = normalizePCAs(getPCAs(props_df))

from  datetime import datetime
t1 = datetime.now().strftime('%Y%m%d_%H-%M-%S')

# perform k means clustering and append cluster column to df

# kmeans = KMeans(n_clusters=10, n_init = 10)
kmeans = KMeans(n_clusters=10, random_state=17) 


if no_pcs == 2:
    clusters = kmeans.fit(norm_PCA[['PC1_normalized','PC2_normalized']])
else:
    clusters = kmeans.fit(norm_PCA[['PC1_normalized','PC2_normalized', 'PC3_normalized']])
    
norm_PCA['Cluster_PC1_PC2'] = pd.Series(clusters.labels_, index=norm_PCA.index)

cluster_df = pd.concat([df, norm_PCA['Cluster_PC1_PC2']], axis = 1)


# write cluster results to csv
cluster_df.to_csv(str(parent) + "/" +str(path.stem) + "___clustered____" + str(no_pcs) + "_PCs___" + str(t1) + ".csv", index = False)

# write PCA results to csv (required for plotting clusters)
normalizedPCAs2csv(norm_PCA)

# plot histogram
# plotClusterHistogram(str(parent) + "/T2_clustered_kmeans_" + str(t1) + ".csv")











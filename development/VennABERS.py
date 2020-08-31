# Straight-forward implementation of IVAP algorithm described in:
# Large-scale probabilistic prediction with and without validity guarantees, Vovk et al.
# https://arxiv.org/pdf/1511.00213.pdf
#
# Paolo Toccaceli
#
# https://github.com/ptocca/VennABERS
#
# 2020-07-09: Fixed bug in p0 calculation

import numpy as np
from sklearn.model_selection import StratifiedKFold


# Some elementary functions to speak the same language as the paper
# (at some point we'll just replace the occurrence of the calls with the function body itself)
def push(x,stack):
    stack.append(x)

    
def pop(stack):
    return stack.pop()


def top(stack):
    return stack[-1]


def nextToTop(stack):
    return stack[-2]


# perhaps inefficient but clear implementation
def nonleftTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)<=0


def nonrightTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)>=0


def slope(a,b):
    ax,ay = a
    bx,by = b
    return (by-ay)/(bx-ax)


def notBelow(t,p1,p2):
    p1x,p1y = p1
    p2x,p2y = p2
    tx,ty = t
    m = (p2y-p1y)/(p2x-p1x)
    b = (p2x*p1y - p1x*p2y)/(p2x-p1x)
    return (ty >= tx*m+b)

kPrime = None

# Because we cannot have negative indices in Python (they have another meaning), I use a dictionary

def algorithm1(P):
    global kPrime
    
    S = []
    P[-1] = np.array((-1,-1))
    push(P[-1],S)
    push(P[0],S)
    for i in range(1,kPrime+1):
        while len(S)>1 and nonleftTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S


def algorithm2(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack

    F1 = np.zeros((kPrime+1,))
    for i in range(1,kPrime+1):
        F1[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i-1] = P[i-2]+P[i]-P[i-1]
        if notBelow(P[i-1],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonleftTurn(P[i-1],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i-1],Sprime)
    return F1


def algorithm3(P):
    global kPrime

    S = []
    push(P[kPrime+1],S)
    push(P[kPrime],S)
    for i in range(kPrime-1,0-1,-1):  # k'-1,k'-2,...,0
        while len(S)>1 and nonrightTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S


def algorithm4(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack
    
    F0 = np.zeros((kPrime+1,))
    for i in range(kPrime,1-1,-1):   # k',k'-1,...,1
        F0[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i] = P[i-1]+P[i+1]-P[i]
        if notBelow(P[i],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonrightTurn(P[i],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i],Sprime)
    return F0


def prepareData(calibrPoints):
    global kPrime
    
    ptsSorted = sorted(calibrPoints)
    
    xs = np.fromiter((p[0] for p in ptsSorted),float)
    ys = np.fromiter((p[1] for p in ptsSorted),float)
    ptsUnique,ptsIndex,ptsInverse,ptsCounts = np.unique(xs, 
                                                        return_index=True,
                                                        return_counts=True,
                                                        return_inverse=True)
    a = np.zeros(ptsUnique.shape)
    np.add.at(a,ptsInverse,ys)
    # now a contains the sums of ys for each unique value of the objects
    
    w = ptsCounts
    yPrime = a/w
    yCsd = np.cumsum(w*yPrime)   # Might as well do just np.cumsum(a)
    xPrime = np.cumsum(w)
    kPrime = len(xPrime)
    
    return yPrime,yCsd,xPrime,ptsUnique


def computeF(xPrime,yCsd):
    global kPrime
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})    
    P[kPrime+1] = P[kPrime] + np.array((1.0,0.0))    # The paper says (1,1)
    
    S = algorithm3(P)
    F0 = algorithm4(P,S)
    
    return F0,F1


def getFVal(F0,F1,ptsUnique,testObjects):
    pos0 = np.searchsorted(ptsUnique,testObjects,side='left')
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='right')+1
    return F0[pos0],F1[pos1]


def ScoresToMultiProbs(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
    
    # compute the F0 and F1 functions from the CSD
    F0,F1 = computeF(xPrime,yCsd)
    
    # compute the values for the given test objects
    p0,p1 = getFVal(F0,F1,ptsUnique,testObjects)
                    
    return p0,p1


""" 
Added functions

author: marc.frey@astrazeneca.com

"""
def orderp0p1(p_0, p_1):
    """
    :param list double p_0:  lower probability bound yielded from Venn-ABERS (ScoresToMultiProbs)
    :param list double p_1:  upper probability bound yielded from Venn-ABERS (ScoresToMultiProbs)
    :return tuple (p0, p1):  sorted tuple of p0 p1 values, where p0<p1
    """
    p0p1 = np.vstack((p_0,p_1)).T
    for id, val in enumerate(p0p1):
        if val[0]>val[1]:
            p0p1[id] = [val[1], val[0]]

    p0 = p0p1[:,0]
    p1 = p0p1[:,1]
    return(p0, p1)

def get_VA_margin_median(calibrPts, test_scores):
    p0,p1 = ScoresToMultiProbs(calibrPts, test_scores)
    # p0,p1 = orderp0p1(p0, p1)
    margin = [p1-p0]
    med = np.median(margin)
    return med

def get_VA_margin_mean(calibrPts, test_scores):
    p0,p1 = ScoresToMultiProbs(calibrPts, test_scores)
    # p0,p1 = orderp0p1(p0, p1)
    margin = [p1-p0]
    med = np.mean(margin)
    return med


def get_VA_median(calibrPts, test_scores):
    p0,p1 = ScoresToMultiProbs(calibrPts, test_scores)
    # p0,p1 = orderp0p1(p0, p1)
    # margin = [p1-p0]
    p0_med = np.median(p0)
    p1_med = np.median(p1)
    return p0_med, p1_med

def get_VA_mean(calibrPts, test_scores):
    p0,p1 = ScoresToMultiProbs(calibrPts, test_scores)
    # p0,p1 = orderp0p1(p0, p1)
    # margin = [p1-p0]
    p0_med = np.mean(p0)
    p1_med = np.mean(p1)
    return p0_med, p1_med

def get_VA_margin_mean_cross(pts):
    np.random.seed(1) 
    try:
    # exception handling in case number of class members is below n_splits
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 10)
    
        split_medians  = []
        split_medians_actives = []
        split_medians_inactives =[]
        X = pts[:,0]
        y = pts[:,1]
        

        for cal_index, test_index in skf.split(X, y):
            #print("TRAIN:", cal_index, "TEST:", test_index)
            X_cal, X_test = X[cal_index], X[test_index]
            y_cal, y_test = y[cal_index], y[test_index]

            calibrPts = np.vstack((X_cal, y_cal)).T
            calibrPts = [tuple(i) for i in calibrPts]
            test_scores = y_test
            
            # for each split get mean of margin
            split_medians.append(get_VA_margin_mean(calibrPts, test_scores))
            split_medians_actives.append(get_VA_margin_mean(calibrPts, test_scores[y_test==True]))
            split_medians_inactives.append(get_VA_margin_mean(calibrPts, test_scores[y_test==False]))
    except:
        return np.nan, np.nan, np.nan
    
    return np.mean(split_medians), np.mean(split_medians_actives), np.mean(split_medians_inactives)

def get_VA_margin_mean_cross_v2(pts, test):
# def get_VA_margin_mean_cross_v2(pts):    
    
    # like "get_VA_margin_median_cross" but takes in dedicted test instead of stratifying on the training set only
    
    np.random.seed(1) 
    try:
    # exception handling in case number of class members is below n_splits
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 10)
    
        split_medians_p0  = []
        split_medians_p1  = []
        # split_medians_actives = []
        # split_medians_inactives =[]
        X = pts[:,0]
        y = pts[:,1]
        

        for cal_index, test_index in skf.split(X, y):
            #print("TRAIN:", cal_index, "TEST:", test_index)
            X_cal, X_test = X[cal_index], X[test_index]
            y_cal, y_test = y[cal_index], y[test_index]

            calibrPts = np.vstack((X_cal, y_cal)).T
            calibrPts = [tuple(i) for i in calibrPts]
            test_scores = test
            # test_scores = y_test
            
            p0, p1 = get_VA_mean(calibrPts, test_scores)
            split_medians_p0.append(p0)
            split_medians_p1.append(p1)
            # split_medians_actives.append(get_VA_margin_median(calibrPts, test_scores[y_test==True]))
            # split_medians_inactives.append(get_VA_margin_median(calibrPts, test_scores[y_test==False]))
    except:
        return np.nan, np.nan
    
    return np.mean(split_medians_p0), np.mean(split_medians_p1)  

    
def get_VA_margin_median_cross_v2(pts, test):
    """
    like "get_VA_margin_median_cross" but takes in dedicted test instead of stratifying on the training set only
    """
    np.random.seed(1) 
    try:
    # exception handling in case number of class members is below n_splits
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 10)
    
        split_medians_p0  = []
        split_medians_p1  = []
        # split_medians_actives = []
        # split_medians_inactives =[]
        X = pts[:,0]
        y = pts[:,1]
        

        for cal_index, test_index in skf.split(X, y):
            #print("TRAIN:", cal_index, "TEST:", test_index)
            X_cal, X_test = X[cal_index], X[test_index]
            y_cal, y_test = y[cal_index], y[test_index]

            calibrPts = np.vstack((X_cal, y_cal)).T
            calibrPts = [tuple(i) for i in calibrPts]
            test_scores = test
            
            p0, p1 = get_VA_median(calibrPts, test_scores)
            split_medians_p0.append(p0)
            split_medians_p1.append(p1)
            # split_medians_actives.append(get_VA_margin_median(calibrPts, test_scores[y_test==True]))
            # split_medians_inactives.append(get_VA_margin_median(calibrPts, test_scores[y_test==False]))
    except:
        return np.nan, np.nan
    
    return np.mean(split_medians_p0), np.mean(split_medians_p1)    
"""

def get_VA_margin_mean_cross_v2(pts, test):
    
    # like "get_VA_margin_median_cross" but takes in dedicted test instead of stratifying on the training set only
    
    np.random.seed(1) 
    try:
    # exception handling in case number of class members is below n_splits
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 10)
    
        split_medians_p0  = []
        split_medians_p1  = []
        # split_medians_actives = []
        # split_medians_inactives =[]
        X = pts[:,0]
        y = pts[:,1]
        

        for cal_index, test_index in skf.split(X, y):
            #print("TRAIN:", cal_index, "TEST:", test_index)
            X_cal, X_test = X[cal_index], X[test_index]
            y_cal, y_test = y[cal_index], y[test_index]

            calibrPts = np.vstack((X_cal, y_cal)).T
            calibrPts = [tuple(i) for i in calibrPts]
            test_scores = test
            
            p0, p1 = get_VA_median(calibrPts, test_scores)
            split_medians_p0.append(p0)
            split_medians_p1.append(p1)
            # split_medians_actives.append(get_VA_margin_median(calibrPts, test_scores[y_test==True]))
            # split_medians_inactives.append(get_VA_margin_median(calibrPts, test_scores[y_test==False]))
    except:
        return np.nan, np.nan
    
    return np.mean(split_medians_p0), np.mean(split_medians_p1) 

"""
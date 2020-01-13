# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:57:46 2019

@author: xu
"""
############################
#### I - Import Package ####
############################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from tqdm import tqdm
from scipy.stats import bernoulli
from rbm import RBM, sample
from scipy import stats


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5
cm = plt.cm.get_cmap('tab10')


'binarized data set'    
def binarize_1D(X, bits):
    X_int=[]
    X_bin=[]
    x_min=np.min(X)
    x_max=np.max(X)
    N=X.shape[0]
    for k in range(N):
        max_value=2**(bits)-1
        x_int=int(max_value*((X[k]-x_min)/(x_max-x_min)))
        x_bin=('{0:0' + str(bits) + 'b}').format(x_int)
        x_bin=list(x_bin)
        x_bin=np.array([int(b) for b in x_bin])
        X_int.append(x_int)
        X_bin.append(x_bin)
    return(np.array(X_bin),np.array(X_int))

'de-binarized dataset'    
def bin_to_float(B, bits, min_data, max_data):
    N=B.shape[0]
    max_value=2**(bits)-1 
    X_real=[]
    X_int=[]
    for k in range(N):
        x_int=0
        for m in range(bits):
            x_int=x_int+ (2**m)*B[k][bits-1-m]
        x_real=min_data+(x_int*(max_data-min_data))/max_value
        X_real.append(x_real)
        X_int.append(x_int)
    return(np.array(X_real))
    

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))[0, 1]
                       
##########################
#### II - Import Data ####
##########################
df = pd.read_excel('./Input/VIX.xlsx',index_col = 'Date')
dft = df.T
Return = df['SPX 500'].pct_change()[1:]
Real_Data = np.array(Return)

#'## Graph: Initialization ##'
#plt.figure()
#plt.hist(Real_Data, bins = 50, alpha = 0.5)

Seed = 888
np.random.seed(Seed)

##########################################
#### III - Sample without replacement ####
##########################################

Res_Return = pd.DataFrame(index = Return.index)
Res_Return['Original'] = Return

for Ind_Simul in range(0, 5):
    Index = np.arange(Real_Data.shape[0])
    np.random.shuffle(Index)
    Res_Return[Ind_Simul] = Real_Data[Index]


Res_Track = Res_Return.add(1).cumprod()

plt.figure()
ax = Res_Track.resample('M').last()['Original'].plot(grid = False, color = "red", x_compat = True, linewidth = 2, style = "--", solid_capstyle = "round", title = 'Sample without replacement')
ax.legend()

Res_Track.resample('M').last().drop(['Original'], axis = 1).plot(grid = False, color = cm.colors, x_compat = True, linewidth = 1, solid_capstyle = "round", title = 'Sample without replacement', ax = ax, alpha = 0.6, legend = False)
#ax.get_legend().remove()
ax.set_xlabel("Date")
ax.set_ylabel("Value")
plt.savefig('./Output/Part_5_SPX_Boostrap_Sans_remise.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

Res_Return.apply(autocorr, axis = 0)

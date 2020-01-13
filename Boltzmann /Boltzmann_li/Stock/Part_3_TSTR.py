# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:39:36 2019

@author: xu
"""

############################
#### I - Import Package ####
############################
import tensorflow as tf
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from tqdm import tqdm
from rbm import RBM, sample
from scipy.stats import bernoulli
from scipy import stats

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5


##########################
#### II - Import Data ####
##########################
df = pd.read_excel('VIX.xlsx',index_col = 'Date')
dft = df.T
D_sp = df['SPX 500']
ret = np.array(D_sp.pct_change()[1:])
mu_r = ret.mean()
s_r = ret.std()


###############################
#### III - Data Processing ####
###############################

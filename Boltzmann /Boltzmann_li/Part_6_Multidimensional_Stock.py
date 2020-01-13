# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:57:46 2019

@author: xu
"""
############################
#### I - Import Package ####
############################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import tensorflow as tf
import seaborn as sns

from rbm import RBM, sample
from scipy import stats
from scipy.stats import bernoulli
from tqdm import tqdm

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5


Seed = 888
np.random.seed(Seed)
tf.random.set_random_seed(Seed)


def heatMap(df, title = 'Correlation'):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5, 5))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    
df = pd.read_csv('./Input/prixSPX.csv',index_col = 0, sep = ';')
Real_Data_Raw = df.pct_change()[1:]
Real_Data = Real_Data_Raw.copy()
for col in Real_Data.columns:
    Real_Data[col] = (Real_Data_Raw[col]  - Real_Data_Raw[col].mean())/Real_Data_Raw[col].std(ddof=0)
    
heatMap(Real_Data_Raw[1:])
plt.savefig('./Output/Part_6_2_MultiDimension_Real_Data_Correlation.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

'binarized data set'    
def binarize_1D(X, bits):
    X_int = []
    X_bin = []
    x_min = np.min(X)
    x_max = np.max(X)
    N = X.shape[0]
    for k in range(N):
        max_value = 2**(bits)-1
        x_int = int(max_value*((X[k]-x_min)/(x_max-x_min)))
        x_bin = ('{0:0' + str(bits) + 'b}').format(x_int)
        x_bin = list(x_bin)
        x_bin = np.array([int(b) for b in x_bin])
        X_int.append(x_int)
        X_bin.append(x_bin)
    return(np.array(X_bin),np.array(X_int))

'de-binarized dataset'    
def bin_to_float(B, bits, min_data, max_data):
    N = B.shape[0]
    max_value = 2**(bits)-1 
    X_real = []
    X_int = []
    for k in range(N):
        x_int = 0
        for m in range(bits):
            x_int = x_int + (2**m)*B[k][bits-1-m]
        x_real = min_data+(x_int*(max_data-min_data))/max_value
        X_real.append(x_real)
        X_int.append(x_int)
    return(np.array(X_real))

def binarize_multi(data, bits):
    B = []
    N_var = len(data.columns)
    l = data.columns
    for k in range(N_var):
        X = data[l[k]]
        x_bin, x_int = binarize_1D(X, bits)
        B.append(x_bin)
    B = pd.concat([pd.DataFrame(b) for b in B], axis = 1)
    return(B)

def debinarize_multi(data_b, bits, min_data, max_data):
    D_n = []
    N_var = len(max_data)
    for k in range(N_var):
        d = data_b[ :, (bits*k):(bits*k+bits)]
        min_d = min_data[k]
        max_d = max_data[k]
        d_n = bin_to_float(d, bits, min_d, max_d)
        D_n.append(d_n)
    D_n = pd.concat([pd.DataFrame(b) for b in D_n], axis = 1)
    return(D_n)

    
Bits = 16
Data_Train = binarize_multi(Real_Data, bits = Bits)


########################
#### III - RBM Model####
########################
tf.reset_default_graph()
'## Initialize RBM Model ##'
rbm_model = RBM(n_visible = 4*Bits, n_hidden = 30*4, lr = tf.constant(0.05, tf.float32), epochs = 50000, mode = 'bernoulli')

'## Placeholder for the visible layer of the RBM computation graph ##'
v = tf.placeholder(tf.float32, shape = [None, rbm_model.n_visible], name = "visible_layer")

'## Update rule ##'
k = 1
train_op = rbm_model.update(v, K = k)

'## Create session ##'
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

'## Training ##'
for epoch in tqdm(range(rbm_model.epochs)):
    idx = np.random.randint(0, Data_Train.shape[0], 500)
    batchXs = Data_Train.loc[idx]
    sess.run(train_op, feed_dict = {v: batchXs.values})

'## Reconstruction ##'
def reconstruction(v,steps = 1):
    for i in range(steps):
        hidden_p = rbm_model.get_probabilities('hidden', v)
        h_state = sample(hidden_p)
        visible_p = rbm_model.get_probabilities('visible', h_state)
        v = sample(visible_p)
    return(v)
    
    
#Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train.values})
#Reconstruction_Data = debinarize_multi(Reconstruction_Bin, Bits, Real_Data.min(axis = 0).tolist(), Real_Data.max(axis = 0).tolist())
#Reconstruction_Data.columns = df.columns
#heatMap(Reconstruction_Data)

##############################
##### III - Generate Data ####
##############################
'## Gibbs Sampling ##'
Noise = np.random.normal(0, 1, size = Data_Train.shape[0]*4).reshape(-1,4)
Noise_Bin = binarize_multi(pd.DataFrame(Noise), bits = Bits)   

plt.figure()
for ind in tqdm([5000]):
    Fake_Bin = sess.run(reconstruction(v, steps = ind), feed_dict = {v: Noise_Bin}) 
    Fake_Data = debinarize_multi(Fake_Bin, Bits, Real_Data.min().tolist(), Real_Data.max().tolist())
    Fake_Data.columns = df.columns
    heatMap(Fake_Data, 'Iter = ' + str(ind))
plt.savefig('./Output/Part_6_3_Fake_Data_Correlation.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

Fake_Data_Raw = Fake_Data.copy()
for col in Fake_Data.columns:
    Fake_Data_Raw[col] = Fake_Data[col]*Real_Data_Raw[col].std(ddof=0) + Real_Data_Raw[col].mean()

heatMap(Fake_Data_Raw)
Result = (df.pct_change()[1:]).quantile([0.01, 0.99])
Result = Result.append(Fake_Data_Raw.quantile([0.01, 0.99]), ignore_index = True)
print(Result)


for ind in Fake_Data_Raw.columns:
    plt.figure()
    bins = np.linspace(np.min(Real_Data_Raw[ind]), np.max(Real_Data_Raw[ind]), 50)
    plt.hist(Fake_Data_Raw[ind], bins = bins, alpha = 0.5, label='Fake')
    plt.hist(Real_Data_Raw[ind], bins = bins, alpha = 0.5, label='Real')
    plt.legend()  
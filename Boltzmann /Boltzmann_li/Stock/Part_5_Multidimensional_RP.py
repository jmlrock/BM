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


##########################
#### II - Import Data ####
##########################
df = pd.read_excel('Data_For_Jules.xlsx', sheet_name = 'Ptf_1', index_col = 'END DATE')
Data_Raw = df[['Equity_EMU', 'Equity_HK', 'Equity_UK', 'Equity_US_Big', 'Equity_Japan', 'Bonds_Australia_10Y', 'Bonds_Canada_10Y', 'Bonds_Germany_10Y', 'Bonds_UK_10Y', 'Bonds_US_10Y']]
Data_Raw.columns = ['Equity EMU', 'Equity HK', 'Equity UK', 'Equity US Big', 'Equity Japan', 'Bonds Australia 10Y', 'Bonds Canada 10Y', 'Bonds Germany 10Y', 'Bonds UK 10Y', 'Bonds US 10Y']
Return_Raw = Data_Raw.pct_change()[1:]

def heatMap(df, title = 'Correlation'):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    


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
    

Bits = 24
Data_Train = binarize_multi(Return_Raw, bits = Bits)

Seed = 88
np.random.seed(Seed)
########################
#### III - RBM Model####
########################
tf.reset_default_graph()
'## Initialize RBM Model ##'
rbm_model = RBM(n_visible = Return_Raw.shape[1]*Bits, n_hidden = 640, lr = tf.constant(0.2, tf.float32), epochs = 50000, mode = 'bernoulli')

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
    
    
Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train.values})
Reconstruction_Data = debinarize_multi(Reconstruction_Bin, Bits, Return_Raw.min().tolist(), Return_Raw.max().tolist())
Reconstruction_Data.columns = Data_Raw.columns

heatMap(Return_Raw, title = 'Orginal')
heatMap(Reconstruction_Data, title = 'Reconstruction')

Reconstruction_Data.index = Return_Raw.index
for ind in Reconstruction_Data.columns:
    plt.figure()
    plt.hist(Reconstruction_Data[ind], bins = 50, alpha = 0.5, label='Fake')
    plt.hist(Return_Raw[ind], bins =50, alpha = 0.5, label='Real')
    plt.legend()

    plt.figure()
    Reconstruction_Data[ind].add(1).cumprod().plot()
    Return_Raw[ind].add(1).cumprod().plot()
    plt.legend(['Fake', 'Real'])

###############################
###### III - Generate Data ####
###############################
#'## Gibbs Sampling ##'
#Noise = np.random.normal(0, 1, size = Data_Train.shape[0]*4).reshape(-1,4)
#Noise_Bin = binarize_multi(pd.DataFrame(Noise), bits = Bits)   
#
#for ind in tqdm([1, 10, 100, 10000]):
#    Fake_Bin = sess.run(reconstruction(v, steps = ind), feed_dict = {v: Noise_Bin}) 
#    Fake_Data = debinarize_multi(Fake_Bin, Bits, Data_Raw.min().tolist(), Data_Raw.max().tolist())
#    Fake_Data.columns = Data_Raw.columns
#    heatMap(Fake_Data, 'Iter = ' + str(ind))
#
#plt.figure()
#plt.hist(Fake_Data['Sample 1'], bins = 50, alpha = 0.5, label='Fake')
#plt.hist(Data_Raw['Sample 1'], bins =50, alpha = 0.5, label='Real')
#plt.legend()
#
#plt.figure()
#plt.hist(Fake_Data['Sample 2'], bins = 50, alpha = 0.5, label='Fake')
#plt.hist(Data_Raw['Sample 2'], bins =50, alpha = 0.5, label='Real')
#plt.legend()
#
#plt.figure()
#plt.hist(Fake_Data['Sample 3'], bins = 50, alpha = 0.5, label='Fake')
#plt.hist(Data_Raw['Sample 3'], bins =50, alpha = 0.5, label='Real')
#plt.legend()
#
#plt.figure()
#plt.hist(Fake_Data['Sample 4'], bins = 50, alpha = 0.5, label='Fake')
#plt.hist(Data_Raw['Sample 4'], bins =50, alpha = 0.5, label='Real')
#plt.legend()
#
#
#plt.figure()
#Data_Raw[['Sample 1', 'Sample 2']].rolling(100).corr().xs('Sample 1', level = 1, drop_level=True)['Sample 2'].plot()
#Fake_Data[['Sample 1', 'Sample 2']].rolling(100).corr().xs('Sample 1', level = 1, drop_level=True)['Sample 2'].plot()
    
    
Reconstruction_Bin_1 = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train.values})
Reconstruction_Data_1 = debinarize_multi(Reconstruction_Bin_1, Bits, Return_Raw.min().tolist(), Return_Raw.max().tolist())
Reconstruction_Data_1.columns = Data_Raw.columns

Reconstruction_Bin_2 = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train.values})
Reconstruction_Data_2 = debinarize_multi(Reconstruction_Bin_2, Bits, Return_Raw.min().tolist(), Return_Raw.max().tolist())
Reconstruction_Data_2.columns = Data_Raw.columns

heatMap(Return_Raw, title = 'Orginal')
heatMap(Reconstruction_Data_1, title = 'Reconstruction 1')
heatMap(Reconstruction_Data_2, title = 'Reconstruction 2')

Reconstruction_Data_1.index = Return_Raw.index
Reconstruction_Data_2.index = Return_Raw.index
for ind in Reconstruction_Data_1.columns:
    plt.figure()
    plt.hist(Reconstruction_Data_1[ind], bins = 50, alpha = 0.5, label='Fake 1')
    plt.hist(Reconstruction_Data_2[ind], bins = 50, alpha = 0.5, label='Fake 2')
    plt.hist(Return_Raw[ind], bins = 50, alpha = 0.5, label='Real')
    plt.legend()

    plt.figure()
    Reconstruction_Data_1[ind].add(1).cumprod().plot()
    Reconstruction_Data_2[ind].add(1).cumprod().plot()
    Return_Raw[ind].add(1).cumprod().plot()
    plt.legend(['Fake 1', 'Fake 2', 'Real'])



'## Gibbs Sampling ##'
Noise = np.random.normal(0, 1, size = Return_Raw.shape[0]*Return_Raw.shape[1]).reshape(-1, Return_Raw.shape[1])
Noise_Bin = binarize_multi(pd.DataFrame(Noise), bits = Bits)   

for ind in tqdm([1, 10, 100, 1000]):
    Fake_Bin = sess.run(reconstruction(v, steps = ind), feed_dict = {v: Noise_Bin}) 
    Fake_Data = debinarize_multi(Fake_Bin, Bits, Return_Raw.min().tolist(), Return_Raw.max().tolist())
    Fake_Data.columns = Return_Raw.columns
    heatMap(Fake_Data, 'Iter = ' + str(ind))

for ind in Fake_Data.columns:
    plt.figure()
    plt.hist(Fake_Data[ind], bins = 50, alpha = 0.5, label='Fake')
    plt.hist(Return_Raw[ind], bins = 50, alpha = 0.5, label='Real')
    plt.legend()

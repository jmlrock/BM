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


############################
#### II - Generate Data ####
############################
'Gaussian Mixture 4 Samples'
def gauss_mix_multi(N, pi, p_anti):
    sample = []
    anti_sample = []
    for i in range(N):
        delta = int(bernoulli.rvs(pi, loc = 0, size = 1))
        m_1 = np.random.normal(0, 0.1, size=1)
        m_2 = np.random.normal(0.2,0.05 ,size=1)
        mixiture = (1-delta)*m_1 + delta*m_2
        sample.append(mixiture)
        
        x = int(bernoulli.rvs(p_anti, loc = 0, size = 1))
        if x==1:
            anti_mixiture = -mixiture
        else:
            anti_mixiture = np.random.normal(0, 0.05, size = 1)
        anti_sample.append(anti_mixiture)
    
    normal_1 = np.random.normal(0, 0.1, size=N)
    normal_2 = np.random.normal(-0.4, 0.1, size=N)
        
    return((np.array(sample), np.array(anti_sample), np.array(normal_1), np.array(normal_2)))


def heatMap(df, title = 'Correlation'):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5, 5))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    
M_1, M_2, M_3, M_4 = gauss_mix_multi(5000, 0.5, 0.9)
Data_Raw = pd.concat([pd.DataFrame(M_1), pd.DataFrame(M_2), pd.DataFrame(M_3), pd.DataFrame(M_4)], axis = 1)
Data_Raw.columns = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4']

plt.figure()
plt.hist(M_1, bins = 50, alpha = 1)
plt.hist(M_2, bins = 50, alpha = 1)
plt.hist(M_3, bins = 50, alpha = 1)
plt.hist(M_4, bins = 50, alpha = 1)
plt.legend(Data_Raw.columns)


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
    
heatMap(Data_Raw)


Bits = 24
Data_Train = binarize_multi(Data_Raw, bits = Bits)

Seed = 88
np.random.seed(Seed)
########################
#### III - RBM Model####
########################
tf.reset_default_graph()
'## Initialize RBM Model ##'
rbm_model = RBM(n_visible = 4*Bits, n_hidden = 500, lr = tf.constant(0.2, tf.float32), epochs = 50000, mode = 'bernoulli')

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
Reconstruction_Data = debinarize_multi(Reconstruction_Bin, Bits, Data_Raw.min().tolist(), Data_Raw.max().tolist())
Reconstruction_Data.columns = Data_Raw.columns


##############################
##### III - Generate Data ####
##############################
'## Gibbs Sampling ##'
Noise = np.random.normal(0, 1, size = Data_Train.shape[0]*4).reshape(-1,4)
Noise_Bin = binarize_multi(pd.DataFrame(Noise), bits = Bits)   

for ind in tqdm([1, 10, 100, 10000]):
    Fake_Bin = sess.run(reconstruction(v, steps = ind), feed_dict = {v: Noise_Bin}) 
    Fake_Data = debinarize_multi(Fake_Bin, Bits, Data_Raw.min().tolist(), Data_Raw.max().tolist())
    Fake_Data.columns = Data_Raw.columns
    heatMap(Fake_Data, 'Iter = ' + str(ind))

plt.figure()
plt.hist(Fake_Data['Sample 1'], bins = 50, alpha = 0.5, label='Fake')
plt.hist(Data_Raw['Sample 1'], bins =50, alpha = 0.5, label='Real')
plt.legend()

plt.figure()
plt.hist(Fake_Data['Sample 2'], bins = 50, alpha = 0.5, label='Fake')
plt.hist(Data_Raw['Sample 2'], bins =50, alpha = 0.5, label='Real')
plt.legend()

plt.figure()
plt.hist(Fake_Data['Sample 3'], bins = 50, alpha = 0.5, label='Fake')
plt.hist(Data_Raw['Sample 3'], bins =50, alpha = 0.5, label='Real')
plt.legend()

plt.figure()
plt.hist(Fake_Data['Sample 4'], bins = 50, alpha = 0.5, label='Fake')
plt.hist(Data_Raw['Sample 4'], bins =50, alpha = 0.5, label='Real')
plt.legend()


plt.figure()
Fake_Data[['Sample 1', 'Sample 2']].rolling(100).corr().xs('Sample 1', level = 1, drop_level=True)['Sample 2'].plot()
Data_Raw[['Sample 1', 'Sample 2']].rolling(100).corr().xs('Sample 1', level = 1, drop_level=True)['Sample 2'].plot()

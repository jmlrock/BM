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


############################
#### II - Generate Data ####
############################
'Gaussian Mixture'
def gauss_mix(N, pi):
    sample = []
    for i in range(N):
        delta = int(bernoulli.rvs(pi, loc = 0, size = 1))
        m_1 = np.random.normal(-2, 0.1, size = 1)
        m_2 = np.random.normal(-1.8, 0.05, size = 1)
        mixiture = (1-delta)*m_1 + delta*m_2
        sample.append(mixiture)
    return(np.array(sample))

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
    
    
Bits = 16
Real_Data = gauss_mix(10000, 0.5)
#'## Graph: Initialization ##'
#plt.figure()
#bins = np.linspace(np.min(Real_Data), np.max(Real_Data), 50)
#plt.hist(Real_Data, bins = bins, alpha = 0.5)
#plt.title('Gaussian Mixture Data Samples')
#plt.savefig('./Output/Part_1_Gaussian_Mixture_1_SimulatedData.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

Data_Train, _= binarize_1D(Real_Data, bits = Bits)


Seed = 888
np.random.seed(Seed)
tf.random.set_random_seed(Seed)

########################
#### III - RBM Model####
########################
tf.reset_default_graph()
'## Initialize RBM Model ##'
rbm_model = RBM(n_visible = Bits, n_hidden = 12, lr = tf.constant(0.05, tf.float32), epochs = 2000, mode = 'bernoulli')

'## Placeholder for the visible layer of the RBM computation graph ##'
v = tf.placeholder(tf.float32, shape = [None, rbm_model.n_visible], name = "visible_layer")


'## Update rule ##'
k = 1
train_op = rbm_model.update(v, K = k)

#'## Free energy ##'
#energy = rbm_model.free_energy(v = v)
#tf.summary.scalar('free_energy', tf.reduce_mean(energy))
#
#'## Merge summaries for Tensorboard visualization ##'
#summary = tf.summary.merge_all()

'## Create session ##'
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

'## Training ##'
for epoch in tqdm(range(rbm_model.epochs)):
    idx = np.random.randint(0, Data_Train.shape[0], 100)
    batchXs = Data_Train[idx]
    sess.run(train_op, feed_dict = {v: batchXs})

'## Reconstruction ##'
def reconstruction(v,steps = 1):
    for i in range(steps):
        hidden_p = rbm_model.get_probabilities('hidden', v)
        h_state = sample(hidden_p)
        visible_p = rbm_model.get_probabilities('visible', h_state)
        v = sample(visible_p)
    return(v)
    
Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train})
Reconstruction_Data = bin_to_float(Reconstruction_Bin, Bits, np.min(Real_Data), np.max(Real_Data))
Recap_2 = pd.DataFrame([Real_Data.flatten(), Reconstruction_Data.flatten()]).T
Recap_2.columns = ['Real', 'Reconstruction']


##############################
##### III - Generate Data ####
##############################
'## Gibbs Sampling ##'
Noise = np.random.normal(-2, 0.1, size = Data_Train.shape[0]).reshape(-1,1)
Noise_Bin, _= binarize_1D(Noise, bits = Bits)   

Fake_Bin = sess.run(reconstruction(v, steps = 1000), feed_dict = {v: Noise_Bin}) 
Fake_Data = bin_to_float(Fake_Bin, Bits, np.min(Real_Data), np.max(Real_Data))

'## Graph: Comparaison Histogram ##'
plt.figure()
bins = np.linspace(np.min(Fake_Data), np.max(Fake_Data), 35)
plt.hist(Noise, bins = bins, alpha = 0.5, label = 'Fake')
plt.hist(Real_Data, bins = bins, alpha = 0.5, label = 'Orginal')
plt.xlabel('Value')
plt.ylabel('Number')
#plt.title('Iteration = ' + str(1000))
plt.legend()
plt.savefig('./Output/Part_1_Gaussian_Mixture_1_1_Noise_Hist.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

plt.figure()
bins = np.linspace(np.min(Fake_Data), np.max(Fake_Data), 35)
plt.hist(Fake_Data, bins = bins, alpha = 0.5, label = 'Fake')
plt.hist(Real_Data, bins = bins, alpha = 0.5, label = 'Orginal')
plt.xlabel('Value')
plt.ylabel('Number')
#plt.title('Iteration = ' + str(1000))
plt.legend()
plt.savefig('./Output/Part_1_Gaussian_Mixture_1_2_Fake_Hist.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

'## Table: Comparaison Statitics ##'
Recap = pd.DataFrame([Real_Data.flatten(), Fake_Data.flatten()]).T
Recap.columns = ['Real', 'Fake']
Result = Recap.quantile([0.01, 0.99])
Result = Result.append(Recap.mean(), ignore_index = True)
Result = Result.append(Recap.std(), ignore_index = True)
Result = np.round(Result, 3)
Result.index = ['1st percentile', '99th percentile', 'Mean', 'Standard deviation']
print(Result)


'## Graph: Comparaison QQ-Plot ##'
plt.figure()
plt.scatter(np.sort(Real_Data.flatten()), np.sort(Noise.flatten()))
x = np.array([np.min(Real_Data), np.max(Real_Data)])
y = np.array([np.min(Real_Data), np.max(Real_Data)])
plt.plot(x,y)
plt.xlabel('Real')
plt.ylabel('Reconstruction')
plt.title('QQ plot')
plt.savefig('./Output/Part_1_Gaussian_Mixture_2_1_Noise_QQPlot.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
stats.ks_2samp(Real_Data.flatten(), Noise.flatten())

plt.figure()
plt.scatter(np.sort(Real_Data.flatten()), np.sort(Fake_Data.flatten()))
x = np.array([np.min(Real_Data), np.max(Real_Data)])
y = np.array([np.min(Real_Data), np.max(Real_Data)])
plt.plot(x,y)
plt.xlabel('Real')
plt.ylabel('Reconstruction')
plt.title('QQ plot')
plt.savefig('./Output/Part_1_Gaussian_Mixture_2_2_Fake_QQPlot.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
stats.ks_2samp(Real_Data.flatten(), Fake_Data.flatten())
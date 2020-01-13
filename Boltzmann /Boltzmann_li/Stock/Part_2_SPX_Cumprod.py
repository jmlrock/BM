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
    
    
##########################
#### II - Import Data ####
##########################
df = pd.read_excel('VIX.xlsx',index_col = 'Date')
dft = df.T
Real_Data = np.array(df['SPX 500'].pct_change()[1:])

Bits = 20
#'## Graph: Initialization ##'
#plt.figure()
#plt.hist(Real_Data, bins = 50, alpha = 0.5)
Data_Train, _= binarize_1D(Real_Data, bits = Bits)


########################
#### III - RBM Model####
########################
tf.reset_default_graph()
'## Initialize RBM Model ##'
rbm_model = RBM(n_visible = Bits, n_hidden = 64, lr = tf.constant(0.2, tf.float32), epochs = 20000, mode = 'bernoulli')

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
    idx = np.random.randint(0, Data_Train.shape[0], 250)
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
print(stats.ks_2samp(Recap_2.Real, Recap_2.Reconstruction))

'## Verification ##'
plt.figure()
plt.scatter(np.sort(Real_Data.flatten()), np.sort(Reconstruction_Data.flatten()))
x = np.array([np.min(Real_Data), np.max(Real_Data)])
y = np.array([np.min(Real_Data), np.max(Real_Data)])
plt.plot(x,y)
plt.xlabel('Real')
plt.ylabel('Reconstruction')

plt.figure()
plt.hist(Reconstruction_Data, bins = 50, alpha = 0.5)
plt.hist(Real_Data, bins = 50, alpha = 0.5)
plt.xlabel('Value')
plt.ylabel('Number')

'## Simulation ##'
plt.figure()
Recap = pd.DataFrame()
Recap['Real'] = Real_Data.flatten()

for Ind in range(5):
    Sample_Bin = sess.run(reconstruction(v),feed_dict={v: Data_Train})
    Sample_Data= bin_to_float(Sample_Bin, Bits, np.min(Real_Data), np.max(Real_Data))
    Recap['Sample' + str(Ind+1)] = Sample_Data.flatten()

Recap.add(1).cumprod().plot()



#plt.figure()
#plt.scatter(np.sort(Sample_Track.Real), np.sort(Sample_Track.Sample))
#x = np.array([np.min(Sample_Track.Real), np.max(Sample_Track.Real)])
#y = np.array([np.min(Sample_Track.Real), np.max(Sample_Track.Real)])
#plt.plot(x,y)
#
#Sample_Track.mean()
#Sample_Track.std()
#Sample_Track.quantile([0.01, 0.99])

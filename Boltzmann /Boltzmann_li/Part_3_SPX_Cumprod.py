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

Seed = 888
np.random.seed(Seed)
tf.random.set_random_seed(Seed)

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
Real_Data_Raw = np.array(df['SPX 500'].pct_change()[1:])
Real_Data = (Real_Data_Raw - Real_Data_Raw.mean())/Real_Data_Raw.std(ddof=0)

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
rbm_model = RBM(n_visible = Bits, n_hidden = 32, lr = tf.constant(0.2, tf.float32), epochs = 50000, mode = 'bernoulli')

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
    idx = np.random.randint(0, Data_Train.shape[0], 500)
    batchXs = Data_Train[idx]
    sess.run(train_op, feed_dict = {v: batchXs})

'## Reconstruction ##'
def reconstruction(v, steps = 1):
    for i in range(steps):
        hidden_p = rbm_model.get_probabilities('hidden', v)
        h_state = sample(hidden_p)
        visible_p = rbm_model.get_probabilities('visible', h_state)
        v = sample(visible_p)
    return(v)
 
plt.figure()
fig = plt.figure()
ax = df['SPX 500'].resample('M').last().plot(color = 'r', linestyle = 'dashed', linewidth = 2)


for Ind in range(0, 10):
    Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train})
    Reconstruction_Data = bin_to_float(Reconstruction_Bin, Bits, np.min(Real_Data), np.max(Real_Data))

    Data_Simul = pd.DataFrame([Real_Data, Reconstruction_Data], index = ['Original', 'Fake'], columns = df['SPX 500'].index[1:]).transpose()
    Return_Simul = Data_Simul*Real_Data_Raw.std(ddof=0) + Real_Data_Raw.mean()
    (Return_Simul['Fake'].add(1).cumprod()*df['SPX 500'][0]).resample('M').last().plot(ax = ax,  alpha = 0.6)

plt.savefig('./Output/Part_3_SPX_3_Ten_Fake_Samples.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)


Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train})
Reconstruction_Data = bin_to_float(Reconstruction_Bin, Bits, np.min(Real_Data), np.max(Real_Data))
Reconstruction_Data_Raw = Reconstruction_Data*Real_Data_Raw.std(ddof=0) + Real_Data_Raw.mean()

plt.figure()
bins = np.linspace(np.min(Real_Data_Raw), np.max(Real_Data_Raw), 50)
plt.hist(Reconstruction_Data_Raw, bins = bins, alpha = 0.5, label = 'Fake')
plt.hist(Real_Data_Raw, bins = bins, alpha = 0.5, label = 'Orginal')
plt.xlabel('Value')
plt.ylabel('Number')
plt.legend()
plt.savefig('./Output/Part_3_SPX_1_Hist.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

plt.figure()
Data_Simul = pd.DataFrame([Real_Data_Raw, Reconstruction_Data_Raw], index = ['Original', 'Fake'], columns = df['SPX 500'].index[1:]).transpose()
Data_Simul.plot(alpha = 0.5)
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.savefig('./Output/Part_3_SPX_2_Return_Plot.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)

plt.figure()
(Data_Simul.add(1).cumprod()*df['SPX 500'][0]).resample('M').last().plot()
plt.savefig('./Output/Part_3_SPX_4_One_Fake_Sample.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
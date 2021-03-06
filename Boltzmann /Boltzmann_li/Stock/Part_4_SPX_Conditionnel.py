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
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
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
Real_Data = np.array(df['SPX 500'].pct_change()[1:])

Bits = 24
#'## Graph: Initialization ##'
#plt.figure()
#plt.hist(Real_Data, bins = 50, alpha = 0.5)
Data_Train, _= binarize_1D(Real_Data, bits = Bits)
Signal = np.roll((Real_Data>0)*1, 1).reshape([Data_Train.shape[0],1])
Data_Train = np.append(Data_Train, Signal, axis = 1)

Seed = 888
np.random.seed(Seed)

########################
#### III - RBM Model####
########################
tf.reset_default_graph()
'## Initialize RBM Model ##'
rbm_model = RBM(n_visible = Bits + 1, n_hidden = 26, lr = tf.constant(0.2, tf.float32), epochs = 100000, mode = 'bernoulli')

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
def reconstruction(v,steps = 1):
    for i in range(steps):
        hidden_p = rbm_model.get_probabilities('hidden', v)
        h_state = sample(hidden_p)
        visible_p = rbm_model.get_probabilities('visible', h_state)
        v = sample(visible_p)
    return(v)
    
Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train})
Reconstruction_Data = bin_to_float(Reconstruction_Bin[:, :-1], Bits, np.min(Real_Data), np.max(Real_Data))
Recap_2 = pd.DataFrame([Real_Data.flatten(), Reconstruction_Data.flatten()]).T
Recap_2.columns = ['Real', 'Reconstruction']
print(stats.ks_2samp(Recap_2.Real, Recap_2.Reconstruction))
Data_Simul = pd.DataFrame([Real_Data, Reconstruction_Data], index = ['Original', 'Fake'], columns = df['SPX 500'].index[1:]).transpose()
Data_Simul.add(1).cumprod().plot()


Seed = 666
np.random.seed(Seed)
Data_Simul = pd.DataFrame(Real_Data, columns = ['Original'], index = df['SPX 500'].index[1:])

for ind in range(0, 5):
    Reconstruction_Bin = sess.run(reconstruction(v, steps = 1), feed_dict = {v: Data_Train})
    Reconstruction_Data = bin_to_float(Reconstruction_Bin[:, :-1], Bits, np.min(Real_Data), np.max(Real_Data))
    Data_Simul['Fake' + str(ind)] = Reconstruction_Data
    
Res = Data_Simul.add(1).cumprod().resample('M').last()*1229.23

plt.figure()
ax = Res['Original'].plot(figsize = (10, 5), grid = False, color = "black", x_compat = True, linewidth = 2, style = "--", solid_capstyle = "round")

Res.drop(['Original'], axis = 1).plot(figsize = (10, 5), grid = False, color = cm.colors, x_compat = True, linewidth = 1.5, solid_capstyle = "round", ax = ax, alpha = 0.75)
ax.legend()
#ax.get_legend().remove()
ax.set_xlabel("Date")
#ax.set_ylabel("Value")
plt.savefig('./Output/Part_4_SPX_Conditonal.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)


'## Verification ##'
plt.figure()
plt.scatter(np.sort(Real_Data.flatten()), np.sort(Reconstruction_Data.flatten()))
x = np.array([np.min(Real_Data), np.max(Real_Data)])
y = np.array([np.min(Real_Data), np.max(Real_Data)])
plt.plot(x,y)
plt.xlabel('Real')
plt.ylabel('Reconstruction')
plt.title('QQ plot')
print(np.corrcoef(Real_Data, Reconstruction_Data)[0, 1])


plt.figure()
bins = np.linspace(np.min(Real_Data), np.max(Real_Data), 50)
plt.hist(Reconstruction_Data, bins = bins, alpha = 0.5, label = 'Fake')
plt.hist(Real_Data, bins = bins, alpha = 0.5, label = 'Orginal')
plt.xlabel('Value')
plt.ylabel('Number')
plt.legend()
plt.title('Histogram')

print(autocorr(Real_Data))
print(autocorr(Reconstruction_Data))

'Statistics'
print(Recap_2.mean())
print(Recap_2.std())
print(Recap_2.skew())
print(Recap_2.kurtosis())
print(Recap_2.quantile([0.01, 0.99]))


#Test = Data_Train[0, :].reshape(1, 24)
#Res = []
#for Ind in range(0, 100):
#    print(Ind)
#    Reconstruction_Bin = sess.run(reconstruction(v, steps = 100), feed_dict = {v: Test})
#    Reconstruction_Data = bin_to_float(Reconstruction_Bin, Bits, np.min(Real_Data), np.max(Real_Data))
#    Res.append(Reconstruction_Data[0])
#    Test = Reconstruction_Bin
#        
#
#
#
#for Ind in range(0, 5):
#    Reconstruction_Bin = sess.run(reconstruction(v, steps = 10), feed_dict = {v: Data_Train})
#    Reconstruction_Data = bin_to_float(Reconstruction_Bin, Bits, np.min(Real_Data), np.max(Real_Data))
#    Data_Simul['Fake' + str(Ind)] = Reconstruction_Data
#
#Data_Simul.add(1).cumprod().plot()

###############################
###### III - Generate Data ####
###############################
#'## Gibbs Sampling ##'
#Noise = np.random.normal(0, 1, size = Data_Train.shape[0]).reshape(-1,1)
#Noise_Bin, _= binarize_1D(Noise, bits = Bits)   
#
#for ind in tqdm([1, 10, 100, 1000, 10000]):
#    Fake_Bin = sess.run(reconstruction(v, steps = ind), feed_dict = {v: Noise_Bin}) 
#    Fake_Data = bin_to_float(Fake_Bin, Bits, np.min(Real_Data), np.max(Real_Data))
#    
#    '## Graph: Comparaison ##'
#    plt.figure()
#    bins = np.linspace(np.min(Real_Data), np.max(Real_Data), 50)
#    plt.hist(Fake_Data, bins = bins, alpha = 0.5, label = 'Fake')
#    plt.hist(Real_Data, bins = bins, alpha = 0.5, label = 'Orginal')
#    plt.xlabel('Value')
#    plt.ylabel('Number')
#    plt.title('Iteration = ' + str(ind))
#    plt.legend()
#    plt.savefig('./Output/Part_2_SPX_4_sampling_N=' + str(ind) + '.pdf', format = 'pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
#
#'Statistics'
#Recap = pd.DataFrame([Real_Data.flatten(), Fake_Data.flatten()]).T
#Recap.columns = ['Real', 'Fake']
#Recap.mean()
#Recap.std()
#Recap.quantile([0.01, 0.99])
#stats.ks_2samp(Recap.Real, Recap.Fake)

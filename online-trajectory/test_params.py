# The corresponding tutorial for this code was released EXCLUSIVELY as a bonus
# If you want to learn about future bonuses, please sign up for my newsletter at:
# https://lazyprogrammer.me

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from sklearn.utils import shuffle
from datetime import datetime

import os
import sys
from rnn_class.lstm import LSTM
from rnn_class.gru import GRU

from rnn import *

# @gh
dt = T.config.floatX
df = pd.read_csv('record.txt', engine='python')
df.columns = ['x', 'y', 't']
x = df.x.as_matrix()
y = df.y.as_matrix()
t = df.t.as_matrix()
N = len(x)

hidden_layers = []
hidden_layers.append({'GRU'  : 20});
hidden_layers.append({'LSTM' : 10});
hidden_layers.append({'GRU'  : 25});
hidden_layers.append({'GRU'  : 24});
hidden_layers.append({'LSTM' : 15});
hidden_layers.append({'LSTM' : 5});
hidden_layers.append({'GRU' : 15});
hidden_layers.append({'LSTM' : 25});

order = 2
# let's see if we can use D past values to predict the next value
D = 5
print N
print D

series = np.zeros((N, order), dtype=dt);
series[:,0] = x;
series[:,1] = y;
#series[:,2] = t;

plt.plot(series[:,0], series[:,1], label="Input data")

smin = series.min(0)
smax = series.max(0)
series = series - smin
series = series / smax

n = N - D
X = np.zeros((n, order, D), dtype=dt)
for d in xrange(D):
    X[:,:,d] = series[d:d+n,:]
Y = series[D:D+n, :]

print "series length:", n
Xtrain = X[:n/2,:,:]
Ytrain = Y[:n/2,:]
Xtest = X[n/2:,:,:]
Ytest = Y[n/2:,:]

Ntrain = len(Xtrain)
Ntest = len(Xtest)
Xtrain = Xtrain.reshape(Ntrain, D, order)
Xtest = Xtest.reshape(Ntest, D, order)

model_file_name = "models/rnn.mdl"

learning_rate=10e-2
epochs=350

train = False
train = True

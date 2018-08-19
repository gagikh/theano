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
hidden_layers.append({'TRNN'  : 20});
hidden_layers.append({'LSTM' : 10});

order = 2
series = np.zeros((N, order), dtype=dt);
series[:,0] = x;
series[:,1] = y;
#series[:,2] = t;

# series = (series - series.mean()) / series.std() # normalize the values so they have mean 0 and variance 1

plt.plot(series[:,0], series[:,1], label="Input data")

smin = series.min(0)
smax = series.max(0)
series = series - smin
series = series / smax

# let's see if we can use D past values to predict the next value
#for D in (5):
for D in range(5, 6):
    D = 5
    n = N - D
    print N
    print D
    print n
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

    model = RNN(hidden_layers)
    model.fit(Xtrain, Ytrain, activation=T.tanh, learning_rate=10e-2, mu=0.5, reg=0, epochs=250, show_fig=False)
    print "train score:", model.score(Xtrain, Ytrain)
    print "test score:", model.score(Xtest, Ytest)

    train_series = np.zeros((n, Xtrain.shape[2]), dtype=dt)
    train_series[:n/2] = model.predict(Xtrain) 
    train_series[n/2:] = np.nan
    # prepend d nan's since the train series is only of size N - D
    c = np.full((d, Xtrain.shape[2]), np.nan, dtype=dt)
    d = np.concatenate([c, train_series]);
    #train_series = Xtrain
    #d = model.predict(Xtrain) 
    d = d * smax + smin
    plt.plot(d[:,0], d[:,1], label="Train data")

    test_series = np.zeros((n, Xtest.shape[2]), dtype=dt)
    test_series[:n/2] = np.nan
    test_series[n/2:] = model.predict(Xtest)
    d = np.concatenate([c, test_series]);
    d = d * smax + smin
    plt.plot(d[:,0], d[:,1], label="Test data")

leg = plt.legend(loc='upper right', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

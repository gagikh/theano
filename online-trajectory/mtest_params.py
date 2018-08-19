# The corresponding tutorial for this code was released EXCLUSIVELY as a bonus
# If you want to learn about future bonuses, please sign up for my newsletter at:
# https://lazyprogrammer.me

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
#import theano.tensor as T
from sklearn.utils import shuffle
from datetime import datetime

import os
import sys

# @gh
# N = 2000
df = pd.read_csv('mrecord.txt', engine='python')
df.columns = ['x', 'y', 't', 'i']

x = df.x.as_matrix()
y = df.y.as_matrix()
t = df.t.as_matrix()
i = df.i.as_matrix()

N = np.max(i)

for k in range(1, N + 1):
    j = (i == k).nonzero()
    plt.plot(x[j], y[j], label="Input data " + str(k))

plt.legend(loc='best')

minx = x.min(0)
miny = y.min(0)
mint = t.min(0)

maxx = x.max(0)
maxy = y.max(0)
maxt = t.max(0)

x = (x - minx) / maxx
y = (y - miny) / maxy
t = (y - mint) / maxt

K = 3 * len(x)
X = np.zeros(K)
# merge
X[range(0,K,3)] = x
X[range(1,K,3)] = y
X[range(2,K,3)] = t

Xtrain = X[:(K/2)]
Ytrain = i[:(K/2)]
Xtest = X[(K/2):]
Ytest = i[(K/2):]

Ntrain = len(Xtrain)
Ntest = len(Xtest)

model_file_name = "models/mrnn.mdl"

learning_rate=10e-2
epochs=350

train = False
train = True
#plt.show()

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

from mrnn import *
from mtest_params import *

model = MRNN(N)
if train:
    model.fit(Xtrain, Ytrain, Etrain, learning_rate=learning_rate, epochs=epochs, show_fig=False)
else:
    model.load_model(model_file_name)

print "train score:", model.score(Xtrain, Ytrain)
print "test score:", model.score(Xtest, Ytest)

train_series = np.zeros((n, Xtrain.shape[2]), dtype=dt)
train_series[:n/2] = model.predict(Xtrain)
train_series[n/2:] = np.nan

# prepend d nan's since the train series is only of size N - D
c = np.full((d, Xtrain.shape[2]), np.nan, dtype=dt)
d = np.concatenate([c, train_series]);
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

if train:
    print "Saving model into ", model_file_name
    model.save_model(model_file_name)

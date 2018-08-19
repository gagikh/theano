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
import cPickle as pickle

import os
import sys
from rnn_class.lstm import LSTM
from rnn_class.gru import GRU

def myr2(T, Y):
    Ym = T.mean(0)
    TY = T - Y;
    TYT = TY.transpose()
    sse = TYT.dot(TY)
    TYm = T - Ym;
    TYmT = TYm.transpose()
    sst = TYmT.dot(TYm)
    return 1 - sse / sst

class RNN(object):
    def __init__(self, layers, shape, activation=T.tanh):
	tmp, self.t, self.D = shape
	Mi = self.D
	mu=0.5
	layer_id = 0;

        self.hidden_layers = []
        for layer in layers:
	    keys = layer.keys()
	    for key in keys:
		Mo = layer[key]
		print "Adding ", key, " cell with ", Mo, " layers"
		name = key + str(layer_id)
		if key == "GRU":
		    ru = GRU(Mi, Mo, activation, name)
		elif key == "LSTM":
		    ru = LSTM(Mi, Mo, activation, name)
		else:
		    print "Invalid key: ", key

		layer_id = layer_id + 1;
		self.hidden_layers.append(ru)
		Mi = Mo

	# inputs count - 5 (t)
	# outputs count - 4 (hidden_layers)
	# output dims - 2 (D)
        # return Z.dot(self.Wo) + self.bo
	# 4x2 * 2x2 + 2x2
	# input 5x2
        Wo = self.init_weight(Mi, self.D)
        bo = self.init_weight(self.t, self.D)
        self.Wo = theano.shared(Wo, "Wo")
        self.bo = theano.shared(bo, "bo")
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params


        lr = T.scalar('lr')
        thX = T.matrix('X')
        thY = T.vector('Y')
        Yhat = self.forward(thX)[-1]

        # let's return py_x too so we can draw a sample instead
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=Yhat,
            allow_input_downcast=True,
        )
        
        cost = T.mean((thY - Yhat)*(thY - Yhat))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.train_op = theano.function(
            inputs=[lr, thX, thY],
            outputs=cost,
            updates=updates
        )

    def init_weight(self, M1, M2):
	return np.random.randn(M1, M2) / np.sqrt(M1 + M2)

    def fit(self, X, Y, learning_rate=10e-2, epochs=2000, show_fig=False):
	N, t, D = X.shape
	assert(t == self.t)
	assert(D == self.D)

        costs = []
        for i in xrange(epochs):
            t0 = datetime.now()
            X, Y = shuffle(X, Y)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in xrange(N):
                c = self.train_op(learning_rate, X[j], Y[j])
                cost += c

            if i % 5 == 0:
                print "i:", i, "cost:", cost, "time for epoch:", (datetime.now() - t0)
            if (i+1) % 500 == 0:
                learning_rate /= 10
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.output(Z)
        return Z.dot(self.Wo) + self.bo

    def score(self, X, Y):
        Yhat = self.predict(X)
        return myr2(Y, Yhat)

    def predict(self, X):
        N = len(X)
        Yhat = np.zeros((N, X.shape[2]))
        for i in xrange(N):
            Yhat[i,:] = self.predict_op(X[i])
        return Yhat

    def save_model(self, f):
	ps = {}
	for p in self.params:
	    ps[p.name] = p.get_value()
	    print "Saving parameter", p.name
	pickle.dump(ps, open(f, "wb"))

    def load_model(self, f):
	ps = pickle.load(open(f, "rb"))
	for p in self.params:
	    print "Loading parameter", p.name
	    p.set_value(ps[p.name])

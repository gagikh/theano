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
from rnn_class.mlstm import MLSTM
from rnn_class.mgru import MGRU

def myr2(T, Y):
    Ym = T.mean(0)
    TY = T - Y;
    TYT = TY.transpose()
    sse = TYT.dot(TY)
    TYm = T - Ym;
    TYmT = TYm.transpose()
    sst = TYmT.dot(TYm)
    return 1 - sse / sst

class MRNN(object):
    def __init__(self, N):
	self.N = N
	self.M = 1

	layer_id = 0;

        self.hidden_layers = []
        for i in range(self.N):
	    lstm = MLSTM(self.N, self.M, "MLSTM_" + str(i))
	    self.hidden_layers.append(lstm)

	self.mgru = MGRU(self.N, self.M, "MGRU")
        self.params = [self.mgru.params]

        e = self.init_weight(1, self.M)

        for ru in self.hidden_layers:
            self.params += ru.params

	self.define_forward_op()
	self.define_train_op()
	self.define_predict_op()

    def init_weight(self, M1, M2):
	return np.random.randn(M1, M2) / np.sqrt(M1 + M2)

    def predict(self, X, Y):
        N = len(X)
        Ihat = np.zeros((N, X.shape[2]))
        Ehat = np.zeros((1, N))
        for i in xrange(N):
            Ihat[i,:], Ehat = self.predict_op(X[i], Y[i], Ehat)
        return Ihat, Ehat

    def fit(self, X, Y, E, learning_rate=10e-2, epochs=2000, show_fig=False):
	N, t, X.shape
	assert(t == self.t)

        costs = []
	eT = zeros(1, M)
        for i in xrange(epochs):
            t0 = datetime.now()
            X, Y, E = shuffle(X, Y, E)
            n_correct = 0
            n_total = 0
            cost = 0

            for j in xrange(N):
                c, eT=self.train_op(learning_rate, X[j], Y[j], E[j], eT)
                cost += c

            if i % 5 == 0:
                print "i:", i, "cost:", cost, "time for epoch:", (datetime.now() - t0)

            if (i + 1) % 500 == 0:
                learning_rate /= 10

            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def define_predict_op(self):
        X = T.vector('X')
        Y = T.vector('Y')
        eT= T.vector('eT')
        [xs, eT1, idx] = self.forward_op(X, Y, eT)

        self.predict_op = theano.function(
            inputs=[X, Y, eT],
            outputs=[idx, eT1],
            allow_input_downcast=True,
        )

    def define_forward_op(self):

        A = T.matrix('A')
        X = T.matrix('X')
        Y = T.matrix('Y')
        eT= T.matrix('eT')

	# C_t1 is the pairwise distance, NxM
	# C_t1 = T.pdist(X=X, Y=Y, metric="euclidean", n_jobs=-1)
	x = T.extra_ops.repeat(X, X.shape[0], axis=0)
	y = Y.transpose()
	c = x - y
	C_t1 = c * c;

	# A = [0,1](Nx(M + 1))
        for i in range(self.N):
	    lstm = self.hidden_layers[i]
	    o = lstm.output(C_t1)
	    T.set_subtensor(A[i, :], o[0,:])

	[xs, x, eT1, es] = self.mgru.output(X, A)

	idx = T.max(eT1)
        self.forward_op = theano.function(
            inputs=[X, Y, eT],
            outputs=[xs, eT1, idx],
            allow_input_downcast=True,
        )

    def define_train_op(self):

        lr = T.scalar('lr')

        XT = T.vector('XT')
        YT = T.matrix('YT')
        E  = T.vector('E')
        eT = T.vector('eT')

	[xs, eT1, idx] = self.forward_op(XT, YT, eT)

	mu=0.5
	LAMBDA = 0.7
	qsi = 0.1
	k = 0.1
	v = 0.1

	L = E * log(eT1) + (1 - E) * log(1 - eT1)
	ds = myr2(Xt1, xs);
	d  = myr2(Xt1, x);
	cost = LAMBDA / N * ds + k / N * d + v * L * eT1 + qsi * es

        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]
        self.train_op = theano.function(
            inputs=[lr, XT, YT, E, eT],
            outputs=[cost, eT1],
            updates=updates
        )

    def score(self, X, Y, E):
        Ehat = self.predict_op(X, Y)
        return myr2(E, Ehat)

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

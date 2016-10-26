import theano, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from theano import tensor as T
from random import randint, shuffle

def load_data(filename, M):
        data = np.genfromtxt(filename, delimiter=',')
        X = data[:,0]
        Y = data[:,1]
        N = len(X)
        train = np.zeros((N+1, M+2+1), dtype=float)
        train[0:N,0] = X
        train[0:N,1] = Y
        train[0:N,M+2] = 1
        return train

def hidden_layer():
	x = T.dvector("x"); # [x, 1]
	w = T.dmatrix("w"); # w[h][2]
        y = T.dot(w, x.T)
	return theano.function(inputs=[x, w], outputs=y)

def activation_fcn():
	y = T.dvector("y");
	F = theano.function(inputs=[y], outputs=T.nnet.sigmoid(y))
        return F

def gradient_fcn():
	y = T.dvector("y");
	G = theano.function(inputs=[y], outputs=y * (1 - y))
        return G

def output_layer():
	x = T.dvector("x");
	w = T.dvector("w");

        y = T.dot(x, w.T)
	return theano.function(inputs=[x, w], outputs=y)

def train_net(L, F, G, O, gamma, alpha, X, W0, W1):

    N = len(X) - 1
    dw0_t = np.empty_like(W0)
    dw1_t = np.empty_like(W1)

    dw0_t[:] = 0
    dw1_t[:] = 0

    for i in range(0, N):
        # x, xd, 0,0,0, 1
        x = X[i]

        # First layer
        l = L(x, W0)
        a = F(l)
        g = G(a)
        
        o = O(a, W1)

        # Diff
        do = x[1] - o

        # Propogation
        # for the last layer, F(so) = so, so F'(so) = 1
        # So, the deltao = d

        dwo = (gamma * do * a)

        dh  = g * do * W1

        dw1 = dwo + alpha * dw1_t

        z = np.matlib.repmat(x, W0.shape[0], 1)

        tmp = np.multiply(dh, z.T)
        dw0 = (gamma * tmp.T) + alpha * dw0_t

        W0 += dw0
        W1 += dw1

        # Update momentum
        dw0_t = dw0
        dw1_t = dw1

        # Update context state data
        X[i+1, 2:(X.shape[1] - 1)] = a;

def test_net (L, F, O, X, W0, W1):

    N = len(X) - 1
    Z = np.zeros((N,), dtype=float)

    for i in range(0, N):
        x = X[i]
        l = L(x, W0)
        a = F(l)
        o = O(a, W1)
        Z[i] = o

    return Z

def main():
    # Hidden units
    # M contexts, 2 inputs
    H = 1
    M = H
    # W = (H+1)
    W0 = np.random.uniform(low=-0.5, high=0.5, size=(H, M+2+1))
    W0[:,2:(M+2+1)] = 1
    W1 = np.random.uniform(low=-0.5, high=0.5, size=(M,))

    X = load_data('signal.txt', M);

    L = hidden_layer()
    F = activation_fcn()
    G = gradient_fcn()
    O = output_layer()

    gamma = 0.01
    # momentum function
    alpha = 0.01

    train_net(L, F, G, O, gamma, alpha, X, W0, W1)

    X = load_data('test_signal.txt', M);
    N = len(X)
    X[1:N, 1] = X[0:(N-1), 1]
    X[0, 1] = 0 # expect the previous value

    Z = test_net (L, F, O, X, W0, W1)

    E = load_data('expected_signal.txt', M);
    Y = E[0:-1, 1]

    N = range(0, len(Z));
    plt.plot(N, Y, 'b')
    plt.plot(N, Z, 'r')
    plt.plot(N, Y-Z, 'g')
    plt.show()

main()

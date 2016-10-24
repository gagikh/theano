import theano, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from theano import tensor as T
from random import randint, shuffle

def load_data(filename):
        train = np.genfromtxt(filename, delimiter=',')
        Y = np.empty_like(train[:,1])
        Y[:] = train[:,1]
        train[:,1] = 1
        return (train, Y)

def first_layer():
	x = T.dvector("x"); # [x, 1]
	w = T.dmatrix("w"); # w[h][2]
        y = T.dot(w, x.T)
	return theano.function(inputs=[x, w], outputs=y)

def second_layer():
	x = T.dvector("x");
	w = T.dvector("w");

        y = T.dot(x, w.T)
	return theano.function(inputs=[x, w], outputs=y)

def activation_fcn():
	y  = T.dvector("y");
	F  = theano.function(inputs=[y], outputs=T.sin(y))
	Fg = theano.function(inputs=[y], outputs=T.cos(y))
        return (F, Fg)

def train_net(L0, L1, F, Fg, gamma, X, Y, w0, w1):

    N = len(Y)
    for i in range(0, N):
        x = X[i]
        y = Y[i] 

        # First layer
        l = L0(x, w0)
        a = F (l)
        ag= Fg(l) 
        
        # Second layer
        o  = L1(a, w1)
        do = y - o

        # Propogation
        # for the last layer, F(so) = so, so F'(so) = 1
        # So, the deltao = d

        dwo = (gamma * do * a)
        # TODO: Update then mult????
        dh  = ag * do * w1
        w1 += dwo

        #dw00 = (gamma * dh * w0[:,0])
        #dw01 = (gamma * dh * w0[:,1])
        dw00 = (gamma * dh * x[0])
        dw01 = (gamma * dh * x[1]) # * 1

        w0[:,0] += dw00
        w0[:,1] += dw01

def test_net (L0, L1, F, X, Y, w0, w1):
    test = X

    N = len(test)
    Z = np.empty_like(Y)

    for i in range(0, N):
        x = X[i]
        l = L0(x, w0)
        a = F (l)
        o = L1(a, w1)
        Z[i] = o

    return Z

def main():
    H = 5
    # W = (H+1)x2
    # w9[:,0] = xn
    # w0[:,1] = thetta 
    w0 = np.random.uniform(low=-0.5, high=0.5, size=(H+1, 2))
    for i in range(1, H+1):
        w0[i - 1][0] = i;

    # We assign these values to have sin(nx + theta) = 1
    pi = 3.141592653
    w0[H][0] = 0
    w0[H][1] = pi / 2

    w1 = np.random.uniform(low=-0.5, high=0.5, size=(H+1,))

    X,Y = load_data('signal.txt');

    L0 = first_layer()
    L1 = second_layer()
    F,Fg = activation_fcn()
    gamma = 0.01

    train_net(L0, L1, F, Fg, gamma, X, Y, w0, w1)

    Z = test_net (L0, L1, F, X, Y, w0, w1)

    N = range(0, len(Y));
    plt.plot(N, Y, 'b')
    plt.plot(N, Z, 'r')
    plt.show()

main()

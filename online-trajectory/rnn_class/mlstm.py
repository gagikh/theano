import numpy as np
import theano
import theano.tensor as T

from util import init_weight

class MLSTM:
    def __init__(self, N, M, layer_name):
        # numpy init

        WCt1 = init_weight(M, 1)
        Whi0 = init_weight(N, N)
        bh = np.zeros((N, 1))
        c = init_weight(M+1, N)
        bc = np.zeros((M+1, 1))

        # theano vars
        self.WCt1   = theano.shared(WCt1,   layer_name + "_WCt1")
        self.Whi0   = theano.shared(Whi0,   layer_name + "_Whi0")
        self.bh	    = theano.shared(bh,	    layer_name + "_bh")
        self.c	    = theano.shared(c,	    layer_name + "_c")
        self.bc	    = theano.shared(bc,	    layer_name + "_bc")

        self.params = [self.WCt1, self.Whi0, self.bh, self.c]

        h_i0 = init_weight(N, 1)
        c_i0 = init_weight(N, 1)
        self.h_i0   = theano.shared(h_i0,   layer_name + "_h_i0")
        self.c_i0   = theano.shared(c_i0,   layer_name + "_c_i0")

    def recurrence(self, C_t1, h_i0, c_i0):
	# x = Nx1
	x = C_t1.dot(self.WCt1) + self.Whi0.dot(h_i0) + self.bh

	# s = Nx1
        s = T.nnet.sigmoid(x)
	# t = Nx1
        t = T.tanh(x)

	# c_i1 = Nx1
	c_i1 = s * c_i0 + s * t;

	# h_i1 = Nx1
        h_i1 = s * T.tanh(c_i1)

	# Nx(M+1)
	Ai_t1 = T.transpose(T.nnet.softmax(c_i1.dot(h_i1) + self.bc));
	return [Ai_t1, h_i1, c_i1]

    def output(self, C_t1):
	Ai, self.h_i0, self.c_i0 = self.recurrence(C_t1, self.h_i0, self.c_i0)
	return Ai

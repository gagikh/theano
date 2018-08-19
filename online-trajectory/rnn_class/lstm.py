# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import numpy as np
import theano
import theano.tensor as T

from util import init_weight

class LSTM:
    def __init__(self, Mi, Mo, activation, layer_name):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi  = np.zeros(Mo)
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf  = np.zeros(Mo)
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc  = np.zeros(Mo)
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo  = np.zeros(Mo)
        c0  = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxi = theano.shared(Wxi, layer_name + "_Wxi")
        self.Whi = theano.shared(Whi, layer_name + "_Whi")
        self.Wci = theano.shared(Wci, layer_name + "_Wci")
        self.bi  = theano.shared(bi,  layer_name + "_bi")
        self.Wxf = theano.shared(Wxf, layer_name + "_Wxf")
        self.Whf = theano.shared(Whf, layer_name + "_Whf")
        self.Wcf = theano.shared(Wcf, layer_name + "_Wcf")
        self.bf  = theano.shared(bf,  layer_name + "_bf")
        self.Wxc = theano.shared(Wxc, layer_name + "_Wxc")
        self.Whc = theano.shared(Whc, layer_name + "_Whc")
        self.bc  = theano.shared(bc,  layer_name + "_bc")
        self.Wxo = theano.shared(Wxo, layer_name + "_Wxo")
        self.Who = theano.shared(Who, layer_name + "_Who")
        self.Wco = theano.shared(Wco, layer_name + "_Wco")
        self.bo  = theano.shared(bo,  layer_name + "_bo")
        self.c0  = theano.shared(c0,  layer_name + "_c0")
        self.h0  = theano.shared(h0,  layer_name + "_h0")
        self.params = [
            self.Wxi,
            self.Whi,
            self.Wci,
            self.bi,
            self.Wxf,
            self.Whf,
            self.Wcf,
            self.bf,
            self.Wxc,
            self.Whc,
            self.bc,
            self.Wxo,
            self.Who,
            self.Wco,
            self.bo,
            self.c0,
            self.h0,
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        # input X should be a matrix (2-D)
        # rows index time
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0, self.c0],
            n_steps=x.shape[0],
        )
        return h

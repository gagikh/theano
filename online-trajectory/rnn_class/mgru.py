import numpy as np
import theano
import theano.tensor as T

from util import init_weight

class MGRU:
    def __init__(self, N, M, layer_name):
	# Predict
	self.params = []
        Wht = init_weight(N, N)
        Wxt = init_weight(N, N)
        bx  = init_weight(N, 1)

        Pw = init_weight(N, N)
        Pb = init_weight(N, 1)

        self.Wht = theano.shared(Wht, layer_name + "_Wht")
        self.Wxt = theano.shared(Wxt, layer_name + "_Wxt")
        self.bx  = theano.shared(bx,  layer_name + "_bx")
        self.Pw  = theano.shared(Pw,  layer_name + "_Pw")
        self.Pb  = theano.shared(Pb,  layer_name + "_Pb")
	self.params += [self.Wht, self.Wxt, self.bx, self.Pw, self.Pb]

	# Update
        WA = init_weight(N, 1)
        bA = init_weight(N, 1)
        Wxt1 = init_weight(N, N)
        bxt1 = init_weight(N, 1)
        Wet1 = init_weight(N, N)
        bet1 = init_weight(N, 1)

        self.WA   = theano.shared(WA,  layer_name + "_WA")
        self.bA   = theano.shared(bA,  layer_name + "_bA")
        self.Wxt1 = theano.shared(Wxt1,layer_name + "_Wxt1")
        self.bxt1 = theano.shared(bxt1,layer_name + "_bxt1")
        self.Wet1 = theano.shared(Wet1, layer_name + "_Wet1")
        self.bet1 = theano.shared(bet1, layer_name + "_bet1")
	self.params += [self.WA, self.bA, self.Wxt1, self.bxt1, self.Wet1, self.bet1]

	# birth/date
        h = init_weight(N, 1)
        x = init_weight(N, 1)
        self.h = theano.shared(h, layer_name + "_h")
        self.x = theano.shared(x, layer_name + "_x")
	self.params += [self.h, self.x]

        e_t0 = init_weight(N, 1)
        self.e_t0 = theano.shared(e_t0, layer_name + "_e_t0")

    def predict(self, h_t0, x_t0):
	# Prediction
        h_t1  = T.tanh(self.Wht.dot(h_t0) + self.Wxt.dot(x_t0) + self.bx)
	x_t1s = self.Pw.dot(h_t1) + self.Pb;
	return [x_t1s, h_t1]

    def update(self, h_t1, x_t1s, z_t1, A_t1, e_t0):
	# Update
	# (M+1)N
	print z_t1.ndim
	print x_t1s.ndim
	c = T.stack(z_t1, x_t1s, axes=1)
	a = A_t1.dot(c) * e_t0;
	# Nx1
	t = T.tanh(a.dot(WA) + h_t1 + bA);
	x_t1 = Wxt1.dot(t) + bxt1;
	e_t1 = T.nnet.sigmoid(Wet1.dot(t) + bet1);
	return [x_t1, e_t1]

    def birth(self, e_t0, e_t1):
	e_t1s = T.abs(e_t0 - e_t1)
	return e_t1s

    def output(self, z_t1, A_t1):
	x_t1s, h = self.predict(self.h, self.x) 
	self.x, e = self.update(h, x_t1s, z_t1, A_t1, self.e_t0)
	e_t1s = self.birth(self.e_t0, e);
	self.e_t0 = e;
	self.h = h
	return [self.x, x_ts1, self.e_t0, e_et1s]

    def loss(self, z_t1, A_t1):
	x, x_ts1, e, e_et1s = self.output(z_t1, A_t1)


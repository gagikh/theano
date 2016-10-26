from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.printing import pydotprint
import theano.d3viz as d3v

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from PIL import Image

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class Params:
    def __init__(self):
        # count of training samples
        self.batch_size = 500

        # height of training grayscale image
        self.im_height = 28

        # width of training grayscale image
        self.im_width = 28

        # Learn rate
        self.learning_rate = 0.1

        # rng state
        self.state = 23455

        # kernel size of first lenet layer
        self.nkerns0 = 3
        
        # kernel size of second lenet layer
        self.nkerns1 = 5

        # Localizing region height
        self.K0 = 5
        
        # Localizing region width
        self.K1 = 5

        self.poolsize = (2, 2)

def lenet5(config):

    x = T.matrix("x")
    # Pixel is edge or not
    y = T.ivector("y")

    assert 0 < config.batch_size
    assert 0 < config.im_height
    assert 0 < config.im_width

    image_shape0 = (config.batch_size, 1, config.im_height, config.im_width)
    layer0_input = x.reshape((config.batch_size, 1, config.im_height, config.im_width))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    rng = np.random.RandomState(config.state)

    # To be more precise, each neuron in the first hidden layer will be
    # connected to a small region of the input neurons, say, for example, a 5x5
    # region, corresponding to 25 input pixels. So, for a particular hidden
    # neuron, we might have connections that look like this: 

    assert 0 < config.nkerns0
    assert 0 < config.K0
    assert 0 < config.K1
    assert 0 < config.poolsize[0]
    assert 0 < config.poolsize[1]

    filter_shape0 = (config.nkerns0, 1, config.K0, config.K1)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=image_shape0,
        filter_shape=filter_shape0,
        poolsize=config.poolsize
    )
    h = (config.im_height - config.K0 + 1) / config.poolsize[0]
    w = (config.im_width  - config.K1 + 1) / config.poolsize[1]
    assert 0 < h
    assert 0 < w

    image_shape1 = (config.batch_size, config.nkerns0, h, w)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    assert 0 < config.nkerns1
    filter_shape1 = (config.nkerns1, config.nkerns0, config.K0, config.K1)

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=image_shape1,
        filter_shape=filter_shape1,
        poolsize=config.poolsize
    )
    h = (h - config.K0 + 1) / config.poolsize[0]
    w = (w - config.K1 + 1) / config.poolsize[1]
    assert 0 < h
    assert 0 < w

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    n_edge_pixels = h * w

    # construct a fully-connected sigmoidal layer

    layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=config.nkerns1 * h * w,
            n_out=n_edge_pixels,
            activation=T.tanh
            )

    # We should estimate the edges, so 1 output image will be produced
    layer3 = LogisticRegression(input=layer2.output, n_in=n_edge_pixels, n_out=1)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    assert 0 < config.learning_rate

    updates = [
        (param_i, param_i - config.learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x, y],
        cost,
        updates=updates
    )
    return train_model

if __name__ == '__main__':
    img = Image.open(open('lena.jpg'))
    # dimensions are (height, width, channel)
    img = np.asarray(img, dtype='float32') / 256.

    images = np.array([img[:,:,0], img[:,:,1], img[:,:,2]])

    params = Params()
    params.batch_size = 2
    params.im_height = 20
    params.im_width = 20
    params.K0 = 5
    params.K1 = 5

    model = lenet5(params)

    pydotprint(model, 'model.png')
    d3v.d3viz(model, 'model.html')

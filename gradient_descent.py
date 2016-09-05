import theano
import numpy as np
import pandas as pd

from theano import tensor as T
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename, sep=",",header=None)
    train  = data.iloc[:,:-1]
    labels = data.iloc[:,-1]

    return (train.as_matrix(), labels.as_matrix())

def normalize(train):
    (N,M) = train.shape

    for i in range(M):
        t = train[:,i]
        t = (t - np.mean(t)) / (np.max(t) - np.min(t))
        train[:,i] = t

def gradient_descent():
    train  = T.dmatrix("train")
    labels = T.dvector("labels")
    thetta = T.dvector("thetta")
    alpha  = T.dscalar("alpha")

    (N,M) = train.shape
    # (N,M) * (M,1) => (N,1)
    y = T.dot(train, thetta.T)
    d = y.T - labels
    z = T.dot(train.T, d) / N

    update = thetta - alpha * z

    fcn = theano.function(inputs=[train, labels, thetta, alpha], outputs=update)
    return fcn

def extend_train(train):
    (N,M) = train.shape
    extended = np.zeros((N, M + 1))
    extended[:,1:M+1] = train
    extended[:,0] = 1
    return extended

def main():
    train, labels = load_data('spambase.data')
    train, labels = load_data('housing.data')
    #train, labels = load_data('wine.data')

    normalize(train)
    extend = extend_train(train)
    (N,M)  = extend.shape

    gda = gradient_descent()
    n = len(labels)

    t1 = extend[0:n,:]
    l1 = labels[0:n]

    alpha = 0.001
    thetta = np.random.uniform(-5, 5, size=M)

    for idx in range(5000):
        thetta = gda(t1, l1, thetta, alpha)

    c = np.zeros(N, dtype=np.float32)
    for idx in range(len(t1)):
        t = t1[idx]
        c[idx] = np.dot(thetta, t)

    x = range(N)
    plt.plot(x, labels, linestyle='--', marker='.', color='r');
    plt.plot(x, c, linestyle='--', marker='.', color='g');
    #plt.plot(x, np.abs(labels - c), linestyle='--', marker='o', color='b');

    plt.draw()
    plt.show()

main()

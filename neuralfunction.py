"""""""""
Project 4 Function File:
This file contains the function for computing the objective value and gradient to optimize.
This function is a neural network to optimize the connection between data and results.
This function receives a one dimensional vector input and converts it to weights in the hidden layer matrices of the neural network.
The parameters of this function are:
    - Training data
    - Data classifications
    - Layer sizes (matrix whose elements are the hidden layer sizes according, i.e., [10, 5, 1] --> 10 x 5, 5 x 1)
The p input determines which function value you'd like to calculate.
    0 - function value
    1 - gradient
    2 - both
    3 - classify data
"""""""""

import numpy as np


def nueralfunction(x, para, p):
    train_data, train_class, layer_size = para.parameter
    # construct weight matrices
    n = len(layer_size)
    W = [np.array([])] * (n - 1)
    a = 0
    for k in range(n - 1):
        c = layer_size[k]
        r = layer_size[k + 1]
        b = a + r * c
        W[k] = (x[a:b]).reshape((r, c))
        a = b + 0
    # forward computation
    LL = [np.array([])] * (n - 1)
    t = W[0].dot(train_data)
    LL[0] = activation(t)
    for k in range(1, n - 1):
        t = W[k].dot(LL[k - 1])
        LL[k] = activation(t)
    # if the task is to classify then stop
    if p == 4:
        return LL[n - 2]
    # if the task is to compute loss then do so
    # along with gradient if requested
    if p == 0:
        f = 0.5 * ((LL[n - 2] - train_class) ** 2).sum()
        return f
    # gradient computation by backpropagation
    if p > 0:
        g = np.zeros((0, 1))
        t = LL[n - 2] - train_class
        for k in range(n - 2, -1, -1):
            h = t * LL[k] * (1. - LL[k])
            t = (W[k].T).dot(h)
            if k > 0:
                G = h.dot(LL[k - 1].T)
            else:
                G = h.dot(train_data.T)
            g = np.concatenate((G.reshape((-1, 1)), g))
        g = g[:,0]
        if p > 1:
            f = 0.5 * ((LL[n - 2] - train_class) ** 2).sum()
            return f, g
        return g


def activation(x):
    x = x.astype(float)
    return 1./(1.+np.exp(-x))

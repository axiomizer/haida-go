import numpy as np


def invert_kernels(kernels):
    return [[np.flip(kernels[i][j]) for i in range(len(kernels))] for j in range(len(kernels[0]))]


def rectify(z):
    return np.maximum(z, 0)


def softmax(a):
    e_to_a = np.exp(a - np.max(a))
    return e_to_a / e_to_a.sum()

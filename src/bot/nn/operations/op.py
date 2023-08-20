import numpy as np
import nnops_ext


def correlate_all_kernels(in_array, kernels):
    in_shape = np.shape(in_array)
    out_filters = len(kernels[0])
    out_array = np.zeros((out_filters, in_shape[1], in_shape[2]))
    for f in range(out_filters):
        for in_f in range(in_shape[0]):
            out_array[f] += nnops_ext.correlate(in_array[in_f], kernels[in_f][f], 1)
    return out_array


def invert_kernels(kernels):
    return [[np.flip(kernels[i][j]) for i in range(len(kernels))] for j in range(len(kernels[0]))]


def rectify(z):
    return np.maximum(z, 0)


def softmax(a):
    e_to_a = np.exp(a - np.max(a))
    return e_to_a / e_to_a.sum()

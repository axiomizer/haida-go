import numpy as np


def convolve(arr1, arr2, overhang=1):
    arr1_shape = np.shape(arr1)
    arr2_shape = np.shape(arr2)
    res_dim1 = arr1_shape[0] - arr2_shape[0] + 1 + (overhang * 2)
    res_dim2 = arr1_shape[1] - arr2_shape[1] + 1 + (overhang * 2)
    result = np.zeros((res_dim1, res_dim2))
    for i in range(res_dim1):
        for j in range(res_dim2):
            product = np.multiply(arr2[max(overhang-i, 0):min(arr1_shape[0]+overhang-i, arr2_shape[0]),
                                       max(overhang-j, 0):min(arr1_shape[1]+overhang-j, arr2_shape[1])],
                                  arr1[max(i-overhang, 0):min(arr2_shape[0]-overhang+i, arr1_shape[0]),
                                       max(j-overhang, 0):min(arr2_shape[1]-overhang+j, arr1_shape[1])])
            result[i, j] = np.sum(product)
    return result


def convolve_all_kernels(in_array, kernels):
    in_shape = np.shape(in_array)
    out_filters = len(kernels[0])
    out_array = np.zeros((out_filters, in_shape[1], in_shape[2]))
    for f in range(out_filters):
        for in_f in range(in_shape[0]):
            out_array[f] += convolve(in_array[in_f], kernels[in_f][f])
    return out_array


def invert_kernels(kernels):
    return [[np.flip(kernels[i][j]) for i in range(len(kernels))] for j in range(len(kernels[0]))]


def rectify(z):
    return np.maximum(z, 0)


def softmax(a):
    e_to_a = np.exp(a - np.max(a))
    return e_to_a / e_to_a.sum()

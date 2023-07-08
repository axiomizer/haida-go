import neuralNet as nn
import numpy as np


def test_convolve():
    arr1 = np.array([[1, 2, 3, 4],
                     [4, 4, 4, 0],
                     [-1, 0, -2, -3],
                     [1, 1, 0, 1]])
    arr2 = np.array([[1, 0, 1],
                     [2, 1, -1],
                     [1, 3, 4]])
    print(nn.convolve(arr1, arr2))


def test_sgd():
    class LastLayer:
        def sgd(self, in_activations, target_policies, target_values):
            return [-2*(target_policies[i] - in_activations[i]) for i in range(len(in_activations))]

    conv = nn.ConvolutionalBlock(2, 2)
    k11 = np.array([[-1, -2, 3], [1, 3, -1], [-1, 0, 0]])
    k12 = np.array([[-2, 2, 2], [-3, 0, -2], [-1, -3, 1]])
    k21 = np.array([[-3, 3, -1], [2, 2, 2], [-2, 0, 0]])
    k22 = np.array([[2, -2, -3], [-1, 2, 3], [-2, 3, -1]])
    conv.kernels = [[k11, k12], [k21, k22]]
    conv.biases = [0, 0]
    conv.to = LastLayer()
    in_activations = [np.array([[[-2, 1, -2], [-3, 3, 3], [-3, -2, 2]], [[0, 1, 0], [-3, 1, 3], [-1, -3, -3]]])]
    target_policies = [np.array([[[0, 2, 0], [1, 3, 2], [3, 3, 0]], [[2, 2, 1], [3, 2, 1], [1, 2, 0]]])]
    dc_da = conv.sgd(in_activations, target_policies, None)
    print(dc_da)

import neural_net as nn
import numpy as np
import unittest


class TestNeuralNet(unittest.TestCase):
    def test_convolve(self):
        arr1 = np.array([[1, 2, 3, 4],
                         [4, 4, 4, 0],
                         [-1, 0, -2, -3],
                         [1, 1, 0, 1]])
        arr2 = np.array([[1, 0, 1],
                         [2, 1, -1],
                         [1, 3, 4]])
        expected_result = np.array([[27, 33, 19, 14],
                                    [-1, 3, 0, 0],
                                    [10, 12, 10, 0],
                                    [0, 0, -2, -1]])
        self.assertTrue(np.array_equal(nn.convolve(arr1, arr2), expected_result))
        arr1 = np.array([[1, 2, 3, 4, 5],
                         [4, 4, 4, 0, 4],
                         [-1, 0, -1, -2, -3],
                         [1, 1, 2, 0, 1],
                         [-3, -2, 1, 3, 2]])
        expected_result = np.array([[27, 33, 19, 25, 25],
                                    [-1, 7, 7, -7, -3],
                                    [10, 19, 12, 13, -4],
                                    [-17, -6, 15, 17, 8],
                                    [0, -6, -5, 6, 8]])
        self.assertTrue(np.array_equal(nn.convolve(arr1, arr2), expected_result))
        # TODO: test this for overhangs other than 1 and for arr2 with even h/w and for arrays with differing h/w

    def test_sgd(self):
        class LastLayer:
            def sgd(self, in_activations, target_policies, target_values):
                return [-2*(target_policies[i] - in_activations[i]) for i in range(len(in_activations))]

        conv = nn.ConvolutionalBlock(2, 2)
        k11 = np.array([[-1., -2., 3.], [1., 3., -1.], [-1., 0., 0.]])
        k12 = np.array([[-2., 2., 2.], [-3., 0., -2.], [-1., -3., 1.]])
        k21 = np.array([[-3., 3., -1.], [2., 2., 2.], [-2., 0., 0.]])
        k22 = np.array([[2., -2., -3.], [-1., 2., 3.], [-2., 3., -1.]])
        conv.kernels = [[k11, k12], [k21, k22]]
        conv.biases = [2, -1]
        conv.to = LastLayer()
        in_activations = [np.array([[[-2, 1, -2], [-3, 3, 3], [-3, -2, 2]], [[0, 1, 0], [-3, 1, 3], [-1, -3, -3]]])]
        target_policies = [np.array([[[0, 2, 0], [1, 3, 2], [3, 3, 0]], [[2, 2, 1], [3, 2, 1], [1, 2, 0]]])]
        dc_da = conv.sgd(in_activations, target_policies, None)
        expected_result = np.array([[[-140., 88., -68.],
                                     [-166., 20., 96.],
                                     [-54., -182., 42.]],
                                    [[80., -112., 158.],
                                     [-122., 292., 238.],
                                     [-108., 14., -42.]]])
        self.assertTrue(np.allclose(dc_da, expected_result))
        # TODO: check weights and biases

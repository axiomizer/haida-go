import neural_net as nn
import numpy as np
import unittest


class TestNeuralNet(unittest.TestCase):
    def test_convolve_even(self):
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

    def test_convolve_odd(self):
        arr1 = np.array([[1, 2, 3, 4, 5],
                         [4, 4, 4, 0, 4],
                         [-1, 0, -1, -2, -3],
                         [1, 1, 2, 0, 1],
                         [-3, -2, 1, 3, 2]])
        arr2 = np.array([[1, 0, 1],
                         [2, 1, -1],
                         [1, 3, 4]])
        expected_result = np.array([[27, 33, 19, 25, 25],
                                    [-1, 7, 7, -7, -3],
                                    [10, 19, 12, 13, -4],
                                    [-17, -6, 15, 17, 8],
                                    [0, -6, -5, 6, 8]])
        self.assertTrue(np.array_equal(nn.convolve(arr1, arr2), expected_result))

    def test_convolve_rectangle(self):
        arr1 = np.array([[3, 1, 0, 2, 0, 1, 1],
                         [1, 2, 2, 0, 1, 2, 3],
                         [2, 3, 0, 2, 2, 3, 0],
                         [3, 0, 3, 1, 1, 0, 3],
                         [1, 2, 0, 1, 3, 3, 0],
                         [0, 1, 0, 2, 0, 1, 0]])
        arr2 = np.array([[0, 1, 2, 1, -2, -3, 2],
                         [1, 1, 2, -2, -3, -1, -3],
                         [3, -1, -1, 0, -2, -3, 3],
                         [2, 2, 3, -3, 2, 1, -1],
                         [0, 1, -3, 1, 3, 0, -2],
                         [-2, 2, -2, 3, 1, 0, -1]])
        expected_result = np.array([[17, 4, 11],
                                    [-18, 16, 6],
                                    [2, 24, -28]])
        self.assertTrue(np.array_equal(nn.convolve(arr1, arr2), expected_result))

    def test_convolve_overhang(self):
        arr1 = np.array([[0, 1, 3],
                         [1, 1, 1],
                         [2, 0, 3]])
        arr2 = np.array([[2, 3, -3],
                         [-1, 1, 1],
                         [-2, -3, 2]])
        expected_result = np.array([[0, 2, 3, -11, -6],
                                    [2, 0, 1, -3, -5],
                                    [5, -7, -3, 2, -1],
                                    [-1, 2, 3, 8, -1],
                                    [-6, 6, -5, 9, 6]])
        self.assertTrue(np.array_equal(nn.convolve(arr1, arr2, overhang=2), expected_result))

    def test_conv_block_sgd(self):
        class LastLayer:
            @staticmethod
            def sgd(a, pi, _):
                return [-2*(pi[i] - a[i]) for i in range(len(a))]

        conv = nn.ConvolutionalBlock(2, 2)
        k11 = np.array([[-1., -2., 3.], [1., 3., -1.], [-1., 0., 0.]])
        k12 = np.array([[-2., 2., 2.], [-3., 0., -2.], [-1., -3., 1.]])
        k21 = np.array([[-3., 3., -1.], [2., 2., 2.], [-2., 0., 0.]])
        k22 = np.array([[2., -2., -3.], [-1., 2., 3.], [-2., 3., -1.]])
        conv.kernels = [[np.copy(k11), np.copy(k12)], [np.copy(k21), np.copy(k22)]]
        conv.biases = [2, -1]
        conv.to = LastLayer()
        in_activations = [np.array([[[-2, 1, -2], [-3, 3, 3], [-3, -2, 2]], [[0, 1, 0], [-3, 1, 3], [-1, -3, -3]]])]
        target_policies = [np.array([[[0, 2, 0], [1, 3, 2], [3, 3, 0]], [[2, 2, 1], [3, 2, 1], [1, 2, 0]]])]
        dc_da = conv.sgd(in_activations, target_policies, None)

        # check dc_da
        expected_result = np.array([[[-140., 88., -68.],
                                     [-166., 20., 96.],
                                     [-54., -182., 42.]],
                                    [[80., -112., 158.],
                                     [-122., 292., 238.],
                                     [-108., 14., -42.]]])
        self.assertTrue(np.allclose(dc_da, expected_result))
        # check kernels
        expected_dc_dk11 = np.array([[32, -100, -24],
                                     [76, 232, -20],
                                     [-232, 172, 108]])
        expected_dc_dk12 = np.array([[-84, 42, -84],
                                     [-174, 150, 78],
                                     [-198, -12, 156]])
        expected_dc_dk21 = np.array([[56, 12, 0],
                                     [20, 208, 36],
                                     [-264, -176, 48]])
        expected_dc_dk22 = np.array([[0, 42, 0],
                                     [-126, 66, 126],
                                     [-114, -102, -54]])
        self.assertTrue(len(conv.kernels) == 2 and len(conv.kernels[0]) == 2 and len(conv.kernels[1]) == 2)
        self.assertTrue(np.array_equal(conv.kernels[0][0], k11 - nn.LEARNING_RATE * expected_dc_dk11))
        self.assertTrue(np.array_equal(conv.kernels[0][1], k12 - nn.LEARNING_RATE * expected_dc_dk12))
        self.assertTrue(np.array_equal(conv.kernels[1][0], k21 - nn.LEARNING_RATE * expected_dc_dk21))
        self.assertTrue(np.array_equal(conv.kernels[1][1], k22 - nn.LEARNING_RATE * expected_dc_dk22))
        # check biases
        self.assertTrue(conv.biases[0] == 2 - nn.LEARNING_RATE * 96)
        self.assertTrue(conv.biases[1] == -1 - nn.LEARNING_RATE * 66)

    def test_res_block_sgd(self):
        class LastLayer:
            @staticmethod
            def sgd(a, pi, _):
                return [-2*(pi[i] - a[i]) for i in range(len(a))]

        res = nn.ResidualBlock(2)
        res.to = [LastLayer()]
        in_activations = [np.array([[[-2, 1, -2], [-3, 3, 3], [-3, -2, 2]], [[0, 1, 0], [-3, 1, 3], [-1, -3, -3]]])]
        target_policies = [np.array([[[0, 2, 0], [1, 3, 2], [3, 3, 0]], [[2, 2, 1], [3, 2, 1], [1, 2, 0]]])]
        res.sgd(in_activations, target_policies, None)
        # TODO: finish implementing this

    # TODO: test_conv_block_feedforward
    # TODO: test_res_block_feedforward

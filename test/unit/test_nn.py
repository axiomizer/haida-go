import nn.neural_net as nn
import numpy as np
import unittest
import nn.hyperparams as hp
import nnops_ext


class TestNeuralNet(unittest.TestCase):
    def test_correlate_even(self):
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
        self.assertTrue(np.array_equal(nnops_ext.correlate(arr1, arr2, 1), expected_result))

    def test_correlate_odd(self):
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
        self.assertTrue(np.array_equal(nnops_ext.correlate(arr1, arr2, 1), expected_result))

    def test_correlate_rectangle(self):
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
        self.assertTrue(np.array_equal(nnops_ext.correlate(arr1, arr2, 1), expected_result))

    def test_correlate_overhang(self):
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
        self.assertTrue(np.array_equal(nnops_ext.correlate(arr1, arr2, 2), expected_result))

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
        self.assertTrue(np.array_equal(conv.kernels[0][0], k11 - hp.LEARNING_RATE * expected_dc_dk11))
        self.assertTrue(np.array_equal(conv.kernels[0][1], k12 - hp.LEARNING_RATE * expected_dc_dk12))
        self.assertTrue(np.array_equal(conv.kernels[1][0], k21 - hp.LEARNING_RATE * expected_dc_dk21))
        self.assertTrue(np.array_equal(conv.kernels[1][1], k22 - hp.LEARNING_RATE * expected_dc_dk22))
        # check biases
        self.assertTrue(conv.biases[0] == 2 - hp.LEARNING_RATE * 96)
        self.assertTrue(conv.biases[1] == -1 - hp.LEARNING_RATE * 66)

    def test_res_block_sgd(self):
        class LastLayer:
            @staticmethod
            def sgd(a, pi, _):
                return [-2*(pi[i] - a[i]) for i in range(len(a))]

        res = nn.ResidualBlock(2)
        k1_11 = np.array([[-2., -2., -3.], [-2., 3., -1.], [-3., 0., -3.]])
        k1_12 = np.array([[1., 2., -2.], [2., -1., 0.], [1., 2., -1.]])
        k1_21 = np.array([[-1., 3., 2.], [-3., -3., -1.], [3., 1., -1.]])
        k1_22 = np.array([[1., 2., -2.], [2., -1., 0.], [-3., -1., 0.]])
        res.kernels1 = [[np.copy(k1_11), np.copy(k1_12)], [np.copy(k1_21), np.copy(k1_22)]]
        res.biases1 = [1, -2]
        k2_11 = np.array([[1., 1., 2.], [-2., 3., -3.], [-2., -3., -2.]])
        k2_12 = np.array([[-3., 2., 2.], [2., -1., -2.], [0., -1., -3.]])
        k2_21 = np.array([[-2., -1., -1.], [2., -2., -2.], [1., 2., 1.]])
        k2_22 = np.array([[1., 0., -1.], [-3., 2., -2.], [-3., 2., -3.]])
        res.kernels2 = [[np.copy(k2_11), np.copy(k2_12)], [np.copy(k2_21), np.copy(k2_22)]]
        res.biases2 = [-1, 1]
        res.to = [LastLayer()]
        in_activations = [np.array([[[1, 0, 2], [0, 0, 1], [3, 1, 3]], [[0, 0, 3], [2, 1, 1], [1, -1, 1]]])]
        target_policies = [np.array([[[2, 1, 0], [0, 2, 0], [1, 3, 3]], [[0, 3, 0], [3, 2, 3], [1, 1, 0]]])]
        dc_da = res.sgd(in_activations, target_policies, None)

        # check dc_da
        expected_result = np.array([[[380, -274, 1314],
                                     [-572, -1286, -226],
                                     [648, 30, 770]],
                                    [[-242, -488, 32],
                                     [686, 1388, 238],
                                     [-886, -1162, -436]]])
        self.assertTrue(np.allclose(dc_da, expected_result))
        # check kernels
        expected_dc_dk1_11 = np.array([[0, 46, 0],
                                       [46, 1356, 232],
                                       [0, 212, 0]])
        expected_dc_dk1_12 = np.array([[0, 622, -82],
                                       [-246, 216, -246],
                                       [298, 972, 26]])
        expected_dc_dk1_21 = np.array([[46, 510, 232],
                                       [-46, 914, -232],
                                       [212, 408, 98]])
        expected_dc_dk1_22 = np.array([[-164, 812, -82],
                                       [216, 432, -56],
                                       [-298, 324, -26]])
        self.assertTrue(len(res.kernels1) == 2 and len(res.kernels1[0]) == 2 and len(res.kernels1[1]) == 2)
        self.assertTrue(np.array_equal(res.kernels1[0][0], k1_11 - hp.LEARNING_RATE * expected_dc_dk1_11))
        self.assertTrue(np.array_equal(res.kernels1[0][1], k1_12 - hp.LEARNING_RATE * expected_dc_dk1_12))
        self.assertTrue(np.array_equal(res.kernels1[1][0], k1_21 - hp.LEARNING_RATE * expected_dc_dk1_21))
        self.assertTrue(np.array_equal(res.kernels1[1][1], k1_22 - hp.LEARNING_RATE * expected_dc_dk1_22))
        expected_dc_dk2_11 = np.array([[0, 0, 0],
                                       [10, 1582, 4],
                                       [0, 0, 0]])
        expected_dc_dk2_12 = np.array([[0, 44, 0],
                                       [300, 144, 160],
                                       [0, 176, 0]])
        expected_dc_dk2_21 = np.array([[0, 740, 0],
                                       [216, 0, 384],
                                       [4, 1462, 34]])
        expected_dc_dk2_22 = np.array([[40, 0, 340],
                                       [0, 494, 0],
                                       [132, 1224, 0]])
        self.assertTrue(len(res.kernels2) == 2 and len(res.kernels2[0]) == 2 and len(res.kernels2[1]) == 2)
        self.assertTrue(np.array_equal(res.kernels2[0][0], k2_11 - hp.LEARNING_RATE * expected_dc_dk2_11))
        self.assertTrue(np.array_equal(res.kernels2[0][1], k2_12 - hp.LEARNING_RATE * expected_dc_dk2_12))
        self.assertTrue(np.array_equal(res.kernels2[1][0], k2_21 - hp.LEARNING_RATE * expected_dc_dk2_21))
        self.assertTrue(np.array_equal(res.kernels2[1][1], k2_22 - hp.LEARNING_RATE * expected_dc_dk2_22))
        # check biases
        self.assertTrue(res.biases1[0] == 1 - hp.LEARNING_RATE * 588)
        self.assertTrue(res.biases1[1] == -2 - hp.LEARNING_RATE * 242)
        self.assertTrue(res.biases2[0] == -1 - hp.LEARNING_RATE * 218)
        self.assertTrue(res.biases2[1] == 1 - hp.LEARNING_RATE * 114)

import numpy as np
import unittest
import nn_ext
import scipy


def loss_derivative(a, pi):
    return [-2*(pi[i] - a[i]) for i in range(len(a))]


class TestExtension(unittest.TestCase):
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
        self.assertTrue(np.array_equal(nn_ext.correlate(arr1, arr2, 1, False), expected_result))

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
        self.assertTrue(np.array_equal(nn_ext.correlate(arr1, arr2, 1, False), expected_result))

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
        self.assertTrue(np.array_equal(nn_ext.correlate(arr1, arr2, 1, False), expected_result))

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
        self.assertTrue(np.array_equal(nn_ext.correlate(arr1, arr2, 2, False), expected_result))

    def test_correlate_flipped(self):
        arr1 = np.random.randint(10, size=(10, 10))
        arr2 = np.random.randint(10, size=(3, 3))
        res1 = nn_ext.correlate(arr1, arr2, 1, True)
        res2 = nn_ext.correlate(arr1, np.flip(arr2), 1, False)
        self.assertTrue(np.array_equal(res1, res2))

    def test_correlate_all(self):
        examples = 10
        in_filters = 5
        out_filters = 6
        in_activations = np.random.randint(10, size=(examples, in_filters, 10, 10))
        kernels = np.random.randint(10, size=(in_filters, out_filters, 3, 3))
        out_activations = nn_ext.correlate_all(in_activations, kernels, False)

        expected_result = np.zeros((examples, out_filters, 10, 10), dtype=int)
        for ex in range(examples):
            for out_f in range(out_filters):
                for in_f in range(in_filters):
                    i = in_activations[ex, in_f]
                    k = kernels[in_f, out_f]
                    # scipy method is only equivalent to haida correlation for 3x3 kernels
                    expected_result[ex, out_f] += scipy.ndimage.correlate(i, k, mode='constant', cval=0)

        self.assertTrue(np.array_equal(out_activations, expected_result))

    def test_correlate_all_flipped(self):
        examples = 10
        in_filters = 5
        out_filters = 6
        err = np.random.randint(10, size=(examples, out_filters, 10, 10))
        kernels = np.random.randint(10, size=(in_filters, out_filters, 3, 3))
        actual_result = nn_ext.correlate_all(err, kernels, True)

        expected_result = np.zeros((examples, in_filters, 10, 10), dtype=int)
        for ex in range(examples):
            for out_f in range(out_filters):
                for in_f in range(in_filters):
                    e = err[ex, out_f]
                    k = np.flip(kernels[in_f, out_f])
                    # scipy method is only equivalent to haida correlation for 3x3 kernels
                    expected_result[ex, in_f] += scipy.ndimage.correlate(e, k, mode='constant', cval=0)

        self.assertTrue(np.array_equal(actual_result, expected_result))

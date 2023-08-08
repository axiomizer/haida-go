import numpy as np
import unittest
import nnops_ext


def loss_derivative(a, pi):
    return [-2*(pi[i] - a[i]) for i in range(len(a))]


class TestOperations(unittest.TestCase):
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

import numpy as np
from src.nn import hyperparams as hp
from src.nn.operations import op
import nnops_ext


class ConvolutionalBlock:
    to = None
    kernels = []
    biases = []

    def __init__(self, in_filters=hp.INPUT_PLANES, out_filters=hp.FILTERS):
        self.kernels = [[np.random.randn(3, 3) for _ in range(out_filters)] for _ in range(in_filters)]
        self.biases = np.random.randn(out_filters)

    def __activate(self, in_activations):
        self.__in_a = in_activations
        self.__a = []
        for in_a in in_activations:
            conv = op.correlate_all_kernels(in_a, self.kernels)
            for f in range(len(self.kernels[0])):
                conv[f] += self.biases[f]
            self.__a.append(op.rectify(conv))

    def feedforward(self, in_activations):
        self.__activate(in_activations)
        return self.__a

    def backprop(self, dc_da):
        da_dz = [self.__a[i] > 0 for i in range(len(self.__a))]
        dc_dz = [np.multiply(dc_da[i], da_dz[i]) for i in range(len(self.__a))]
        dc_da_prev = [op.correlate_all_kernels(x, op.invert_kernels(self.kernels)) for x in dc_dz]
        self.__update_params(dc_dz)
        return dc_da_prev

    def __update_params(self, dc_dz):
        # weights
        for i in range(len(self.kernels)):
            for j in range(len(self.kernels[0])):
                dc_dw = np.zeros((3, 3))
                for x in range(len(self.__in_a)):
                    dc_dw += nnops_ext.correlate(self.__in_a[x][i], dc_dz[x][j], 1)
                self.kernels[i][j] -= hp.LEARNING_RATE * dc_dw
        # biases
        for i in range(len(self.kernels[0])):
            dc_db = 0
            for x in range(len(self.__in_a)):
                dc_db += np.sum(dc_dz[x][i])
            self.biases[i] -= hp.LEARNING_RATE * dc_db

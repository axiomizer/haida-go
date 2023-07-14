import numpy as np
import hyperparams as hp
import nn_util as util


class ResidualBlock:
    to = []
    kernels1 = []
    biases1 = None
    kernels2 = []
    biases2 = None
    __a1 = []
    __a2 = []

    def __init__(self, filters=hp.FILTERS):
        self.kernels1 = [[np.random.randn(3, 3) for _ in range(filters)] for _ in range(filters)]
        self.biases1 = np.random.randn(filters)
        self.kernels2 = [[np.random.randn(3, 3) for _ in range(filters)] for _ in range(filters)]
        self.biases2 = np.random.randn(filters)

    def __activate(self, in_activations):
        self.__a1 = []
        for in_a in in_activations:
            conv = util.convolve_all_kernels(in_a, self.kernels1)
            for f in range(len(self.kernels1[0])):
                conv[f] += self.biases1[f]
            self.__a1.append(util.rectify(conv))
        self.__a2 = []
        for i in range(len(in_activations)):
            conv = util.convolve_all_kernels(self.__a1[i], self.kernels2)
            for f in range(len(self.kernels2[0])):
                conv[f] += self.biases2[f]
            self.__a2.append(util.rectify(conv + in_activations[i]))

    def feedforward(self, in_activations):
        self.__activate(in_activations)
        if len(self.to) == 1:
            return self.to[0].feedforward(self.__a2)
        else:
            result = [self.to[i].feedforward(self.__a2) for i in range(len(self.to))]
            return [list(i) for i in zip(*result)]

    def sgd(self, in_activations, target_policies, target_values):
        self.__activate(in_activations)
        dc_da2s = [self.to[i].sgd(self.__a2, target_policies, target_values) for i in range(len(self.to))]
        dc_da2 = [sum(x) for x in zip(*dc_da2s)]
        da2_dz2 = [self.__a2[i] > 0 for i in range(len(self.__a2))]
        dc_dz2 = [np.multiply(dc_da2[i], da2_dz2[i]) for i in range(len(self.__a2))]
        dc_da1 = [util.convolve_all_kernels(x, util.invert_kernels(self.kernels2)) for x in dc_dz2]
        da1_dz1 = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1 = [np.multiply(dc_da1[i], da1_dz1[i]) for i in range(len(self.__a1))]
        dc_da_prev = []
        for i in range(len(in_activations)):
            dc_da_prev.append(util.convolve_all_kernels(dc_dz1[i], util.invert_kernels(self.kernels1)) + dc_dz2[i])
        self.__update_params(self.__a1, dc_dz2, self.kernels2, self.biases2)
        self.__update_params(in_activations, dc_dz1, self.kernels1, self.biases1)
        return dc_da_prev

    @staticmethod
    def __update_params(activations, dc_dz, kernels, biases):
        # weights
        for i in range(len(kernels)):
            for j in range(len(kernels[0])):
                dc_dw = np.zeros((3, 3))
                for x in range(len(activations)):
                    dc_dw += util.convolve(activations[x][i], dc_dz[x][j])
                kernels[i][j] -= hp.LEARNING_RATE * dc_dw
        # biases
        for i in range(len(kernels[0])):
            dc_db = 0
            for x in range(len(activations)):
                dc_db += np.sum(dc_dz[x][i])
            biases[i] -= hp.LEARNING_RATE * dc_db

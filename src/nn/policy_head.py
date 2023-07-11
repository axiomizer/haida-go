import hyperparams as hp
import numpy as np
import nn_util as util


class PolicyHead:
    kernels = []
    biases1 = None
    weights = None  # BOARD_SIZE^2 * 2 -> BOARD_SIZE^2 + 1
    biases2 = []
    __a1 = []  # activations of convolutional layer
    __a2 = []  # activations of fully-connected linear layer (logit probabilities)
    __p = []  # output after softmax

    def __init__(self, in_filters=hp.FILTERS):
        self.kernels = [[np.random.randn() for _ in range(2)] for _ in range(in_filters)]
        self.biases1 = np.random.randn(2)
        self.weights = np.random.randn((hp.BOARD_SIZE ** 2) + 1, (hp.BOARD_SIZE ** 2) * 2)
        self.biases2 = [np.random.randn((hp.BOARD_SIZE ** 2) + 1)]

    def __activate(self, in_activations):
        self.__a1 = []
        for in_a in in_activations:
            in_shape = np.shape(in_a)
            conv = np.zeros((2, in_shape[1], in_shape[2]))
            for f in range(len(self.kernels[0])):
                conv[f] = sum([in_a[i] * self.kernels[i][f] for i in range(len(self.kernels))]) + self.biases1[f]
            self.__a1.append(util.rectify(conv))
        self.__a2 = []
        for i in range(len(in_activations)):
            self.__a2.append(np.matmul(self.weights, self.__a1[i].flatten()) + self.biases2[i])
        self.__p = [util.softmax(a) for a in self.__a2]

    def feedforward(self, in_activations):
        self.__activate(in_activations)
        return self.__p

    def sgd(self, in_activations, target_policies, _):
        self.__activate(in_activations)
        dc_da2 = [self.__p[i] - target_policies[i] for i in range(len(in_activations))]
        dc_da1_flat = [np.matmul(np.transpose(self.weights), dc_da2[i]) for i in range(len(in_activations))]
        dc_da1 = [np.reshape(dc_da1_flat[i], (2, hp.BOARD_SIZE, hp.BOARD_SIZE)) for i in range(len(in_activations))]
        da1_dz1 = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1 = [np.multiply(dc_da1[i], da1_dz1[i]) for i in range(len(self.__a1))]
        dc_da_prev = []
        for i in range(len(in_activations)):
            dc_da_prev_example = np.zeros((len(self.kernels), hp.BOARD_SIZE, hp.BOARD_SIZE))
            for f in range(len(self.kernels)):
                dc_da_prev_example[f] = sum([dc_dz1[i][fp] * self.kernels[f][fp] for fp in range(len(self.kernels[f]))])
            dc_da_prev.append(dc_da_prev_example)
        self.__update_layer2_params(dc_da2)
        self.__update_layer1_params(dc_dz1, in_activations)
        return dc_da_prev

    def __update_layer2_params(self, dc_da2):
        dc_dw = np.zeros(len(self.weights), len(self.weights[0]))
        for i in range(len(dc_da2)):
            dc_dw += np.outer(dc_da2[i], self.__a1[i])
        self.weights -= (hp.LEARNING_RATE / len(dc_da2)) * dc_dw
        dc_db = np.zeros(len(self.weights))
        for i in range(len(dc_da2)):
            dc_db += dc_da2[i]
        self.biases2 -= (hp.LEARNING_RATE / len(dc_da2)) * dc_db

    def __update_layer1_params(self, dc_dz1, in_activations):
        for i in range(len(self.kernels)):
            for j in range(len(self.kernels[i])):
                dc_dk = 0
                for x in range(len(dc_dz1)):
                    dc_dk += np.sum(dc_dz1[x][j] * in_activations[x][i])
                self.kernels[i][j] -= (hp.LEARNING_RATE / len(in_activations)) * dc_dk
        for i in range(len(self.kernels[0])):
            dc_db = 0
            for x in range(len(dc_dz1)):
                dc_db += np.sum(dc_dz1[x][i])
            self.biases1[i] -= (hp.LEARNING_RATE / len(in_activations)) * dc_db

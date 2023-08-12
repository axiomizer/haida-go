from math import isclose
import numpy as np
from src.nn.operations import op
from src.nn.batch_norm import BatchNorm


class PolicyHead:
    def __init__(self, in_filters, board_size, config):
        self.cfg = config
        self.board_size = board_size

        # convolutional layer; kernels indexed as [from][to]
        self.kernels = [[np.random.randn() for _ in range(2)] for _ in range(in_filters)]

        # batch norm
        self.bn = BatchNorm(2, config)

        # fully-connected linear layer: weights indexed as [to][from]
        self.weights = np.random.randn((board_size ** 2) + 1, (board_size ** 2) * 2)
        self.biases = np.random.randn((board_size ** 2) + 1)

        self.__in_a = None  # input activations
        self.__a1 = None  # output activations of convolutional layer
        self.__a2 = None  # output activations of fully-connected linear layer (logit probabilities)
        self.__p = None  # output after softmax

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z = []
        for in_a in in_activations:
            in_shape = np.shape(in_a)
            conv = np.zeros((2, in_shape[1], in_shape[2]))
            for f in range(len(self.kernels[0])):
                conv[f] = sum([in_a[i] * self.kernels[i][f] for i in range(len(self.kernels))])
            z.append(conv)
        self.__a1 = [op.rectify(z_hat) for z_hat in self.bn.feedforward(z)]
        self.__a2 = []
        for i in range(len(in_activations)):
            self.__a2.append(np.matmul(self.weights, self.__a1[i].flatten()) + self.biases)
        self.__p = [op.softmax(a) for a in self.__a2]
        return self.__p

    def backprop(self, target_policies):
        # the formula for dc_da2 is only valid if the target policy sums to 1
        for pi in target_policies:
            if not isclose(sum(pi), 1):
                raise ValueError('target policy (pi) must sum to 1. actual sum was {}'.format(sum(pi)))
        dc_da2 = [(self.__p[i] - target_policies[i]) / len(self.__in_a) for i in range(len(self.__in_a))]
        dc_da1_flat = [np.matmul(np.transpose(self.weights), dc_da2[i]) for i in range(len(self.__in_a))]
        dc_da1 = [np.reshape(dc_da1_flat[i], (2, self.board_size, self.board_size)) for i in range(len(self.__in_a))]
        da1_dz1_hat = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1_hat = [np.multiply(dc_da1[i], da1_dz1_hat[i]) for i in range(len(self.__a1))]
        dc_dz1 = self.bn.backprop(dc_dz1_hat)
        dc_da_prev = []
        for i in range(len(self.__in_a)):
            dc_da_prev_example = np.zeros((len(self.kernels), self.board_size, self.board_size))
            for f in range(len(self.kernels)):
                dc_da_prev_example[f] = sum([dc_dz1[i][fp] * self.kernels[f][fp] for fp in range(len(self.kernels[f]))])
            dc_da_prev.append(dc_da_prev_example)
        self.__update_layer2_params(dc_da2)
        self.__update_layer1_params(dc_dz1)
        return dc_da_prev

    def __update_layer2_params(self, dc_da2):
        dc_dw = np.zeros((len(self.weights), len(self.weights[0])))
        for i in range(len(dc_da2)):
            dc_dw += np.outer(dc_da2[i], self.__a1[i])
        self.cfg.theta_update_rule(self.weights, dc_dw)
        dc_db = np.zeros(len(self.weights))
        for i in range(len(dc_da2)):
            dc_db += dc_da2[i]
        self.cfg.theta_update_rule(self.biases, dc_db)

    def __update_layer1_params(self, dc_dz1):
        for i in range(len(self.kernels)):
            for j in range(len(self.kernels[i])):
                dc_dk = 0
                for x in range(len(dc_dz1)):
                    dc_dk += np.sum(dc_dz1[x][j] * self.__in_a[x][i])
                self.cfg.theta_update_rule(self.kernels[i][j], dc_dk)

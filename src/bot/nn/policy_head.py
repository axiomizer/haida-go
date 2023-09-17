from math import isclose
import numpy as np
from src.bot.nn.batch_norm import BatchNorm
from src.bot.nn.shared import AbstractNet


class PolicyHead(AbstractNet):
    def __init__(self, in_filters, board_size, config=None):
        super().__init__(config)
        self.board_size = board_size
        self.in_filters = in_filters
        self.l1_filters = 2

        # convolutional layer; kernels indexed as [from][to]
        self.kernels = np.random.randn(in_filters, self.l1_filters, 1, 1)
        self.__dc_dk_runavg = [[np.zeros((1, 1)) for _ in range(self.l1_filters)] for _ in range(in_filters)]

        # batch norm
        self.bn = BatchNorm(self.l1_filters, self.cfg)

        # fully-connected linear layer: weights indexed as [to][from]
        std = ((board_size ** 2) * self.l1_filters) ** -0.5
        self.weights = np.random.normal(scale=std, size=((board_size ** 2) + 1, (board_size ** 2) * self.l1_filters))
        self.biases = np.zeros(((board_size ** 2) + 1,))
        self.__dc_dw_runavg = np.zeros(((board_size ** 2) + 1, (board_size ** 2) * self.l1_filters))
        self.__dc_db_runavg = np.zeros((board_size ** 2) + 1)

        self.__in_a = None  # input activations
        self.__a1 = None  # output activations of convolutional layer
        self.__a2 = None  # output activations of fully-connected linear layer (logit probabilities)
        self.__p = None  # output after softmax

    @staticmethod
    def __softmax(a):
        e_to_a = np.exp(a - np.max(a))
        return e_to_a / e_to_a.sum()

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z = np.einsum('ijkl,jmno->imkl', in_activations, self.kernels)
        self.__a1 = np.maximum(self.bn.feedforward(z), 0)
        a1_flat = np.reshape(self.__a1, (len(self.__a1), np.prod(self.__a1.shape[1:])))
        self.__a2 = np.einsum('ij,kj->ki', self.weights, a1_flat) + self.biases
        self.__p = np.empty(self.__a2.shape, dtype=float)
        for i in range(len(self.__a2)):
            self.__p[i] = self.__softmax(self.__a2[i])
        return self.__p

    def loss(self, target):
        losses = [-1 * np.dot(target[i], np.log(self.__p[i])) for i in range(len(target))]
        return sum(losses) / len(target)

    # calculate the error with respect to the logit probabilities
    def error(self, target):
        # the formula for dc_da2 is only valid if the target policy sums to 1
        for pi in target:
            if not isclose(sum(pi), 1):
                raise ValueError('target policy (pi) must sum to 1. actual sum was {}'.format(sum(pi)))
        dc_da2 = [(self.__p[i] - target[i]) / len(self.__in_a) for i in range(len(self.__in_a))]
        return dc_da2

    def backprop(self, dc_da2):
        dc_da1_flat = np.einsum('kj,ik->ij', self.weights, dc_da2)
        shape = (len(dc_da1_flat), self.l1_filters, self.board_size, self.board_size)
        dc_da1 = np.reshape(dc_da1_flat, shape)
        dc_dz1_hat = np.multiply(dc_da1, self.__a1 > 0)
        dc_dz1 = self.bn.backprop(dc_dz1_hat)
        dc_da_prev = np.einsum('ijkl,mjno->imkl', dc_dz1, self.kernels)
        self.__update_layer2_params(dc_da2)
        self.__update_layer1_params(dc_dz1)
        return dc_da_prev

    def __update_layer2_params(self, dc_da2):
        dc_dw = np.zeros((len(self.weights), len(self.weights[0])))
        for i in range(len(dc_da2)):
            dc_dw += np.outer(dc_da2[i], self.__a1[i])
        self.update_theta(self.weights, dc_dw, self.__dc_dw_runavg)
        dc_db = np.zeros(len(self.weights))
        for i in range(len(dc_da2)):
            dc_db += dc_da2[i]
        self.update_theta(self.biases, dc_db, self.__dc_db_runavg)

    def __update_layer1_params(self, dc_dz1):
        for i in range(self.in_filters):
            for j in range(self.l1_filters):
                dc_dk = 0
                for x in range(len(dc_dz1)):
                    dc_dk += np.sum(dc_dz1[x][j] * self.__in_a[x][i])
                self.update_theta(self.kernels[i][j], dc_dk, self.__dc_dk_runavg[i][j])

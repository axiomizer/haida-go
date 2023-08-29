from math import isclose
import numpy as np
from src.bot.nn.operations import op
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

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z = []
        for in_a in in_activations:
            in_shape = np.shape(in_a)
            conv = np.zeros((self.l1_filters, in_shape[1], in_shape[2]))
            for f in range(self.l1_filters):
                conv[f] = sum([in_a[i] * self.kernels[i][f] for i in range(self.in_filters)])
            z.append(conv)
        self.__a1 = [op.rectify(z_hat) for z_hat in self.bn.feedforward(z)]
        self.__a2 = []
        for i in range(len(in_activations)):
            self.__a2.append(np.matmul(self.weights, self.__a1[i].flatten()) + self.biases)
        self.__p = [op.softmax(a) for a in self.__a2]
        return self.__p

    # calculate the error with respect to the logit probabilities
    def error(self, target):
        # the formula for dc_da2 is only valid if the target policy sums to 1
        for pi in target:
            if not isclose(sum(pi), 1):
                raise ValueError('target policy (pi) must sum to 1. actual sum was {}'.format(sum(pi)))
        dc_da2 = [(self.__p[i] - target[i]) / len(self.__in_a) for i in range(len(self.__in_a))]
        return dc_da2

    def backprop(self, dc_da2):
        dc_da1_flat = [np.matmul(np.transpose(self.weights), dc_da2[i]) for i in range(len(self.__in_a))]
        shape = (self.l1_filters, self.board_size, self.board_size)
        dc_da1 = [np.reshape(dc_da1_flat[i], shape) for i in range(len(self.__in_a))]
        da1_dz1_hat = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1_hat = [np.multiply(dc_da1[i], da1_dz1_hat[i]) for i in range(len(self.__a1))]
        dc_dz1 = self.bn.backprop(dc_dz1_hat)
        dc_da_prev = []
        for i in range(len(self.__in_a)):
            dc_da_prev_example = np.zeros((self.in_filters, self.board_size, self.board_size))
            for f in range(self.in_filters):
                dc_da_prev_example[f] = sum([dc_dz1[i][fp] * self.kernels[f][fp] for fp in range(self.l1_filters)])
            dc_da_prev.append(dc_da_prev_example)
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

import numpy as np
from src.nn.operations import op
from src.nn.batch_norm import BatchNorm
from src.nn.shared import AbstractNet


class ValueHead(AbstractNet):
    def __init__(self, in_filters, board_size, config=None):
        super().__init__(config)
        self.in_filters = in_filters
        self.board_size = board_size

        self.l1_kernels = np.random.randn(in_filters)
        self.__l1_dc_dk_runavg = np.zeros(in_filters)
        self.bn = BatchNorm(1, self.cfg)
        self.l2_weights = np.random.randn(256, board_size ** 2)  # indexed as [to][from]
        self.__l2_dc_dw_runavg = np.zeros((256, board_size ** 2))
        self.l2_biases = np.random.randn(256)
        self.__l2_dc_db_runavg = np.zeros(256)
        self.l3_weights = np.random.randn(256)
        self.__l3_dc_dw_runavg = np.zeros(256)
        self.l3_bias = np.random.randn(1)
        self.__l3_dc_db_runavg = np.zeros(1)

        self.__in_a = None
        self.__a1 = None
        self.__a2 = None
        self.__v = None

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z = []
        for in_a in in_activations:
            conv = sum([in_a[i] * self.l1_kernels[i] for i in range(self.in_filters)])
            z.append(np.expand_dims(conv, 0))
        self.__a1 = [op.rectify(np.squeeze(z_hat)) for z_hat in self.bn.feedforward(z)]
        self.__a2 = []
        for i in range(len(in_activations)):
            z = np.matmul(self.l2_weights, self.__a1[i].flatten()) + self.l2_biases
            self.__a2.append(op.rectify(z))
        self.__v = []
        for i in range(len(in_activations)):
            z = np.dot(self.l3_weights, self.__a2[i]) + self.l3_bias[0]
            self.__v.append(np.tanh(z))
        return self.__v

    def error(self, target):
        return [-2 * (target[i] - self.__v[i]) / len(self.__in_a) for i in range(len(self.__in_a))]

    def backprop(self, err):
        dc_dz3 = [err[i] * (1 - self.__v[i] ** 2) for i in range(len(self.__in_a))]
        dc_da2 = [self.l3_weights * dc_dz3[i] for i in range(len(self.__in_a))]
        da2_dz2 = [self.__a2[i] > 0 for i in range(len(self.__a2))]
        dc_dz2 = [np.multiply(dc_da2[i], da2_dz2[i]) for i in range(len(self.__a2))]
        dc_da1_flat = [np.matmul(np.transpose(self.l2_weights), dc_dz2[i]) for i in range(len(self.__in_a))]
        dc_da1 = [np.reshape(dc_da1_flat[i], (self.board_size, self.board_size)) for i in range(len(self.__in_a))]
        da1_dz1_hat = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1_hat = [np.expand_dims(np.multiply(dc_da1[i], da1_dz1_hat[i]), 0) for i in range(len(self.__a1))]
        dc_dz1 = [np.squeeze(dc_dz1_ex) for dc_dz1_ex in self.bn.backprop(dc_dz1_hat)]
        dc_da_prev = []
        for i in range(len(self.__in_a)):
            dc_da_prev_example = np.zeros((self.in_filters, self.board_size, self.board_size))
            for f in range(self.in_filters):
                dc_da_prev_example[f] = dc_dz1[i] * self.l1_kernels[f]
            dc_da_prev.append(dc_da_prev_example)
        self.__update_layer3_params(dc_dz3)
        self.__update_layer2_params(dc_dz2)
        self.__update_layer1_params(dc_dz1)
        return dc_da_prev

    def __update_layer3_params(self, dc_dz3):
        dc_dw = np.zeros(len(self.l3_weights))
        for i in range(len(dc_dz3)):
            dc_dw += dc_dz3[i] * self.__a2[i]
        self.update_theta(self.l3_weights, dc_dw, self.__l3_dc_dw_runavg)
        self.update_theta(self.l3_bias, sum(dc_dz3), self.__l3_dc_db_runavg)

    def __update_layer2_params(self, dc_dz2):
        dc_dw = np.zeros((len(self.l2_weights), len(self.l2_weights[0])))
        for i in range(len(dc_dz2)):
            dc_dw += np.outer(dc_dz2[i], self.__a1[i])
        self.update_theta(self.l2_weights, dc_dw, self.__l2_dc_dw_runavg)
        self.update_theta(self.l2_biases, sum(dc_dz2), self.__l2_dc_db_runavg)

    def __update_layer1_params(self, dc_dz1):
        dc_dk = np.zeros(self.in_filters)
        for f in range(self.in_filters):
            for x in range(len(dc_dz1)):
                dc_dk[f] += np.sum(dc_dz1[x] * self.__in_a[x][f])
        self.update_theta(self.l1_kernels, dc_dk, self.__l1_dc_dk_runavg)

    def checkpoint(self):
        pass

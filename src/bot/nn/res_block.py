import numpy as np
from src.bot.nn.batch_norm import BatchNorm
import nn_ext
from src.bot.nn.shared import AbstractNet


class ResidualBlock(AbstractNet):
    def __init__(self, filters, config=None):
        super().__init__(config)
        self.filters = filters

        self.kernels1 = np.random.randn(filters, filters, 3, 3)
        self.__dc_dk1_runavg = [[np.zeros((3, 3)) for _ in range(filters)] for _ in range(filters)]
        self.bn1 = BatchNorm(filters, self.cfg)
        self.kernels2 = np.random.randn(filters, filters, 3, 3)
        self.__dc_dk2_runavg = [[np.zeros((3, 3)) for _ in range(filters)] for _ in range(filters)]
        self.bn2 = BatchNorm(filters, self.cfg)

        self.__in_a = None
        self.__a1 = None
        self.__a2 = None

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z1 = nn_ext.correlate_all(in_activations, self.kernels1, False)
        self.__a1 = [np.maximum(z1_hat, 0) for z1_hat in self.bn1.feedforward(z1)]
        z2 = nn_ext.correlate_all(self.__a1, self.kernels2, False)
        z2_hat = self.bn2.feedforward(z2)
        self.__a2 = []
        for i in range(len(in_activations)):
            self.__a2.append(np.maximum(z2_hat[i] + in_activations[i], 0))
        return self.__a2

    def error(self, target):
        raise NotImplementedError()

    def backprop(self, dc_da2):
        da2_dz2_hat = [self.__a2[i] > 0 for i in range(len(self.__a2))]
        dc_dz2_hat = [np.multiply(dc_da2[i], da2_dz2_hat[i]) for i in range(len(self.__a2))]
        dc_dz2 = self.bn2.backprop(dc_dz2_hat)
        dc_da1 = nn_ext.correlate_all(dc_dz2, self.kernels2, True)
        da1_dz1_hat = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1_hat = [np.multiply(dc_da1[i], da1_dz1_hat[i]) for i in range(len(self.__a1))]
        dc_dz1 = self.bn1.backprop(dc_dz1_hat)
        dc_da_prev = nn_ext.correlate_all(dc_dz1, self.kernels1, True) + dc_dz2_hat
        self.__update_params(dc_dz1, dc_dz2)
        return dc_da_prev

    def __update_params(self, dc_dz1, dc_dz2):
        for i in range(self.filters):
            for j in range(self.filters):
                dc_dk2 = np.zeros((3, 3))
                dc_dk1 = np.zeros((3, 3))
                for x in range(len(self.__in_a)):
                    dc_dk2 += nn_ext.correlate(self.__a1[x][i], dc_dz2[x][j], 1, False)
                    dc_dk1 += nn_ext.correlate(self.__in_a[x][i], dc_dz1[x][j], 1, False)
                self.update_theta(self.kernels2[i][j], dc_dk2, self.__dc_dk2_runavg[i][j])
                self.update_theta(self.kernels1[i][j], dc_dk1, self.__dc_dk1_runavg[i][j])

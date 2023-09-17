import numpy as np
from src.bot.nn.ext import op
from src.bot.nn.batch_norm import BatchNorm
import nn_ext
from src.bot.nn.shared import AbstractNet


class ConvolutionalBlock(AbstractNet):
    def __init__(self, in_filters, out_filters, config=None):
        super().__init__(config)

        self.kernels = np.random.randn(in_filters, out_filters, 3, 3)
        self.__dc_dk_runavg = [[np.zeros((3, 3)) for _ in range(out_filters)] for _ in range(in_filters)]
        self.bn = BatchNorm(out_filters, self.cfg)

        self.__in_a = None
        self.__a = None

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z = nn_ext.correlate_all(in_activations, self.kernels)
        self.__a = [op.rectify(z_hat) for z_hat in self.bn.feedforward(z)]
        return self.__a

    def error(self, target):
        raise NotImplementedError()

    def backprop(self, dc_da):
        da_dz_hat = [self.__a[i] > 0 for i in range(len(self.__a))]
        dc_dz_hat = [np.multiply(dc_da[i], da_dz_hat[i]) for i in range(len(self.__a))]
        dc_dz = self.bn.backprop(dc_dz_hat)
        dc_da_prev = nn_ext.correlate_all(dc_dz, op.invert_kernels(self.kernels))
        self.__update_params(dc_dz)
        return dc_da_prev

    def __update_params(self, dc_dz):
        for i in range(len(self.kernels)):
            for j in range(len(self.kernels[0])):
                dc_dk = np.zeros((3, 3))
                for x in range(len(self.__in_a)):
                    dc_dk += nn_ext.correlate(self.__in_a[x][i], dc_dz[x][j], 1)
                self.update_theta(self.kernels[i][j], dc_dk, self.__dc_dk_runavg[i][j])

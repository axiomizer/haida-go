import numpy as np
from src.nn.operations import op
from src.nn.batch_norm import BatchNorm
import nnops_ext
from src.nn.shared import AbstractNet


class ResidualBlock(AbstractNet):
    def __init__(self, filters, config=None):
        super().__init__(config)

        self.kernels1 = [[np.random.randn(3, 3) for _ in range(filters)] for _ in range(filters)]
        self.bn1 = BatchNorm(filters, self.cfg)
        self.kernels2 = [[np.random.randn(3, 3) for _ in range(filters)] for _ in range(filters)]
        self.bn2 = BatchNorm(filters, self.cfg)

        self.__in_a = None
        self.__a1 = None
        self.__a2 = None

    def feedforward(self, in_activations):
        self.__in_a = in_activations
        z1 = [op.correlate_all_kernels(in_a, self.kernels1) for in_a in in_activations]
        self.__a1 = [op.rectify(z1_hat) for z1_hat in self.bn1.feedforward(z1)]
        z2 = [op.correlate_all_kernels(a1, self.kernels2) for a1 in self.__a1]
        z2_hat = self.bn2.feedforward(z2)
        self.__a2 = []
        for i in range(len(in_activations)):
            self.__a2.append(op.rectify(z2_hat[i] + in_activations[i]))
        return self.__a2

    def error(self, target):
        raise NotImplementedError()

    def backprop(self, dc_da2):
        da2_dz2_hat = [self.__a2[i] > 0 for i in range(len(self.__a2))]
        dc_dz2_hat = [np.multiply(dc_da2[i], da2_dz2_hat[i]) for i in range(len(self.__a2))]
        dc_dz2 = self.bn2.backprop(dc_dz2_hat)
        dc_da1 = [op.correlate_all_kernels(x, op.invert_kernels(self.kernels2)) for x in dc_dz2]
        da1_dz1_hat = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1_hat = [np.multiply(dc_da1[i], da1_dz1_hat[i]) for i in range(len(self.__a1))]
        dc_dz1 = self.bn1.backprop(dc_dz1_hat)
        dc_da_prev = []
        for i in range(len(self.__in_a)):
            dc_da_prev.append(op.correlate_all_kernels(dc_dz1[i], op.invert_kernels(self.kernels1)) + dc_dz2_hat[i])
        self.__update_params(self.__a1, dc_dz2, self.kernels2)
        self.__update_params(self.__in_a, dc_dz1, self.kernels1)
        return dc_da_prev

    def __update_params(self, activations, dc_dz, kernels):
        for i in range(len(kernels)):
            for j in range(len(kernels[0])):
                dc_dw = np.zeros((3, 3))
                for x in range(len(activations)):
                    dc_dw += nnops_ext.correlate(activations[x][i], dc_dz[x][j], 1)
                self.update_theta(kernels[i][j], dc_dw)

    def checkpoint(self):
        pass

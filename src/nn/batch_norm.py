import numpy as np
from math import sqrt
from src.nn.shared import AbstractNet


class BatchNorm(AbstractNet):
    def __init__(self, filters, config=None):
        super().__init__(config)
        self.filters = filters

        # trainable parameters
        self.gamma = np.ones(filters)
        self.beta = np.zeros(filters)

        # values from last call to feedforward
        self.__x = None
        self.__x_hat = None
        self.__mean = None
        self.__variance = None
        self.__num_samples = 0

    def feedforward(self, x):
        batch_size = len(x)
        self.__x = x
        self.__num_samples = batch_size * np.prod(x[0][0].shape)
        self.__x_hat = [np.zeros(x[0].shape) for _ in range(batch_size)]
        y = [np.zeros(x[0].shape) for _ in range(batch_size)]
        self.__mean = np.zeros(self.filters)
        self.__variance = np.zeros(self.filters)
        for f in range(self.filters):
            for b in range(batch_size):
                self.__mean[f] += np.sum(x[b][f])
            self.__mean[f] /= self.__num_samples
            for b in range(batch_size):
                self.__variance[f] += np.sum((x[b][f] - self.__mean[f]) ** 2)
            self.__variance[f] /= self.__num_samples
            for b in range(batch_size):
                self.__x_hat[b][f] = (x[b][f] - self.__mean[f]) / sqrt(self.__variance[f] + self.cfg.batch_norm_epsilon)
                y[b][f] = self.__x_hat[b][f] * self.gamma[f] + self.beta[f]
        return y

    def error(self, target):
        raise NotImplementedError()

    def backprop(self, err):
        dc_dx_hat = [np.zeros(err[0].shape) for _ in range(len(err))]
        dc_dx = [np.zeros(err[0].shape) for _ in range(len(err))]
        dc_dgamma = np.zeros(self.filters)
        dc_dbeta = np.zeros(self.filters)
        for f in range(self.filters):
            for b in range(len(err)):
                dc_dx_hat[b][f] = err[b][f] * self.gamma[f]
                dc_dgamma[f] += np.sum(err[b][f] * self.__x_hat[b][f])
                dc_dbeta[f] += np.sum(err[b][f])
            dc_dvariance = 0
            factor = (-1 / 2) * ((self.__variance[f] + self.cfg.batch_norm_epsilon) ** (-3 / 2))
            for b in range(len(err)):
                dx_hat_dvariance = (self.__x[b][f] - self.__mean[f]) * factor
                dc_dvariance += np.sum(np.multiply(dc_dx_hat[b][f], dx_hat_dvariance))
            dc_dmean = 0
            dx_hat_dmean = -1 / sqrt(self.__variance[f] + self.cfg.batch_norm_epsilon)
            temp = 0
            for b in range(len(err)):
                temp += np.sum(self.__x[b][f])
                dc_dmean += np.sum(dc_dx_hat[b][f]) * dx_hat_dmean
            dvariance_dmean = (-2 / self.__num_samples) * (temp - self.__mean[f] * self.__num_samples)
            dc_dmean += dc_dvariance * dvariance_dmean
            for b in range(len(err)):
                path1 = dc_dx_hat[b][f] / sqrt(self.__variance[f] + self.cfg.batch_norm_epsilon)
                path2 = dc_dvariance * (2 / self.__num_samples) * (self.__x[b][f] - self.__mean[f])
                path3 = dc_dmean / self.__num_samples
                dc_dx[b][f] = path1 + path2 + path3
        self.update_theta(self.gamma, dc_dgamma)
        self.update_theta(self.beta, dc_dbeta)
        return dc_dx

    def checkpoint(self):
        pass

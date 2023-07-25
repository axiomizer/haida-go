import nn.hyperparams as hp
import numpy as np
import nn.operations as op


class ValueHead:
    l1_kernels = []
    l1_bias = None
    l2_weights = None  # BOARD_SIZE^2 -> 256 (indexed as [to][from])
    l2_biases = None
    l3_weights = None
    l3_bias = None

    def __init__(self, in_filters=hp.FILTERS, board_size=hp.BOARD_SIZE):
        self.board_size = board_size
        self.l1_kernels = np.random.randn(in_filters)
        self.l1_bias = np.random.randn()
        self.l2_weights = np.random.randn(256, board_size ** 2)
        self.l2_biases = np.random.randn(256)
        self.l3_weights = np.random.randn(256)
        self.l3_bias = np.random.randn()

    def __activate(self, in_activations):
        self.__a1 = []
        for in_a in in_activations:
            conv = sum([in_a[i] * self.l1_kernels[i] for i in range(len(self.l1_kernels))]) + self.l1_bias
            self.__a1.append(op.rectify(conv))
        self.__a2 = []
        for i in range(len(in_activations)):
            z = np.matmul(self.l2_weights, self.__a1[i].flatten()) + self.l2_biases
            self.__a2.append(op.rectify(z))
        self.__v = []
        for i in range(len(in_activations)):
            z = np.dot(self.l3_weights, self.__a2[i]) + self.l3_bias
            self.__v.append(np.tanh(z))

    def feedforward(self, in_activations):
        self.__activate(in_activations)
        return self.__v

    def sgd(self, in_activations, _, target_values):
        self.__activate(in_activations)
        dc_dv = [-2 * (target_values[i] - self.__v[i]) / len(in_activations) for i in range(len(in_activations))]
        dc_dz3 = [dc_dv[i] * (1 - self.__v[i] ** 2) for i in range(len(in_activations))]
        dc_da2 = [self.l3_weights * dc_dz3[i] for i in range(len(in_activations))]
        da2_dz2 = [self.__a2[i] > 0 for i in range(len(self.__a2))]
        dc_dz2 = [np.multiply(dc_da2[i], da2_dz2[i]) for i in range(len(self.__a2))]
        dc_da1_flat = [np.matmul(np.transpose(self.l2_weights), dc_dz2[i]) for i in range(len(in_activations))]
        dc_da1 = [np.reshape(dc_da1_flat[i], (self.board_size, self.board_size)) for i in range(len(in_activations))]
        da1_dz1 = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1 = [np.multiply(dc_da1[i], da1_dz1[i]) for i in range(len(self.__a1))]
        dc_da_prev = []
        for i in range(len(in_activations)):
            dc_da_prev_example = np.zeros((len(self.l1_kernels), self.board_size, self.board_size))
            for f in range(len(self.l1_kernels)):
                dc_da_prev_example[f] = dc_dz1[i] * self.l1_kernels[f]
            dc_da_prev.append(dc_da_prev_example)
        self.__update_layer3_params(dc_dz3)
        self.__update_layer2_params(dc_dz2)
        self.__update_layer1_params(dc_dz1, in_activations)
        return dc_da_prev

    def __update_layer3_params(self, dc_dz3):
        dc_dw = np.zeros(len(self.l3_weights))
        for i in range(len(dc_dz3)):
            dc_dw += dc_dz3[i] * self.__a2[i]
        self.l3_weights -= hp.LEARNING_RATE * dc_dw
        self.l3_bias -= hp.LEARNING_RATE * sum(dc_dz3)

    def __update_layer2_params(self, dc_dz2):
        dc_dw = np.zeros((len(self.l2_weights), len(self.l2_weights[0])))
        for i in range(len(dc_dz2)):
            dc_dw += np.outer(dc_dz2[i], self.__a1[i])
        self.l2_weights -= hp.LEARNING_RATE * dc_dw
        self.l2_biases -= hp.LEARNING_RATE * sum(dc_dz2)

    def __update_layer1_params(self, dc_dz1, in_activations):
        for f in range(len(self.l1_kernels)):
            dc_dk = 0
            for x in range(len(dc_dz1)):
                dc_dk += np.sum(dc_dz1[x] * in_activations[x][f])
            self.l1_kernels[f] -= hp.LEARNING_RATE * dc_dk
        self.l1_bias -= hp.LEARNING_RATE * np.sum(dc_dz1)

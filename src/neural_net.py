import numpy as np

RESIDUAL_BLOCKS = 3  # 19
INPUT_PLANES = 7  # 17
FILTERS = 16  # 256
BOARD_SIZE = 9  # 19
LEARNING_RATE = 0.001  # alpha go zero uses annealing (extended data table 3)


def convolve(arr1, arr2, overhang=1):
    arr1_shape = np.shape(arr1)
    arr2_shape = np.shape(arr2)
    res_dim1 = arr1_shape[0] - arr2_shape[0] + 1 + (overhang * 2)
    res_dim2 = arr1_shape[1] - arr2_shape[1] + 1 + (overhang * 2)
    result = np.zeros((res_dim1, res_dim2))
    for i in range(res_dim1):
        for j in range(res_dim2):
            product = np.multiply(arr2[max(overhang-i, 0):min(arr1_shape[0]+overhang-i, arr2_shape[0]),
                                       max(overhang-j, 0):min(arr1_shape[1]+overhang-j, arr2_shape[1])],
                                  arr1[max(i-overhang, 0):min(arr2_shape[0]-overhang+i, arr1_shape[0]),
                                       max(j-overhang, 0):min(arr2_shape[1]-overhang+j, arr1_shape[1])])
            result[i, j] = np.sum(product)
    return result


def convolve_all_kernels(in_array, kernels):
    in_shape = np.shape(in_array)
    out_filters = len(kernels[0])
    out_array = np.zeros((out_filters, in_shape[1], in_shape[2]))
    for f in range(out_filters):
        for in_f in range(in_shape[0]):
            out_array[f] += convolve(in_array[in_f], kernels[in_f][f])
    return out_array


def invert_kernels(kernels):
    return [[np.flip(kernels[i][j]) for i in range(len(kernels))] for j in range(len(kernels[0]))]


def rectify(z):
    return np.maximum(z, 0)


class ConvolutionalBlock:
    to = None
    kernels = []
    biases = []

    def __init__(self, in_filters=INPUT_PLANES, out_filters=FILTERS):
        self.kernels = [[np.random.randn(3, 3) for _ in range(out_filters)] for _ in range(in_filters)]
        self.biases = np.random.randn(out_filters)

    def __activate(self, in_activations):
        a = []
        for in_a in in_activations:
            conv = convolve_all_kernels(in_a, self.kernels)
            for f in range(len(self.kernels[0])):
                conv[f] += self.biases[f]
            a.append(rectify(conv))
        return a

    def feedforward(self, in_activations):
        a = self.__activate(in_activations)
        return self.to.feedforward(a)

    def sgd(self, in_activations, target_policies, target_values):
        a = self.__activate(in_activations)
        dc_da = self.to.sgd(a, target_policies, target_values)
        da_dz = [a[i] > 0 for i in range(len(a))]
        dc_dz = [np.multiply(dc_da[i], da_dz[i]) for i in range(len(a))]
        dc_da_prev = [convolve_all_kernels(x, invert_kernels(self.kernels)) for x in dc_dz]
        self.__update_params(in_activations, dc_dz)
        return dc_da_prev

    def __update_params(self, in_activations, dc_dz):
        # weights
        for i in range(len(self.kernels)):
            for j in range(len(self.kernels[0])):
                dc_dw = np.zeros((3, 3))
                for x in range(len(in_activations)):
                    dc_dw += convolve(in_activations[x][i], dc_dz[x][j])
                self.kernels[i][j] -= (LEARNING_RATE / len(in_activations)) * dc_dw
        # biases
        for i in range(len(self.kernels[0])):
            dc_db = 0
            for x in range(len(in_activations)):
                dc_db += np.sum(dc_dz[x][i])
            self.biases[i] -= (LEARNING_RATE / len(in_activations)) * dc_db


class ResidualBlock:
    to = []
    kernels1 = []
    biases1 = []
    kernels2 = []
    biases2 = []
    __a1 = []
    __a2 = []

    def __init__(self, filters=FILTERS):
        self.kernels1 = [[np.random.randn(3, 3) for _ in range(filters)] for _ in range(filters)]
        self.biases1 = np.random.randn(filters)
        self.kernels2 = [[np.random.randn(3, 3) for _ in range(filters)] for _ in range(filters)]
        self.biases2 = np.random.randn(filters)

    def __activate(self, in_activations):
        self.__a1 = []
        for in_a in in_activations:
            conv = convolve_all_kernels(in_a, self.kernels1)
            for f in range(len(self.kernels1[0])):
                conv[f] += self.biases1[f]
            self.__a1.append(rectify(conv))
        self.__a2 = []
        for i in range(len(in_activations)):
            conv = convolve_all_kernels(self.__a1[i], self.kernels2)
            for f in range(len(self.kernels2[0])):
                conv[f] += self.biases2[f]
            self.__a2.append(rectify(conv + in_activations[i]))

    def feedforward(self, in_activations):
        self.__activate(in_activations)
        if len(self.to) == 1:
            return self.to[0].feedforward(self.__a2)
        else:
            return [self.to[i].feedforward(self.__a2) for i in range(len(self.to))]

    def sgd(self, in_activations, target_policies, target_values):
        self.__activate(in_activations)
        dc_da2s = [self.to[i].sgd(self.__a2, target_policies, target_values) for i in range(len(self.to))]
        dc_da2 = [sum(x) for x in zip(*dc_da2s)]
        da2_dz2 = [self.__a2[i] > 0 for i in range(len(self.__a2))]
        dc_dz2 = [np.multiply(dc_da2[i], da2_dz2[i]) for i in range(len(self.__a2))]
        dc_da1 = [convolve_all_kernels(x, invert_kernels(self.kernels2)) for x in dc_dz2]
        self.__update_params(self.__a1, dc_dz2, self.kernels2, self.biases2)
        da1_dz1 = [self.__a1[i] > 0 for i in range(len(self.__a1))]
        dc_dz1 = [np.multiply(dc_da1[i], da1_dz1[i]) for i in range(len(self.__a1))]
        dc_da_prev = []
        for i in range(len(in_activations)):
            dc_da_prev.append(convolve_all_kernels(dc_dz1[i], invert_kernels(self.kernels1)) + dc_dz2[i])
        self.__update_params(in_activations, dc_dz1, self.kernels1, self.biases1)
        return dc_da_prev

    @staticmethod
    def __update_params(activations, dc_dz, kernels, biases):
        # weights
        for i in range(len(kernels)):
            for j in range(len(kernels[0])):
                dc_dw = np.zeros((3, 3))
                for x in range(len(activations)):
                    dc_dw += convolve(activations[x][i], dc_dz[x][j])
                kernels[i][j] -= (LEARNING_RATE / len(activations)) * dc_dw
        # biases
        for i in range(len(kernels[0])):
            dc_db = 0
            for x in range(len(activations)):
                dc_db += np.sum(dc_dz[x][i])
            biases[i] -= (LEARNING_RATE / len(activations)) * dc_db


class PolicyHead:
    to = []


class ValueHead:
    to = []


class NeuralNet:
    head = None

    # TODO: L2 regularization
    # TODO: Batch Norm (can get rid of biases once this is implemented)
    def __init__(self):
        conv = ConvolutionalBlock()
        res = [ResidualBlock() for _ in range(RESIDUAL_BLOCKS)]
        conv.to = res[0]
        for i in range(RESIDUAL_BLOCKS-1):
            res[i].to = [res[i+1]]
        pol = PolicyHead()
        val = ValueHead()
        res[-1].to = [pol, val]
        self.head = conv

    def feedforward(self, in_activations):
        return self.head.feedforward(in_activations)

    def sgd(self, examples):
        self.head.sgd(examples[0], examples[1], examples[2])

    def create_checkpoint(self):
        return

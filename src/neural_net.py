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
    in_filters = None
    out_filters = None
    to = None
    kernels = []
    biases = []

    def __init__(self, in_filters=INPUT_PLANES, out_filters=FILTERS):
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernels = [[np.random.randn(3, 3) for i in range(out_filters)] for j in range(in_filters)]
        self.biases = np.random.randn(out_filters)

    def sgd(self, in_activations, target_policies, target_values):
        z = []
        for in_a in in_activations:
            conv = convolve_all_kernels(in_a, self.kernels)
            for f in range(self.out_filters):
                conv[f] += self.biases[f]
            z.append(conv)
        a = []
        for i in range(len(z)):
            a.append(rectify(z[i]))
        dc_da = self.to.sgd(a, target_policies, target_values)
        da_dz = [a[i] > 0 for i in range(len(a))]
        dc_dz = [np.multiply(dc_da[i], da_dz[i]) for i in range(len(a))]
        dc_da_prev = []
        for dc_dz_example in dc_dz:
            dc_da_prev.append(convolve_all_kernels(dc_dz_example, invert_kernels(self.kernels)))
        self.update_params(in_activations, dc_dz)
        return dc_da_prev
        # TODO: L2 regularization
        # TODO: Batch Norm (can get rid of biases once this is implemented)

    def update_params(self, in_activations, dc_dz):
        # weights
        for i in range(self.in_filters):
            for j in range(self.out_filters):
                dc_dw = np.zeros((3, 3))
                for x in range(len(in_activations)):
                    dc_dw += convolve(in_activations[x][i], dc_dz[x][j])
                self.kernels[i][j] -= (LEARNING_RATE / len(in_activations)) * dc_dw
        # biases
        for i in range(self.out_filters):
            dc_db = 0
            for x in range(len(in_activations)):
                dc_db += np.sum(dc_dz[x][i])
            self.biases[i] -= (LEARNING_RATE / len(in_activations)) * dc_db


class ResidualBlock:
    to = []
    kernels1 = []
    kernels2 = []

    def __init__(self):
        self.kernels1 = [[np.random.randn(3, 3) for i in range(FILTERS)] for j in range(FILTERS)]
        self.kernels2 = [[np.random.randn(3, 3) for i in range(FILTERS)] for j in range(FILTERS)]

    def sgd(self, in_activations, target_policies, target_values):
        # These will be helpful when implementing this function
        # dc_das = [self.to[i].sgd(a, target_policies, target_values) for i in range(len(self.to))]
        # dc_da = [sum(x) for x in zip(*dc_das)]
        return


class PolicyHead:
    to = []


class ValueHead:
    to = []


class NeuralNet:
    head = None

    def __init__(self):
        conv = ConvolutionalBlock()
        res = [ResidualBlock() for i in range(RESIDUAL_BLOCKS)]
        conv.to = res[0]
        for i in range(RESIDUAL_BLOCKS-1):
            res[i].to = [res[i+1]]
        pol = PolicyHead()
        val = ValueHead()
        res[-1].to = [pol, val]
        self.head = conv

    def sgd(self, examples):
        self.head.sgd(examples[0], examples[1], examples[2])

    def create_checkpoint(self):
        return

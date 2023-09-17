import numpy as np

INPUT_CHANNELS = 17
FILTERS = 16  # 256
RESIDUAL_BLOCKS = 19
BOARD_SIZE = 19
MINIBATCH_SIZE = 4  # 32 per worker

LEARNING_RATE = 0.01


def mse_derivative(a, pi):
    return [(-2 / (pi[i].size * len(a))) * (pi[i] - a[i]) for i in range(len(a))]


def softmax(a):
    e_to_a = np.exp(a - np.max(a))
    return e_to_a / e_to_a.sum()

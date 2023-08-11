RESIDUAL_BLOCKS = 3  # 19
INPUT_PLANES = 7  # 17
FILTERS = 16  # 256
BOARD_SIZE = 9  # 19
LEARNING_RATE = 0.001  # alphago zero uses annealing (extended data table 3)
MINIBATCH_SIZE = 32  # 32 on each of 64 workers; 2048 total
BATCHNORM_EPSILON = 1e-05  # alphago zero paper doesn't specify; this is the value from pytorch
WEIGHT_DECAY = 0.0001

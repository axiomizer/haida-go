EPOCHS = 10
STEPS_PER_EPOCH = 1000

# self-play settings
EPISODES = 250  # 25000
SIMULATIONS = 16  # 1600
STEPS_SAVED = 20
C_PUCT = 0.1
TEMP_THRESHHOLD = 30
KOMI = 7.5

# neural net architecture
BOARD_SIZE = 19
RESIDUAL_BLOCKS = 19
HISTORY_PLANES = 7
INPUT_CHANNELS = 2 * HISTORY_PLANES + 3
FILTERS = 256

# other neural net hyperparameters
MINIBATCH_SIZE = 32  # 32 on each of 64 workers; 2048 total
LR_SCHED = [(0,      0.01),
            (400000, 0.001),
            (600000, 0.0001)]
WEIGHT_DECAY = 0.0001  # l2 regularization parameter
MOMENTUM = 0.9

# game settings
KOMI = 7.5
BOARD_SIZE = 19

# training
EPOCHS = 10
TRAINING_STEPS = 50
LR_SCHED = [(0,      0.01),
            (400000, 0.001),
            (600000, 0.0001)]
MINIBATCH_SIZE = 128  # 32 on each of 64 workers; 2048 total
WEIGHT_DECAY = 0.0001  # l2 regularization parameter
MOMENTUM = 0.9

# self-play settings
EPISODES = 250  # 25000
SIMULATIONS = 16  # 1600
STEPS_SAVED = 20
C_PUCT = 0.1
TEMP_THRESHHOLD = 30

# neural net architecture
RESIDUAL_BLOCKS = 19
HISTORY_PLANES = 14
INPUT_CHANNELS = HISTORY_PLANES + 3
FILTERS = 256

import evaluator
import MCTS
from nn.haida_net import HaidaNet
import random


STEPS_PER_EPOCH = 1000

# self-play settings
EPISODES = 250  # 25000
SIMULATIONS = 16  # 1600
GAMES_SAVED = EPISODES * 20

# neural net architecture
BOARD_SIZE = 19
RESIDUAL_BLOCKS = 19
INPUT_CHANNELS = 17
FILTERS = 256

# other neural net hyperparameters
MINIBATCH_SIZE = 32  # 32 on each of 64 workers; 2048 total
LR_SCHED = [(0,      0.01),
            (400000, 0.001),
            (600000, 0.0001)]
WEIGHT_DECAY = 0.0001  # l2 regularization parameter


def pipeline():
    nn = HaidaNet(BOARD_SIZE, RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS)
    nn.configure(lr_sched=LR_SCHED, weight_decay=WEIGHT_DECAY)
    checkpoints = [nn.checkpoint()]
    best = 0  # index of best checkpoint
    examples = TrainingExamples()

    while True:
        for _ in range(STEPS_PER_EPOCH):
            new_training_examples = MCTS.self_play(checkpoints[best], EPISODES, SIMULATIONS)
            examples.put(new_training_examples)
            minibatch = examples.get_minibatch()
            nn.train(minibatch)
        checkpoints.append(nn.checkpoint())
        if evaluator.evaluate(checkpoints[best], checkpoints[-1]):
            best = len(checkpoints)-1


class TrainingExamples:
    examples = []  # grouped by iteration of self-play in which they were generated

    def put(self, new_examples):  # new_examples is a list of tuples (s, pi, z)
        self.examples = self.examples[1 - GAMES_SAVED / EPISODES:]
        self.examples.append(new_examples)

    def get_minibatch(self):
        flat = [example for iteration in self.examples for example in iteration]
        if MINIBATCH_SIZE > len(flat):
            raise ValueError("Not enough training examples to create a mini-batch")
        samples = random.sample(range(len(flat)), MINIBATCH_SIZE)
        minibatch = [[flat[i][0] for i in samples],
                     [flat[i][1] for i in samples],
                     [flat[i][2] for i in samples]]
        return minibatch  # minibatch is a single tuple (s, pi, z)

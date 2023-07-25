import evaluator
import MCTS
from nn.neural_net import NeuralNet
import random

BATCH_SIZE = 20  # 2048
EPISODES = 250  # 25000
SIMULATIONS = 16  # 1600
STEPS_PER_CHECKPOINT = 1000
GAMES_SAVED = EPISODES * 20


def pipeline():
    nn = NeuralNet()
    checkpoints = [nn.create_checkpoint()]
    best = 0  # index of best checkpoint
    training_steps = 0
    examples = TrainingExamples()

    while True:
        new_training_examples = MCTS.self_play(checkpoints[best], EPISODES, SIMULATIONS)
        examples.put(new_training_examples)

        nn.sgd(examples.get_minibatch())
        training_steps += 1

        if training_steps % STEPS_PER_CHECKPOINT is 0:
            checkpoints.append(nn.create_checkpoint())
            if evaluator.evaluate(checkpoints[best], checkpoints[-1]):
                best = len(checkpoints)-1


class TrainingExamples:
    examples = []  # grouped by iteration of self-play in which they were generated

    def put(self, new_examples):  # new_examples is a list of tuples (s, pi, z)
        self.examples = self.examples[1 - GAMES_SAVED/EPISODES:]
        self.examples.append(new_examples)

    def get_minibatch(self):
        flat = [example for iteration in self.examples for example in iteration]
        if BATCH_SIZE > len(flat):
            raise ValueError("Not enough training examples to create a mini-batch")
        samples = random.sample(range(len(flat)), BATCH_SIZE)
        minibatch = [[flat[i][0] for i in samples],
                     [flat[i][1] for i in samples],
                     [flat[i][2] for i in samples]]
        return minibatch  # minibatch is a single tuple (s, pi, z)

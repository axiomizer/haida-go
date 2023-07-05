import evaluator
from hyperparameters import hp
import MCTS
from neuralNet import NeuralNet
import numpy as np
import random


def pipeline():
    nn = NeuralNet()
    checkpoints = [nn.create_checkpoint()]
    best = 0  # index of best checkpoint
    training_steps = 0
    examples = TrainingExamples()

    while True:
        new_training_examples = MCTS.self_play(checkpoints[best], hp.episodes, hp.simulations)
        examples.put(new_training_examples)

        nn.optimize(examples.get_minibatch())
        training_steps += 1

        if training_steps % hp.steps_per_checkpoint is 0:
            checkpoints.append(nn.create_checkpoint())
            if evaluator.evaluate(checkpoints[best], checkpoints[-1]):
                best = len(checkpoints)-1


class TrainingExamples:
    examples = []  # grouped by iteration of self-play in which they were generated

    def put(self, new_examples):  # new_examples is a list of tuples of np arrays (s, pi, z)
        self.examples = self.examples[1 - hp.games_saved/hp.episodes:]
        self.examples.append(new_examples)

    def get_minibatch(self):
        flat = [example for iteration in self.examples for example in iteration]
        if hp.batch_size > len(flat):
            raise ValueError("Not enough training examples to create a mini-batch")
        samples = random.sample(range(len(flat)), hp.batch_size)
        minibatch = [np.array([flat[i][0] for i in samples]),
                     np.array([flat[i][1] for i in samples]),
                     np.array([flat[i][2] for i in samples])]
        return minibatch  # minibatch is a single tuple (s, pi, z) of np arrays

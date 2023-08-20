import random
from src.bot.config import *
import numpy as np


class TrainingExample:
    def __init__(self, nn_input, pi):
        self.nn_input = nn_input
        self.pi = pi
        self.z = None


class TrainingExamples:
    examples = []  # grouped by iteration of self-play in which they were generated

    def put(self, new_examples):
        self.examples = self.examples[1 - STEPS_SAVED:]
        self.examples.append(new_examples)

    def get_minibatch(self):
        flat = [example for iteration in self.examples for example in iteration]
        if MINIBATCH_SIZE > len(flat):
            raise ValueError("Not enough training examples to create a mini-batch")
        samples = random.sample(range(len(flat)), MINIBATCH_SIZE)
        minibatch = [np.ascontiguousarray([flat[i].nn_input for i in samples]),
                     np.ascontiguousarray([flat[i].pi for i in samples]),
                     np.ascontiguousarray([flat[i].z for i in samples])]
        return minibatch

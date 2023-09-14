import random
import numpy as np


class TrainingExample:
    def __init__(self, nn_input, pi, z):
        self.nn_input = nn_input
        self.pi = pi
        self.z = z


class Pool:
    class Builder:
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.examples = []

        def put_sample(self, new_examples, num_samples):
            self.examples.extend(random.sample(new_examples, num_samples))

        def build(self):
            random.shuffle(self.examples)
            return Pool(self.batch_size, self.examples)

    def __init__(self, batch_size, examples):
        self.batch_size = batch_size
        self.examples = examples
        self.ind = 0

    def num_batches(self):
        return (len(self.examples) - self.ind) // self.batch_size

    def get_minibatch(self):
        if self.num_batches() < 1:
            raise ValueError('Not enough examples left')
        examples = self.examples[self.ind:self.ind+self.batch_size]
        self.ind += self.batch_size
        return [np.ascontiguousarray([ex.nn_input for ex in examples]),
                np.ascontiguousarray([ex.pi for ex in examples]),
                np.ascontiguousarray([ex.z for ex in examples])]


class EvolvingPool:
    examples = []  # list of lists; one entry for each call to put()

    def __init__(self, steps_saved, batch_size):
        self.steps_saved = steps_saved
        self.batch_size = batch_size

    def put(self, new_examples):
        self.examples = self.examples[1 - self.steps_saved:]
        self.examples.append(new_examples)

    def get_minibatch(self):
        flat = [example for iteration in self.examples for example in iteration]
        if self.batch_size > len(flat):
            raise ValueError("Not enough training examples to create a mini-batch")
        samples = random.sample(range(len(flat)), self.batch_size)
        minibatch = [np.ascontiguousarray([flat[i].nn_input for i in samples]),
                     np.ascontiguousarray([flat[i].pi for i in samples]),
                     np.ascontiguousarray([flat[i].z for i in samples])]
        return minibatch

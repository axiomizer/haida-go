from abc import ABC, abstractmethod
from numpy import ndarray


# methods and config data shared among blocks of a network

class Config:
    def __init__(self):
        self.learning_rate = 0.01
        self.lr_sched = []
        self.lr_step = 0
        self.weight_decay = 0
        self.momentum = 0
        self.batch_norm_epsilon = 1e-05  # alphago zero doesn't specify; use value from pytorch as default

    def set_lr(self):
        temp = self.learning_rate
        for m, lr in self.lr_sched:
            if self.lr_step >= m:
                temp = lr
        self.learning_rate = temp


class AbstractNet(ABC):
    def __init__(self, config):
        self.cfg = config or Config()  # one shared config instance

    # example of learning rate schedule: [(0, 0.01), (100, 0.001), (200, 0.0001)]
    def configure(self, learning_rate=None, lr_sched=None, weight_decay=None, momentum=None, batch_norm_epsilon=None):
        if learning_rate is not None:
            self.cfg.learning_rate = learning_rate
        if lr_sched is not None:
            self.cfg.lr_sched = lr_sched
            self.cfg.set_lr()
        if weight_decay is not None:
            self.cfg.weight_decay = weight_decay
        if momentum is not None:
            self.cfg.momentum = momentum
        if batch_norm_epsilon is not None:
            self.cfg.batch_norm_epsilon = batch_norm_epsilon

    def step_lr_sched(self):
        self.cfg.lr_step += 1
        self.cfg.set_lr()

    def update_theta(self, theta, err, err_runavg):
        if type(theta) != ndarray:
            raise ValueError('Found theta of unexpected type: {} (may not be mutable)'.format(type(theta)))
        if type(err_runavg) != ndarray:
            raise ValueError('Found err_runavg of unexpected type: {} (may not be mutable)'.format(type(err_runavg)))
        err_runavg *= self.cfg.momentum
        err_runavg += err + 2 * self.cfg.weight_decay * theta
        theta -= self.cfg.learning_rate * err_runavg

    def train(self, minibatch):
        self.feedforward(minibatch[0])
        err = self.error(minibatch[1:])
        self.backprop(err)
        self.step_lr_sched()

    @abstractmethod
    def feedforward(self, activations):
        pass

    @abstractmethod
    def error(self, target):
        pass

    @abstractmethod
    def backprop(self, err):
        pass

    @abstractmethod
    def checkpoint(self):
        pass

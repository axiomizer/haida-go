from numpy import ndarray


class Config:
    def __init__(self, learning_rate=0.001, weight_decay=0.0001, batch_norm_epsilon=1e-05):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_norm_epsilon = batch_norm_epsilon  # alphago zero doesn't specify; use value from pytorch as default

    def theta_update_rule(self, theta, err):
        if type(theta) != ndarray:
            raise ValueError('Found theta of unexpected type: {} (may not be mutable)'.format(type(theta)))
        theta *= 1 - 2 * self.learning_rate * self.weight_decay
        theta -= self.learning_rate * err

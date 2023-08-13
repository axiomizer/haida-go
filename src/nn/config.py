from numpy import ndarray


class Config:
    # example learning rate schedule:
    #    [(0,   0.01),
    #     (100, 0.001),
    #     (200, 0.0001)]
    # alphago zero doesn't specify batch_norm_epsilon; use value from pytorch as default
    def __init__(self, learning_rate=0.01, lr_sched=None, weight_decay=0, batch_norm_epsilon=1e-05):
        self.learning_rate = learning_rate
        self.__lr_sched = lr_sched
        if lr_sched is not None:
            self.__step = 0
            self.__set_lr()
        self.weight_decay = weight_decay
        self.batch_norm_epsilon = batch_norm_epsilon

    def __set_lr(self):
        learning_rate = self.learning_rate
        for m, lr in self.__lr_sched:
            if self.__step >= m:
                learning_rate = lr
        self.learning_rate = learning_rate

    def step_lr_sched(self):
        if self.__lr_sched is None:
            return
        self.__step += 1
        self.__set_lr()

    def theta_update_rule(self, theta, err):
        if type(theta) != ndarray:
            raise ValueError('Found theta of unexpected type: {} (may not be mutable)'.format(type(theta)))
        theta *= 1 - 2 * self.learning_rate * self.weight_decay
        theta -= self.learning_rate * err

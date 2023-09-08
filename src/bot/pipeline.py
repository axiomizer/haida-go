from src.bot.agent import Agent, GameOver
from src.bot.nn.haida_net import HaidaNet
from src.bot.training_examples import TrainingExamples
from src.bot.config import *
from src.bot.evaluation import pit
from util.progress_bar import ProgressBar
import copy


def self_play(nn):
    new_training_examples = []
    for _ in range(EPISODES):
        agent = Agent(nn, BOARD_SIZE, HISTORY_PLANES, True)
        while True:
            try:
                agent.move(SIMULATIONS)
            except GameOver:
                break
        new_training_examples += agent.training_examples
    return new_training_examples


def train():
    nn = HaidaNet(BOARD_SIZE, RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS)
    nn.configure(lr_sched=LR_SCHED, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, compute_batch_stats=True)

    # doing some self-play up front generates training examples and helps initialize batch norm stats
    self_play(nn)

    checkpoints = [copy.deepcopy(nn)]
    checkpoints[0].configure(compute_batch_stats=False)
    best = 0  # index of best checkpoint
    examples = TrainingExamples()
    progress_bar = ProgressBar(EPOCHS * STEPS_PER_EPOCH)
    progress_bar.start()
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            new_training_examples = self_play(checkpoints[best])
            examples.put(new_training_examples)
            minibatch = examples.get_minibatch()
            nn.train(minibatch)
            progress_bar.increment()
        checkpoints.append(copy.deepcopy(nn))
        checkpoints[-1].configure(compute_batch_stats=False)
        if pit(checkpoints[-1], checkpoints[best]):
            best = len(checkpoints) - 1
    progress_bar.end()

    return checkpoints[best]

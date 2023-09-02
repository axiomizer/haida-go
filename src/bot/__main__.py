from src.bot.agent import Agent, GameOver
from src.bot.nn.haida_net import HaidaNet
from src.bot import evaluator
from src.bot.training_examples import TrainingExamples
from src.bot.config import *
from util.progress_bar import ProgressBar
import os
import pickle
import sys
import copy
from pathlib import Path


TRAINED_NETS_PATH = os.path.join('src', 'bot', 'trained_nets')


def self_play(nn):
    new_training_examples = []
    for _ in range(EPISODES):
        agent = Agent(nn, SIMULATIONS, BOARD_SIZE, HISTORY_PLANES)
        while True:
            try:
                agent.move()
            except GameOver:
                break
        new_training_examples += agent.training_examples
    return new_training_examples


def pipeline(name):
    Path(TRAINED_NETS_PATH).mkdir(exist_ok=True)
    filename = os.path.join(TRAINED_NETS_PATH, '{}.pickle'.format(name))
    if os.path.isfile(filename):
        raise ValueError('file already exists: {}'.format(filename))

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
        if evaluator.evaluate(checkpoints[best], checkpoints[-1]):
            best = len(checkpoints) - 1
    progress_bar.end()

    # save neural net
    with open(filename, 'wb') as f:
        pickle.dump(checkpoints[best], f, pickle.HIGHEST_PROTOCOL)
    print('Trained net saved to: {}'.format(filename))


if len(sys.argv) <= 1:
    sys.exit(2)
else:
    pipeline(sys.argv[1])

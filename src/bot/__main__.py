from src.bot.agent import Agent
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


def pipeline(name):
    Path(TRAINED_NETS_PATH).mkdir(exist_ok=True)
    filename = os.path.join(TRAINED_NETS_PATH, '{}.pickle'.format(name))
    if os.path.isfile(filename):
        raise ValueError('file already exists: {}'.format(filename))

    nn = HaidaNet(BOARD_SIZE, RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS)
    nn.configure(lr_sched=LR_SCHED, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    checkpoints = [copy.deepcopy(nn)]
    best = 0  # index of best checkpoint
    examples = TrainingExamples()
    progress_bar = ProgressBar(EPOCHS * STEPS_PER_EPOCH)
    progress_bar.start()
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            new_training_examples = []
            for _ in range(EPISODES):
                tree = Agent(checkpoints[best], SIMULATIONS, BOARD_SIZE, HISTORY_PLANES)
                while not tree.game_concluded():
                    tree.move()
                new_training_examples += tree.training_examples
            examples.put(new_training_examples)
            minibatch = examples.get_minibatch()
            nn.train(minibatch)
            progress_bar.increment()
        checkpoints.append(copy.deepcopy(nn))
        if evaluator.evaluate(checkpoints[best], checkpoints[-1]):
            best = len(checkpoints) - 1
    progress_bar.end()

    # save neural net
    with open(filename, 'wb') as f:
        pickle.dump(nn, f, pickle.HIGHEST_PROTOCOL)
    print('Trained net saved to: {}'.format(filename))


def exhibit(name):
    filename = os.path.join(TRAINED_NETS_PATH, '{}.pickle'.format(name))
    with open(filename, 'rb') as f:
        nn = pickle.load(f)
    tree = Agent(nn, SIMULATIONS, BOARD_SIZE, HISTORY_PLANES)
    while not tree.game_concluded():
        tree.move()
        print(tree.root)
        input('hit enter to continue...')


if len(sys.argv) <= 2:
    sys.exit(2)
else:
    if sys.argv[1] == 'train':
        pipeline(sys.argv[2])
    elif sys.argv[1] == 'exhibit':
        exhibit(sys.argv[2])
    else:
        sys.exit(2)

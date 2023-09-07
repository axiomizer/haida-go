from src.bot.agent import Agent, GameOver
from src.bot.nn.haida_net import HaidaNet
from src.bot.training_examples import TrainingExamples
from src.bot.config import *
from src.game import Color
from util.progress_bar import ProgressBar
import os
import pickle
import sys
import copy
from pathlib import Path
import itertools


TRAINED_NETS_PATH = os.path.join('src', 'bot', 'trained_nets')


# return True if nn1 wins, False if nn2 wins
def pit(nn1, nn2):
    agent1 = Agent(nn1, BOARD_SIZE, HISTORY_PLANES, False)
    agent2 = Agent(nn2, BOARD_SIZE, HISTORY_PLANES, False)
    while True:
        try:
            action = agent1.move(SIMULATIONS)
            agent2.inform_move(action)
            action = agent2.move(SIMULATIONS)
            agent1.inform_move(action)
        except GameOver:
            break
    if agent1.root.winner is not agent2.root.winner:
        raise ValueError('agents disagree about who won')
    return agent1.root.winner is Color.BLACK


def rank_bots():
    # load bots
    nn_names = [name.split('.')[0] for name in os.listdir(TRAINED_NETS_PATH)]
    nns = {}  # name -> nn instance
    for name in nn_names:
        filename = os.path.join(TRAINED_NETS_PATH, '{}.pickle'.format(name))
        with open(filename, 'rb') as f:
            nns[name] = pickle.load(f)

    # pit bots
    games_per_pair = 3
    progress_bar = ProgressBar(len(nn_names) * (len(nn_names) - 1) * games_per_pair)
    progress_bar.start()
    elo = {nn: 1000 for nn in nns}  # name -> elo
    for nn1_name, nn2_name in itertools.permutations(nns, 2):
        for _ in range(games_per_pair):
            result = pit(nns[nn1_name], nns[nn2_name])
            e1 = 1 / (1 + 10 ** ((elo[nn2_name] - elo[nn1_name]) / 400))
            e2 = 1 / (1 + 10 ** ((elo[nn1_name] - elo[nn2_name]) / 400))
            elo[nn1_name] += 32 * (result - e1)
            elo[nn2_name] += 32 * ((not result) - e2)
            progress_bar.increment()
    progress_bar.end()

    # print ratings
    for k, v in sorted(elo.items(), key=lambda item: item[1]):
        print('{}:\t{}'.format(k, int(v)))


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
        if pit(checkpoints[-1], checkpoints[best]):
            best = len(checkpoints) - 1
    progress_bar.end()

    # save neural net
    with open(filename, 'wb') as f:
        pickle.dump(checkpoints[best], f, pickle.HIGHEST_PROTOCOL)
    print('Trained net saved to: {}'.format(filename))


if len(sys.argv) <= 1:
    sys.exit(2)
if sys.argv[1] == 'train':
    if len(sys.argv) <= 2:
        sys.exit(2)
    pipeline(sys.argv[2])
elif sys.argv[1] == 'rank':
    rank_bots()

from src.bot.agent import Agent, GameOver
from src.bot.config import *
from src.game import Color
from src.progress_bar import ProgressBar
from src.sgf import SGF
import os
import pickle
import itertools


EVALUATION_GAMES_PATH = os.path.join('src', 'bot', 'evaluation_games')


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


def rank_bots(path):
    # load bots
    nn_names = [name.split('.')[0] for name in os.listdir(path)]
    nns = {}  # name -> nn instance
    for name in nn_names:
        filename = os.path.join(path, '{}.pickle'.format(name))
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


def exhibit(nn):
    agent = Agent(nn, BOARD_SIZE, HISTORY_PLANES, False)
    moves = []
    print('Playing match', end='', flush=True)
    while not agent.game_over():
        action = agent.move(SIMULATIONS)
        if action == BOARD_SIZE ** 2:
            moves.append(None)
        else:
            moves.append((action // BOARD_SIZE, action % BOARD_SIZE))
        print('.', end='', flush=True)
    print('', flush=True)
    sgf = SGF.build(BOARD_SIZE, KOMI, agent.root.winner, moves)
    filename = os.path.join(EVALUATION_GAMES_PATH, '{}.sgf'.format('match'))
    with open(filename, 'w') as f:
        f.write(sgf.to_string())
    print('SGF file saved to: {}'.format(filename), flush=True)

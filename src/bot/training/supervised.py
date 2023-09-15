import os
import tarfile
import numpy as np
from src.game import Color, Board
from src.bot.training.training_examples import TrainingExample, Pool
from src.bot import nn_input
from src.bot.config import *
from src.bot.nn.haida_net import HaidaNet
from src.sgf import SGF, GameResult
from src.progress_bar import ProgressBar


KGS_GAMES_PATH = os.path.join('src', 'bot', 'training', 'kgs')


def validate_sgf(sgf: SGF):
    r = sgf.root
    if r.board_size != BOARD_SIZE or r.ruleset != 'Chinese' or r.komi != KOMI or r.handicap != 0:
        return False
    if r.result.winner is not GameResult.Winner.BLACK and r.result.winner is not GameResult.Winner.WHITE:
        return False
    color_to_play = Color.BLACK
    for m in sgf.moves:
        if m.player is not color_to_play:
            return False
        color_to_play = color_to_play.opponent()
    return True


# assumes sgf has been validated
def examples_from_sgf(sgf: SGF):
    size = sgf.root.board_size
    training_examples = []
    board = Board(size)
    nn_in = None
    z = 1 if sgf.root.result.winner is GameResult.Winner.BLACK else -1
    for m in sgf.moves:
        nn_in = nn_input.compose(nn_in, board, HISTORY_PLANES)
        if m.move is None:
            action = size ** 2
        else:
            action = m.move[0] * size + m.move[1]
        pi = np.zeros(size ** 2 + 1)
        pi[action] = 1
        ex = TrainingExample(nn_in, pi, z)
        training_examples.append(ex)
        z *= -1
        if m.move is not None:
            board.place_stone(m.move, m.player)
    return training_examples


def examples_from_kgs_data(examples_per_game):
    pool = Pool.Builder(MINIBATCH_SIZE)
    skipped_games = 0
    total_games = 0
    archives = [ar for ar in os.listdir(KGS_GAMES_PATH) if not ar.startswith('.')]  # get kgs games

    print('Parsing KGS data...')
    progress_bar = ProgressBar(len(archives))
    progress_bar.start()
    for archive in archives:
        with tarfile.open(os.path.join(KGS_GAMES_PATH, archive), 'r:gz') as tf:
            for member in tf:
                if member.isdir():
                    continue
                total_games += 1
                f = tf.extractfile(member)
                string = f.read().decode('UTF-8')
                sgf = SGF.from_string(string)
                if (not validate_sgf(sgf)) or len(sgf.moves) < examples_per_game:
                    skipped_games += 1
                    continue
                new_training_examples = examples_from_sgf(sgf)
                pool.put_sample(new_training_examples, examples_per_game)
        progress_bar.increment()
    progress_bar.end()
    print('Usable games: {}/{}'.format(total_games - skipped_games, total_games))
    return pool.build()


def train():
    nn = HaidaNet(BOARD_SIZE, RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS)
    nn.configure(lr_sched=LR_SCHED, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, compute_batch_stats=True)
    pool = examples_from_kgs_data(30)
    if pool.num_batches() < (TRAINING_STEPS + 1) * EPOCHS:
        raise ValueError('Not enough training examples')

    for i in range(EPOCHS):
        print('Training epoch {}/{}...'.format(i + 1, EPOCHS))
        progress_bar = ProgressBar(TRAINING_STEPS)
        progress_bar.start()
        for _ in range(TRAINING_STEPS):
            minibatch = pool.get_minibatch()
            nn.train(minibatch)
            progress_bar.increment()
        progress_bar.end()

        # get loss
        minibatch = pool.get_minibatch()
        nn.feedforward(minibatch[0])
        loss = nn.loss(minibatch[1:])
        print('Loss: {}'.format(loss))
    return nn

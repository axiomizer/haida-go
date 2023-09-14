import os
import tarfile
import numpy as np
from src.game import Color, Board
from src.bot.training.training_examples import TrainingExample, Pool
from src.bot import nn_input
from src.bot.config import *
from src.bot.nn.haida_net import HaidaNet
from src.progress_bar import ProgressBar


KGS_GAMES_PATH = os.path.join('src', 'bot', 'training', 'kgs')


# https://www.red-bean.com/sgf/index.html
class SGF:
    class Node:
        def __init__(self, string):
            self.properties = {}
            end_ind = -1
            ind = string.index('[')
            while ind != -1:
                prop_ident = string[end_ind+1:ind].strip()
                prop_values = []
                end_ind = ind - 1
                while ind != -1 and string[end_ind+1:ind].strip() == '':
                    end_ind = string.index(']', ind)
                    prop_values.append(string[ind+1:end_ind].strip())
                    ind = string.find('[', end_ind)
                self.properties[prop_ident] = prop_values

    BOARD_SIZE = 19

    def __init__(self, string):
        sequence = self.__main_branch(string)
        nodes = sequence.split(';')[1:]

        # root node
        root = SGF.Node(nodes[0])
        self.__validate('GM' in root.properties and root.properties['GM'] == ['1'])
        self.__validate('SZ' in root.properties and root.properties['SZ'] == [str(BOARD_SIZE)])
        self.__validate('RU' in root.properties and root.properties['RU'] == ['Chinese'])
        self.__validate('KM' in root.properties and float(root.properties['KM'][0]) == KOMI)
        self.__validate('AB' not in root.properties and
                        'AW' not in root.properties and
                        'HA' not in root.properties)  # no handicap
        self.__validate('RE' in root.properties and root.properties['RE'][0][0] in {'B', 'W'})  # game result
        if root.properties['RE'][0][0] == 'B':
            self.result = 1
        else:
            self.result = -1

        # move nodes
        moves = [SGF.Node(n) for n in nodes[1:]]
        self.moves = []  # list of integers
        color_to_play = Color.BLACK
        for move in moves:
            if color_to_play is Color.BLACK:
                self.__validate('B' in move.properties and 'W' not in move.properties)
                pos = move.properties['B'][0]
            else:
                self.__validate('W' in move.properties and 'B' not in move.properties)
                pos = move.properties['W'][0]
            self.__validate(pos == '' or (len(pos) == 2 and all(map(lambda c: ord('a') <= ord(c) <= ord('z'), pos))))
            color_to_play = color_to_play.opponent()

            if pos == '':
                self.moves.append(self.BOARD_SIZE ** 2)
            else:
                column = ord(pos[0]) - ord('a')
                row = ord(pos[1]) - ord('a')
                self.moves.append(row * self.BOARD_SIZE + column)

    def get_training_examples(self):
        training_examples = []
        board = Board(self.BOARD_SIZE)
        nn_in = None
        z = self.result
        color_to_play = Color.BLACK
        for action in self.moves:
            nn_in = nn_input.compose(nn_in, board, HISTORY_PLANES)
            pi = np.zeros(self.BOARD_SIZE ** 2 + 1)
            pi[action] = 1
            ex = TrainingExample(nn_in, pi, z)
            training_examples.append(ex)
            z *= -1
            if action != self.BOARD_SIZE ** 2:
                coordinates = (action // self.BOARD_SIZE, action % self.BOARD_SIZE)
                board.place_stone(coordinates, color_to_play)
            color_to_play = color_to_play.opponent()
        return training_examples

    @staticmethod
    def __validate(condition, string=None):
        if not condition:
            raise ValueError('Cannot handle sgf file ({})'.format(string))

    def __main_branch(self, data):
        depth = 0
        start = data.find('(')
        if start == -1:
            return data
        for i in range(start, len(data)):
            if data[i] == '(':
                depth += 1
            elif data[i] == ')':
                depth -= 1
            if depth == 0:
                return data[:start] + self.__main_branch(data[start+1:i])
        raise ValueError('Unbalanced parentheses')


def parse_kgs_examples(examples_per_game):
    training_examples = Pool.Builder(MINIBATCH_SIZE)
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
                try:
                    sgf = SGF(string)
                except ValueError:
                    skipped_games += 1
                    continue
                if len(sgf.moves) < examples_per_game:
                    skipped_games += 1
                    continue
                new_training_examples = sgf.get_training_examples()
                training_examples.put_sample(new_training_examples, examples_per_game)
        progress_bar.increment()
    progress_bar.end()
    print('Usable games: {}/{}'.format(total_games - skipped_games, total_games))
    return training_examples.build()


def train():
    nn = HaidaNet(BOARD_SIZE, RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS)
    nn.configure(lr_sched=LR_SCHED, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, compute_batch_stats=True)
    training_examples = parse_kgs_examples(30)
    if training_examples.num_batches() < (TRAINING_STEPS + 1) * EPOCHS:
        raise ValueError('Not enough training examples')

    for i in range(EPOCHS):
        print('Training epoch {}/{}...'.format(i + 1, EPOCHS))
        progress_bar = ProgressBar(TRAINING_STEPS)
        progress_bar.start()
        for _ in range(TRAINING_STEPS):
            minibatch = training_examples.get_minibatch()
            nn.train(minibatch)
            progress_bar.increment()
        progress_bar.end()

        # get loss
        minibatch = training_examples.get_minibatch()
        nn.feedforward(minibatch[0])
        loss = nn.loss(minibatch[1:])
        print('Loss: {}'.format(loss))
    return nn

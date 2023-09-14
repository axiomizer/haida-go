from src.game import Board, Color, IllegalMove
from src.bot.nn.haida_net import HaidaNet
from src.bot.training.training_examples import TrainingExample
from src.bot import nn_input
import numpy as np
from src.bot.config import *


class GameOverNode:
    def __init__(self, parent_node, board: Board):
        color_to_play = parent_node.color_to_play.opponent() if parent_node is not None else Color.BLACK
        b_score, w_score = board.score(KOMI)
        self.winner = Color.BLACK if b_score > w_score else Color.WHITE
        self.z = 1 if (b_score > w_score) == (color_to_play is Color.BLACK) else -1


class Node:
    def __init__(self, parent_node, board: Board, consecutive_passes, nn_in, p):
        self.parent = parent_node
        self.depth = parent_node.depth + 1 if parent_node is not None else 0
        self.children = {}  # action -> node
        self.actions_determined_illegal = set()

        # game state
        self.board = board
        self.nn_in = nn_in
        self.color_to_play = parent_node.color_to_play.opponent() if parent_node is not None else Color.BLACK
        self.consecutive_passes = consecutive_passes

        # search stats
        num_actions = self.board.size ** 2 + 1
        self.N = np.zeros(num_actions, dtype=int)
        self.W = np.zeros(num_actions, dtype=float)
        self.Q = np.zeros(num_actions, dtype=float)
        self.P = p

    def get_criterion(self):
        return self.Q + C_PUCT * self.P * (sum(self.N) ** 0.5) / (1 + self.N)

    def get_distribution(self, temperature: int):
        if temperature == 0:
            max_arr = self.N == np.max(self.N)
            return max_arr / np.count_nonzero(max_arr)
        exp = self.N ** (1 / temperature)
        return exp / sum(exp)

    def update(self, action, v):
        self.N[action] += 1
        self.W[action] += v
        self.Q[action] = self.W[action] / self.N[action]


class GameOver(Exception):
    pass


class Agent:
    def __init__(self, nn: HaidaNet, board_size, history_planes, generate_training_examples):
        self.nn = nn
        self.history_planes = history_planes
        self.board_size = board_size

        self.prior_positions = []  # instances of Board in chronological order
        self.generate_training_examples = generate_training_examples
        self.training_examples = []  # instances of TrainingExample in chronological order
        self.temperature = 1

        # initialize root
        board = Board(board_size)
        nn_in = nn_input.compose(None, board, history_planes)
        nn_out = self.nn.feedforward(np.expand_dims(nn_in, 0))
        self.root = Node(None, board, 0, nn_in, nn_out[0][0])

    def game_over(self):
        return type(self.root) is GameOverNode

    def inform_move(self, action):
        if type(self.root) is GameOverNode:
            raise GameOver()

        self.prior_positions.append(self.root.board)

        # update root
        if action not in self.root.children:
            self.__create_leaf(self.root, action)
        self.root = self.root.children[action]
        self.root.parent = None

        # update temperature
        if len(self.prior_positions) == TEMP_THRESHHOLD:
            self.temperature = 0

    def move(self, num_simulations):
        if type(self.root) is GameOverNode:
            raise GameOver()
        for _ in range(num_simulations):
            self.__mcts(self.root)

        # add new training example
        if self.generate_training_examples:
            pi = self.root.get_distribution(1)
            new_example = TrainingExample(self.root.nn_in, pi, None)
            self.training_examples.append(new_example)

        # add prior position
        self.prior_positions.append(self.root.board)

        # update root
        pi = self.root.get_distribution(self.temperature)
        action = np.random.choice(range(self.board_size ** 2 + 1), p=pi)
        self.root = self.root.children[action]
        self.root.parent = None

        # update temperature
        if len(self.prior_positions) == TEMP_THRESHHOLD:
            self.temperature = 0

        # check game over
        if type(self.root) is GameOverNode and self.generate_training_examples:
            z = self.root.z
            for ex in reversed(self.training_examples):
                z *= -1
                ex.z = z

        return action

    def __is_repeated_position(self, board, node):
        curr = node
        while curr is not None:
            if board == curr.board:
                return True
            curr = curr.parent
        return board in self.prior_positions

    # returns value of new leaf, or raises exception if action is illegal
    def __create_leaf(self, node, action):
        if action == self.board_size ** 2:  # if action is a pass
            new_board = node.board.copy()
            consecutive_passes = node.consecutive_passes + 1
        else:
            new_board = node.board.copy()
            coordinates = (action // self.board_size, action % self.board_size)
            new_board.place_stone(coordinates, node.color_to_play)
            if self.__is_repeated_position(new_board, node):
                raise IllegalMove()
            consecutive_passes = 0
        if consecutive_passes == 2 or node.depth + 1 > 2 * (self.board_size ** 2):  # game over
            new_leaf = GameOverNode(node, new_board)
            v = new_leaf.z
        else:
            nn_in = nn_input.compose(node.nn_in, new_board, self.history_planes)
            nn_out = self.nn.feedforward(np.expand_dims(nn_in, 0))
            new_leaf = Node(node, new_board, consecutive_passes, nn_in, nn_out[0][0])
            v = nn_out[1][0]
        node.children[action] = new_leaf
        return v

    # monte carlo tree search: recursive function returning value of node (from its own perspective)
    def __mcts(self, node):
        if type(node) is GameOverNode:
            return node.z
        criterion = node.get_criterion()
        while True:
            # try picking best action
            best_action = None
            for a in range(len(criterion)):
                if a in node.actions_determined_illegal:
                    continue
                if best_action is None or criterion[a] > criterion[best_action]:
                    best_action = a
            if best_action is None:
                raise ValueError('no legal moves found for node:\n{}'.format(node))

            if best_action in node.children:  # recurse if possible
                v = self.__mcts(node.children[best_action])
            else:  # create leaf, if legal move
                try:
                    v = self.__create_leaf(node, best_action)
                except IllegalMove:
                    node.actions_determined_illegal.add(best_action)
                    continue
            node.update(best_action, -1 * v)
            return -1 * v

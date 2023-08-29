from src.game import Board, Color, IllegalMove
from src.bot.nn.haida_net import HaidaNet
from src.bot.training_examples import TrainingExample
import numpy as np
from src.bot.config import *


class Node:
    def __init__(self, parent_node, depth, board: Board, color_to_play: Color, consecutive_passes):
        self.parent = parent_node
        self.depth = depth
        self.children = {}  # action -> node
        self.num_actions = board.size ** 2 + 1
        self.action_determined_illegal = np.zeros(self.num_actions, dtype=bool)

        # game state
        self.board = board
        self.color_to_play = color_to_play
        self.consecutive_passes = consecutive_passes

        # search stats
        self.N = None
        self.W = None
        self.Q = None
        self.P = None

    def check_game_over(self):
        return self.consecutive_passes == 2 or self.depth > 2 * (self.board.size ** 2)

    def final_reward(self):
        if not self.check_game_over():
            raise ValueError('game is not over\n{}'.format(self))
        b_score, w_score = self.board.score(KOMI)
        return 1 if (b_score > w_score) == (self.color_to_play is Color.BLACK) else -1

    def get_criterion(self):
        return self.Q + C_PUCT * self.P * (sum(self.N) ** 0.5) / (1 + self.N)

    def get_distribution(self, temperature: int):
        if temperature == 0:
            max_arr = self.N == np.max(self.N)
            return max_arr / np.count_nonzero(max_arr)
        exp = self.N ** (1 / temperature)
        return exp / sum(exp)

    def expand(self, p):
        if self.N is not None or self.W is not None or self.Q is not None or self.P is not None:
            raise ValueError('node is already expanded:\n{}'.format(self))
        self.N = np.zeros(self.num_actions, dtype=int)
        self.W = np.zeros(self.num_actions, dtype=float)
        self.Q = np.zeros(self.num_actions, dtype=float)
        self.P = p

    def update(self, action, v):
        self.N[action] += 1
        self.W[action] += v
        self.Q[action] = self.W[action] / self.N[action]

    def __str__(self):
        if self.N is not None:
            tp = 'W to play' if self.color_to_play is Color.WHITE else 'B to play'
            n = 'N:{}'.format(self.N)
            q = 'Q:[{}]'.format(' '.join(['{0:0.1f}'.format(q) for q in self.Q]))
            length = len(q)
            tp = tp + ' '*(length - len(tp))
            n = n + ' '*(length - len(n))
            b = self.board.__str__().split('\n')
            b = [s + ' '*(length - len(s)) for s in b]
            return '{}\n{}\n{}\n{}'.format(tp, n, q, '\n'.join(b))
        else:
            return self.board.__str__()


class Agent:
    def __init__(self, nn: HaidaNet, num_simulations, board_size, history_planes):
        self.nn = nn
        self.num_simulations = num_simulations
        self.history_planes = history_planes
        self.board_size = board_size

        # lists should be in chronological order
        self.prior_positions = []  # instances of Board
        self.training_examples = []

        self.temperature = 1

        # initialize root
        self.root = Node(None, 0, Board(board_size), Color.BLACK, 0)
        p, _ = self.__evaluate(self.root)
        self.root.expand(p)

    def __str__(self):
        def concat(str1, str2):
            spl1 = str1.split('\n')
            spl2 = str2.split('\n')
            if len(spl1) > len(spl2):
                spl2 += [' ' * len(spl2[0])] * (len(spl1) - len(spl2))
            elif len(spl2) > len(spl1):
                spl1 += [' ' * len(spl1[0])] * (len(spl2) - len(spl1))
            return '\n'.join([a + ' ' + b for a, b in zip(spl1, spl2)])

        def recurse(node):
            ret = node.__str__()
            if len(node.children) == 0:
                return ret
            spl = [recurse(c) for _, c in node.children.items()]
            cat = spl[0]
            for s in spl[1:]:
                cat = concat(cat, s)
            length = len(cat.split('\n')[0])
            ret = '\n'.join([n + ' '*(length - len(n)) for n in ret.split('\n')])
            return ret + '\n' + cat

        return recurse(self.root)

    def game_concluded(self):
        return self.root.check_game_over()

    def move(self):
        for _ in range(self.num_simulations):
            self.__mcts(self.root)

        # add new training example
        pi = self.root.get_distribution(1)
        new_example = TrainingExample(self.__compose_nn_input_planes(self.root), pi)
        self.training_examples.append(new_example)

        # add prior position
        self.prior_positions.append(self.root.board)

        # update root
        if self.temperature != 1:
            pi = self.root.get_distribution(self.temperature)
        action = np.random.choice(range(self.board_size ** 2 + 1), p=pi)
        self.root = self.root.children[action]
        self.root.parent = None

        # update temperature
        if len(self.prior_positions) == TEMP_THRESHHOLD:
            self.temperature = 0

        # check game over
        if self.root.check_game_over():
            z = self.root.final_reward()
            for ex in reversed(self.training_examples):
                z *= -1
                ex.z = z

    def __is_repeated_position(self, board, node):
        curr = node
        while curr is not None:
            if board == curr.board:
                return True
            curr = curr.parent
        return board in self.prior_positions

    # returns new leaf, or returns None if action is illegal
    def __create_leaf(self, node, action):
        if action == self.board_size ** 2:  # if action is a pass
            new_leaf = Node(node,
                            node.depth + 1,
                            node.board.copy(),
                            node.color_to_play.opponent(),
                            node.consecutive_passes + 1)
            node.children[action] = new_leaf
            return new_leaf
        new_board = node.board.copy()
        coordinates = (action // self.board_size, action % self.board_size)
        try:
            new_board.place_stone(coordinates, node.color_to_play)
        except IllegalMove:
            return None
        if self.__is_repeated_position(new_board, node):
            return None
        new_leaf = Node(node, node.depth + 1, new_board, node.color_to_play.opponent(), 0)
        node.children[action] = new_leaf
        return new_leaf

    # monte carlo tree search: recursive function returning value of node from node's perspective
    def __mcts(self, node):
        if node.check_game_over():
            return node.final_reward()
        criterion = node.get_criterion()
        while True:
            # try picking best action
            best_action = None
            for a in range(len(criterion)):
                if node.action_determined_illegal[a]:
                    continue
                if best_action is None or criterion[a] > criterion[best_action]:
                    best_action = a
            if best_action is None:
                raise ValueError('no legal moves found for node:\n{}'.format(node))

            # recurse if possible
            if best_action in node.children:
                v = self.__mcts(node.children[best_action])
                node.update(best_action, -1 * v)
                return -1 * v

            # create leaf, if legal move
            new_leaf = self.__create_leaf(node, best_action)
            if new_leaf is None:
                node.action_determined_illegal[best_action] = True
                continue
            if new_leaf.check_game_over():
                v = new_leaf.final_reward()
            else:
                p, v = self.__evaluate(new_leaf)
                new_leaf.expand(p)
            node.update(best_action, -1 * v)
            return -1 * v

    def __compose_nn_input_planes(self, leaf):
        nn_in = np.empty((self.history_planes * 2 + 3, self.board_size, self.board_size), dtype=bool)
        curr = leaf
        history_plane = 0
        while curr is not None and history_plane <= self.history_planes:
            nn_in[history_plane * 2] = curr.board.to_nn_plane(leaf.color_to_play)
            nn_in[history_plane * 2 + 1] = curr.board.to_nn_plane(leaf.color_to_play.opponent())
            history_plane += 1
            curr = curr.parent
        prior_position_idx = len(self.prior_positions) - 1
        while prior_position_idx >= 0 and history_plane <= self.history_planes:
            nn_in[history_plane * 2] =\
                self.prior_positions[prior_position_idx].to_nn_plane(leaf.color_to_play)
            nn_in[history_plane * 2 + 1] =\
                self.prior_positions[prior_position_idx].to_nn_plane(leaf.color_to_play.opponent())
            history_plane += 1
            prior_position_idx -= 1
        while history_plane <= self.history_planes:
            nn_in[history_plane * 2] = np.zeros((self.board_size, self.board_size), dtype=bool)
            nn_in[history_plane * 2 + 1] = np.zeros((self.board_size, self.board_size), dtype=bool)
            history_plane += 1
        nn_in[self.history_planes * 2 + 2] =\
            np.full((self.board_size, self.board_size), leaf.color_to_play == Color.BLACK, dtype=bool)
        return nn_in

    def __evaluate(self, leaf):
        nn_in = np.expand_dims(self.__compose_nn_input_planes(leaf), 0)
        nn_out = self.nn.feedforward(nn_in)
        p = nn_out[0][0]
        v = nn_out[1][0]
        if p.shape != (self.board_size ** 2 + 1,) or type(v) is not np.float64:
            raise ValueError('neural network output has unexpected type:\np.shape = {}\ntype(v)={}'
                             .format(p.shape, type(v)))
        return p, v

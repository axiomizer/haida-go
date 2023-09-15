from src.game import Color
from enum import Enum


# https://www.red-bean.com/sgf/index.html


class SgfSyntaxError(Exception):
    pass


def parse_node_properties(string):
    properties = {}
    end_ind = -1
    ind = string.index('[')
    while ind != -1:
        prop_ident = string[end_ind + 1:ind].strip()
        prop_values = []
        end_ind = ind - 1
        while ind != -1 and string[end_ind + 1:ind].strip() == '':
            end_ind = string.index(']', ind)
            prop_values.append(string[ind + 1:end_ind].strip())
            ind = string.find('[', end_ind)
        properties[prop_ident] = prop_values
    return properties


class GameResult:
    class Winner(Enum):
        DRAW = 0
        BLACK = 1
        WHITE = 2
        NO_RESULT = 3
        UNKNOWN = 4

    class By(Enum):
        SCORE = 0
        RESIGN = 1
        TIME = 2
        FORFEIT = 3

    def __init__(self, string):
        if string == '0' or string == 'Draw':
            self.winner = GameResult.Winner.DRAW
        elif string[:2] == 'B+':
            self.winner = GameResult.Winner.BLACK
        elif string[:2] == 'W+':
            self.winner = GameResult.Winner.WHITE
        elif string == 'Void':
            self.winner = GameResult.Winner.NO_RESULT
        elif string == '?':
            self.winner = GameResult.Winner.UNKNOWN
        else:
            raise SgfSyntaxError('Failed to parse game result: {}'.format(string))
        self.by = None
        self.margin = None
        if self.winner is GameResult.Winner.BLACK or self.winner is GameResult.Winner.WHITE and len(string) > 2:
            s = string[2:]
            if s == 'R' or s == 'Resign':
                self.by = GameResult.By.RESIGN
            elif s == 'T' or s == 'Time':
                self.by = GameResult.By.TIME
            elif s == 'F' or s == 'Forfeit':
                self.by = GameResult.By.FORFEIT
            else:
                self.by = GameResult.By.SCORE
                try:
                    self.margin = float(s)
                except ValueError:
                    raise SgfSyntaxError('Unexpected string in game result: {}'.format(s))


class RootNode:
    def __init__(self, string):
        properties = parse_node_properties(string)
        if 'GM' not in properties or properties['GM'] != ['1']:
            raise TypeError('Tried to parse an sgf file for a game other than Go')
        self.board_size = int(properties['SZ'][0]) if 'SZ' in properties else 19
        self.ruleset = properties['RU'][0] if 'RU' in properties else None
        self.komi = float(properties['KM'][0])  # assume this property will be present
        self.handicap = int(properties['HA'][0]) if 'HA' in properties else 0
        self.handicap_points = [(ord(p[1]) - ord('a'), ord(p[0]) - ord('a')) for p in properties['AB']]\
            if self.handicap > 0 else []
        self.result = GameResult(properties['RE'][0])


class MoveNode:
    def __init__(self, string):
        properties = parse_node_properties(string)
        if 'B' in properties and 'W' not in properties:
            self.player = Color.BLACK
            move = properties['B'][0]
        elif 'W' in properties and 'B' not in properties:
            self.player = Color.WHITE
            move = properties['W'][0]
        else:
            raise SgfSyntaxError('Expected either a black move or a white move: {}'.format(string))
        if move == '':
            self.move = None  # move is a pass
        elif len(move) != 2 or not all(map(lambda c: ord('a') <= ord(c) <= ord('z'), move)):
            raise SgfSyntaxError('Failed to parse move: {}'.format(move))
        else:
            self.move = (ord(move[1]) - ord('a'), ord(move[0]) - ord('a'))


class SGF:
    def __init__(self, string):
        sequence = self.__main_branch(string)
        nodes = sequence.split(';')[1:]
        self.root = RootNode(nodes[0])
        self.moves = [MoveNode(n) for n in nodes[1:]]

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

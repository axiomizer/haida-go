from src.game import Color
from enum import Enum
from typing import List, Tuple
from itertools import cycle


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

    def __init__(self, winner, by, margin):
        self.winner = winner
        self.by = by
        self.margin = margin

    def to_string(self):
        if self.winner is GameResult.Winner.DRAW:
            return 'Draw'
        if self.winner is GameResult.Winner.NO_RESULT:
            return 'Void'
        if self.winner is GameResult.Winner.UNKNOWN:
            return '?'
        ret = 'B+' if self.winner is GameResult.Winner.BLACK else 'W+'
        if self.by is None:
            return ret
        if self.by is GameResult.By.SCORE:
            return ret + str(self.margin)
        if self.by is GameResult.By.RESIGN:
            return ret + 'R'
        if self.by is GameResult.By.TIME:
            return ret + 'T'
        if self.by is GameResult.By.FORFEIT:
            return ret + 'F'
        return None

    @staticmethod
    def from_string(string):
        if string == '0' or string == 'Draw':
            winner = GameResult.Winner.DRAW
        elif string[:2] == 'B+':
            winner = GameResult.Winner.BLACK
        elif string[:2] == 'W+':
            winner = GameResult.Winner.WHITE
        elif string == 'Void':
            winner = GameResult.Winner.NO_RESULT
        elif string == '?':
            winner = GameResult.Winner.UNKNOWN
        else:
            raise SgfSyntaxError('Failed to parse game result: {}'.format(string))
        by = None
        margin = None
        if winner is GameResult.Winner.BLACK or winner is GameResult.Winner.WHITE and len(string) > 2:
            s = string[2:]
            if s == 'R' or s == 'Resign':
                by = GameResult.By.RESIGN
            elif s == 'T' or s == 'Time':
                by = GameResult.By.TIME
            elif s == 'F' or s == 'Forfeit':
                by = GameResult.By.FORFEIT
            else:
                by = GameResult.By.SCORE
                try:
                    margin = float(s)
                except ValueError:
                    raise SgfSyntaxError('Unexpected string in game result: {}'.format(s))
        return GameResult(winner, by, margin)


class RootNode:
    def __init__(self,
                 board_size: int,
                 ruleset: str,
                 komi: float,
                 handicap: int,
                 handicap_points,
                 result: GameResult):
        self.board_size = board_size
        self.ruleset = ruleset
        self.komi = komi
        self.handicap = handicap
        self.handicap_points = handicap_points
        self.result = result

    def to_string(self):
        ret = ';'
        ret += 'GM[1]'
        ret += 'SZ[{}]'.format(self.board_size)
        ret += 'RU[{}]'.format(self.ruleset) if self.ruleset is not None else ''
        ret += 'KM[{}]'.format(self.komi)
        if self.handicap > 0:
            ret += 'HA[{}]'.format(self.handicap)
            pts = ['[' + chr(h[1] + ord('a')) + chr(h[0] + ord('a')) + ']' for h in self.handicap_points]
            ret += 'AB{}'.format(''.join(pts))
        ret += 'RE[{}]'.format(self.result.to_string())
        return ret

    @staticmethod
    def from_string(string):
        properties = parse_node_properties(string)
        if 'GM' not in properties or properties['GM'] != ['1']:
            raise TypeError('Tried to parse an sgf file for a game other than Go')
        board_size = int(properties['SZ'][0]) if 'SZ' in properties else 19
        ruleset = properties['RU'][0] if 'RU' in properties else None
        komi = float(properties['KM'][0])  # assume this property will be present
        handicap = int(properties['HA'][0]) if 'HA' in properties else 0
        handicap_pts = [(ord(p[1]) - ord('a'), ord(p[0]) - ord('a')) for p in properties['AB']] if handicap > 0 else []
        result = GameResult.from_string(properties['RE'][0]) if 'RE' in properties else None
        return RootNode(board_size, ruleset, komi, handicap, handicap_pts, result)


class MoveNode:
    def __init__(self, player: Color, move):
        self.player = player
        self.move = move  # tuple of int, or None (signifying pass)

    def to_string(self):
        p = 'B' if self.player is Color.BLACK else 'W'
        m = '' if self.move is None else chr(self.move[1] + ord('a')) + chr(self.move[0] + ord('a'))
        return ';{}[{}]'.format(p, m)

    @staticmethod
    def from_string(string):
        properties = parse_node_properties(string)
        if 'B' in properties and 'W' not in properties:
            player = Color.BLACK
            move = properties['B'][0]
        elif 'W' in properties and 'B' not in properties:
            player = Color.WHITE
            move = properties['W'][0]
        else:
            raise SgfSyntaxError('Expected either a black move or a white move: {}'.format(string))
        if move == '':
            position = None  # move is a pass
        elif len(move) != 2 or not all(map(lambda c: ord('a') <= ord(c) <= ord('z'), move)):
            raise SgfSyntaxError('Failed to parse move: {}'.format(move))
        else:
            position = (ord(move[1]) - ord('a'), ord(move[0]) - ord('a'))
        return MoveNode(player, position)


class SGF:
    def __init__(self, root: RootNode, moves: List[MoveNode]):
        self.root = root
        self.moves = moves

    def to_string(self):
        move_nodes = [m.to_string() for m in self.moves]
        return '({}{})'.format(self.root.to_string(), ''.join(move_nodes))

    @staticmethod
    def build(size: int, komi: float, winner: Color, moves: List[Tuple[int, int]]):
        if winner is Color.BLACK:
            result = GameResult(GameResult.Winner.BLACK, None, None)
        elif winner is Color.WHITE:
            result = GameResult(GameResult.Winner.WHITE, None, None)
        else:
            result = GameResult(GameResult.Winner.UNKNOWN, None, None)
        root_node = RootNode(size, 'Chinese', komi, 0, [], result)
        move_nodes = [MoveNode(c, a) for c, a in zip(cycle([Color.BLACK, Color.WHITE]), moves)]
        return SGF(root_node, move_nodes)

    @staticmethod
    def from_string(string):
        sequence = SGF.__main_branch(string)
        nodes = sequence.split(';')[1:]
        return SGF(RootNode.from_string(nodes[0]), [MoveNode.from_string(n) for n in nodes[1:]])

    @staticmethod
    def __main_branch(data):
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
                return data[:start] + SGF.__main_branch(data[start+1:i])
        raise ValueError('Unbalanced parentheses')

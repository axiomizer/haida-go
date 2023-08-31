import numpy as np
from enum import Enum
import itertools


class Color(Enum):
    BLACK = 0
    WHITE = 1

    def opponent(self):
        if self is self.BLACK:
            return self.WHITE
        else:
            return self.BLACK


class IllegalMove(Exception):
    pass


class Board:
    def __init__(self, size):
        self.size = size
        self.unique_group_id = 1
        # 0 means no stone; positive is a black group; negative is a white group
        self.board = np.zeros((size, size), dtype=int)

    def __eq__(self, other):
        if isinstance(other, Board):
            return self.board.shape == other.board.shape and (np.sign(self.board) == np.sign(other.board)).all()
        else:
            return False

    def __hash__(self):
        return hash(np.sign(self.board))

    def copy(self):
        c = Board(self.size)
        c.unique_group_id = self.unique_group_id
        c.board = np.copy(self.board)
        return c

    def __get_new_grp_id(self, color: Color):
        grp_id = self.unique_group_id
        self.unique_group_id += 1
        if color is Color.BLACK:
            return grp_id
        else:
            return grp_id * -1

    def __adjacent_positions(self, position):
        adj = []
        if position[0] > 0:
            adj.append((position[0] - 1, position[1]))
        if position[0] < self.size - 1:
            adj.append((position[0] + 1, position[1]))
        if position[1] > 0:
            adj.append((position[0], position[1] - 1))
        if position[1] < self.size - 1:
            adj.append((position[0], position[1] + 1))
        return adj

    def __find_liberties(self, group: int):
        liberties = set()
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] != group:
                    continue
                liberties.update([adj for adj in self.__adjacent_positions((x, y)) if self.board[adj] == 0])
        return liberties

    # set all stones in the given groups to a new value
    def __set_all_grps(self, groups: set[int], new_val: int):
        if len(groups) == 0:
            return
        for i, j in itertools.product(range(self.size), range(self.size)):
            if self.board[i, j] in groups:
                self.board[i, j] = new_val

    def to_nn_plane(self, color: Color):
        if color is Color.BLACK:
            return self.board > 0
        else:
            return self.board < 0

    def place_stone(self, position: tuple[int, int], color: Color):
        if position[0] < 0 or position[0] >= self.size or position[1] < 0 or position[1] >= self.size:
            raise IllegalMove('position {} out of bounds'.format(position))
        if self.board[position] != 0:
            raise IllegalMove('position {} already occupied'.format(position))

        current_player_has_liberty = False
        adj_groups = []
        for adj in self.__adjacent_positions(position):
            if self.board[adj] == 0:
                current_player_has_liberty = True
            else:
                adj_groups.append(self.board[adj])
        adj_group_lib = [len(self.__find_liberties(g)) > 1 for g in adj_groups]

        captures = set()  # adjacent opponent groups that will run out of liberties
        merge = set()  # adjacent groups belonging to the current player that will be merged
        for g, l, in zip(adj_groups, adj_group_lib):
            if (g > 0) != (color is Color.BLACK):  # opponent group
                if not l:
                    captures.add(g)
            else:  # current player's group
                merge.add(g)
                if l:
                    current_player_has_liberty = True
        if len(captures) == 0 and not current_player_has_liberty:
            raise IllegalMove('suicidal move at {}'.format(position))

        self.__set_all_grps(captures, 0)
        self.board[position] = self.__get_new_grp_id(color)
        self.__set_all_grps(merge, self.board[position])

    def score(self, komi):
        b = np.sign(self.board)
        dame = np.zeros((self.size, self.size), dtype=bool)
        for i, j in itertools.product(range(self.size), range(self.size)):
            if b[i, j] == 0 and not dame[i, j]:
                visited = {(i, j)}
                colors = self.__explore_region(visited, b, (i, j))
                if len(colors) == 0 or len(colors) == 2:
                    for v in visited:
                        dame[v] = True
                elif Color.BLACK in colors:
                    for v in visited:
                        b[v] = 1
                else:
                    for v in visited:
                        b[v] = -1
        b_score = np.count_nonzero(b == 1)
        w_score = np.count_nonzero(b == -1) + komi
        return b_score, w_score

    # used for scoring
    def __explore_region(self, visited, b, position):
        colors = set()
        for adj in self.__adjacent_positions(position):
            if adj in visited:
                continue
            if b[adj] == 1:
                colors.add(Color.BLACK)
            elif b[adj] == -1:
                colors.add(Color.WHITE)
            elif b[adj] == 0:
                visited.add(adj)
                colors.update(self.__explore_region(visited, b, adj))
        return colors

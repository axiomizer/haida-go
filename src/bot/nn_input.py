from src.game import Color, Board
import numpy as np


def compose(nn_in_prev, board: Board, hp):
    if nn_in_prev is None or not nn_in_prev[hp + 2][0][0]:
        color_to_play = Color.BLACK
    else:
        color_to_play = Color.WHITE
    nn_in = np.empty((hp + 3, board.size, board.size), dtype=bool)
    nn_in[0] = to_nn_plane(board, color_to_play)
    nn_in[1] = to_nn_plane(board, color_to_play.opponent())
    if nn_in_prev is None:
        nn_in[2:2 + hp] = np.zeros((hp, board.size, board.size), dtype=bool)
    else:
        nn_in[2:2 + hp:2] = nn_in_prev[1:hp:2]
        nn_in[3:2 + hp:2] = nn_in_prev[0:hp:2]
    nn_in[hp + 2] = np.full((board.size, board.size), color_to_play == Color.BLACK, dtype=bool)
    return nn_in


def to_nn_plane(board: Board, color: Color):
    if color is Color.BLACK:
        return board.board > 0
    else:
        return board.board < 0

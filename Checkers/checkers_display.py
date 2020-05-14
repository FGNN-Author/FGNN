from bqplot import pyplot as plt
from bqplot import Axis
from bqplot import Figure
from bqplot import LinearScale
from bqplot import Scatter
from bqplot import ColorScale

import numpy as np

from checkers_helpers import *

class BoardDisplay():

    def __init__(self, canInteract=True):

        # Setup & Axis stuff...
        x_sc = LinearScale(min=0, max=8)
        y_sc = LinearScale(min=0, max=8)
        y_sc.reverse = True

        x_ax = Axis(label='X', scale=x_sc)
        y_ax = Axis(label='Y', scale=y_sc, orientation='vertical')

        # Display starting position for checkers game.

        # Colour checkerboard... Extra stuff for alignment to grid.
        vals = np.zeros((8,8))
        vals[::2,::2] = -1
        vals[1::2, 1::2] = -1

        col_sc = ColorScale(colors=['white', 'lightgray'])
        bg = plt.gridheatmap(vals, scales={'column': x_sc, 'row': y_sc, 'color': col_sc})
        bg.row = np.arange(8)
        bg.column = np.arange(8)

        self.bg = bg

        # Foreground...
        # colors of pieces
        col_sc = ColorScale(colors=['firebrick', 'black'])

        # Create empty scatter grid.
        fg = Scatter(x=[], y=[])
        fg.scales = {'x': x_sc, 'y': y_sc, 'color': col_sc}
        fg.color  = []
        fg.default_size = 550
        fg.enable_move = canInteract
        print(fg.drag_size)
        fg.drag_size = 0.1
        print(fg.drag_size)

        self.fg = fg

        fig = Figure(marks=[bg, fg], axes=[x_ax, y_ax])

        # Force square.
        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1

        # display(fig)
        self.fig = fig

    def display(self):
        display(self.fig)

    # def _matToBoard(x,y):
    #     return x*2 + ((y+1)%2), y

    # Given an 8*8 mat representing the board, return a list of pieces and their colours (0 or 1)
    def _piecesFromPos(position):
        # Easier later if each piece is returned separated into separate lists.
        p_x = []
        p_y = []
        p_col = []
        p_isKing = []

        for x in range(4):
            for y in range(8):
                v = position[y,x]
                if v != 0:
                    cx, cy = matToBoard(x,y)
                    p_x.append(cx)
                    p_y.append(cy)

                    p_col.append(1 if v > 0 else 0)
                    p_isKing.append(abs(v) > 1)

        return p_x, p_y, p_col, p_isKing


    def update(self, position):
        # TODO: Display kings differently.
        x,y,cols,isKing = BoardDisplay._piecesFromPos(position)
        y = np.array(y) + 0.5
        x = np.array(x) + 0.5

        self.fg.x = x
        self.fg.y = y
        self.fg.color = cols


# Display the raw matrix that'd be used as training input.
class SimpleSquareBoardDisplay():
    def __init__(self):
        vals = np.zeros((8,8))

        x_sc = LinearScale(min=0, max=8)
        y_sc = LinearScale(min=0, max=8)
        y_sc.reverse = True

        x_sc.allow_padding = False
        y_sc.allow_padding = False

        # -1 are the empty squares (not in output from network)
        col_sc = ColorScale(min=-1, max=1, colors=['red', 'white', 'black'])

        x_ax = Axis(label='X', scale=x_sc)
        y_ax = Axis(label='Y', scale=y_sc, orientation='vertical')

        bg = plt.gridheatmap(vals, scales={'column': x_sc, 'row': y_sc, 'color': col_sc})

        self.board = bg

        fig = Figure(marks=[bg], axes=[x_ax, y_ax])
        fig.min_aspect_ratio = 1  # Idfk why this makes it have an aspect ratio around 0.5 but w/e
        fig.max_aspect_ratio = 1

        self.fig = fig

    def update(self, pos):
        self.board.color = pos

# Display the raw matrix that'd be used as training input.
class SimpleBoardDisplay():
    def __init__(self):
        vals = getDefaultPosition()

        x_sc = LinearScale(min=0, max=4)
        y_sc = LinearScale(min=0, max=8)
        y_sc.reverse = True

        x_sc.allow_padding = False
        y_sc.allow_padding = False

        # -1 are the empty squares (not in output from network)
        col_sc = ColorScale(min=-1, max=1, colors=['red', 'white', 'black'])

        x_ax = Axis(label='X', scale=x_sc)
        y_ax = Axis(label='Y', scale=y_sc, orientation='vertical')

        bg = plt.gridheatmap(vals, scales={'column': x_sc, 'row': y_sc, 'color': col_sc})

        self.board = bg

        fig = Figure(marks=[bg], axes=[x_ax, y_ax])
        fig.min_aspect_ratio = 0.63  # Idfk why this makes it have an aspect ratio around 0.5 but w/e
        fig.max_aspect_ratio = 0.63

        self.fig = fig

    def update(self, pos):
        self.board.color = pos

# Display one-hot move encoding (output vector)
class MoveHeatmapDisplay():
    def __init__(self):
        # Setup & Axis stuff...
        x_sc = LinearScale(min=0, max=8)
        y_sc = LinearScale(min=0, max=8)

        x_ax = Axis(label='X', scale=x_sc)
        y_ax = Axis(label='Y', scale=y_sc, orientation='vertical')
        y_sc.reverse = True

        col_sc = ColorScale(min=-1, max=1, colors=['white', 'blue', 'red'])
        bg = plt.gridheatmap(np.zeros((16,16)), scales={'column': x_sc, 'row': y_sc, 'color': col_sc})
        bg.row = np.arange(16)/2
        bg.column = np.arange(16)/2

        self.board = bg

        fig = Figure(marks=[bg], axes=[x_ax, y_ax])

        # Force square.
        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1

        # Give random data
        self.update(np.random.random((8,4,4)))

        self.fig = fig


    def update(self, raw_moves): # raw_moves should be (8,4,4)
        b = np.reshape(raw_moves, (8,4,2,2))

        out = np.zeros((16,16)) - 1

        for x in range(4):
            for y in range(8):
                bx,by = matToBoard(x,y)
                out[by*2:by*2+2, bx*2:bx*2+2] = b[y,x].T

        self.board.color = out

    # Display
    def updateMove(self, st, end):
        raw_moves = oneHotMove(st, end)
        self.update(raw_moves)

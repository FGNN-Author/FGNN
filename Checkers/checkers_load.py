import re
import numpy as np
from bqplot import pyplot as plt

from checkers_helpers import *

def load_games(file_name="22k_checkers.txt"):
    with open(file_name) as f:
        blah = f.read()

        # Games separated by two newlines.
        games = blah.split("\n\n")
    return [CheckersGame(g) for g in games]

class CheckersGame:
    # Parse a game from a string description.
    def __init__(self, gameStr):
        # Drop metadata lines. (which start with '[')
        moveLines = [line for line in gameStr.split("\n")
                    if len(line) > 0 and line[0] != '[' ]

        # Single string containing all moves, space separated.
        movesStr = " ".join(moveLines)

        # Drop everything between pairs of {}, these are game notes
        movesStr = re.sub("[\{].*?[\}]", "", movesStr)

        # Drop last word - it's the result of the game,
        # and also any word with a '.', as that's the move number.
        moves = [m for m in movesStr.split(" ")[:-1]
                 if not "." in m and m is not ""]

        # Convert moves as either 'a-b' or 'axbxc' to a list.
        self.moves = [CheckersGame.parseMove(m) for m in moves]

    def __repr__(self):
        return "GAME:" + str(self.moves) + "\n\n"

    # Store as tuple (doesCapture, [squares])
    def parseMove(m):
        if 'x' in m:
            doesCapture = True;
            squares = m.split('x')
        else:
            doesCapture = False;
            squares = m.split('-')

        # Should error if there's anything else.
        squares = [int(s) for s in squares]

        return (doesCapture, squares)


if __name__ == "__main__":
    print(load_games())

# Generator for learning
def dataGenerator(games):

    for g_ix, g in enumerate(games):
        # print("Loading game", g_ix)
        pos = getDefaultPosition()

        # Go through each move
        for mov_ix, m in enumerate(g.moves):
            doesC, sqrs = m

            # In a multihop sequence, give every position separately to training.
            for st, end in zip(sqrs, sqrs[1:]):
                x_i = np.copy(pos)

                y_i = oneHotMove(st, end)

                # Positive are always current piece to play
                if mov_ix % 2 == 1:
                    x_i = -x_i
                    # Flip both so current player always plays upwards.
                    # I can only hope this is right lol
                    x_i = np.flip(x_i, axis=0)
                    x_i = np.flip(x_i, axis=1)
                    y_i = oneHotMove(33-st, 33-end)

                else:
                    y_i = oneHotMove(st, end)

                yield x_i, y_i

                # After, we can apply move.
                makeSubMove(doesC, st, end, pos)

# Infinite generator of batches.
def makeBatches(batch_s=64, games=None):
    if games is None:
        games = load_games()

    while True:
        dg = dataGenerator(games)

        ix = 0
        X = np.zeros((batch_s, 8, 4, 1))
        Y = np.zeros((batch_s, 128))
        for x_i, y_i in dg:

            x = np.expand_dims(x_i, axis=2) # Cause conv2d
            y = y_i.flatten()

            #print(X.shape, x.shape, X[ix].shape)
            X[ix] = x
            Y[ix] = y

            ix += 1
            if ix >= batch_s:
                ix = 0
                yield X, Y

# Convert games into single numpy array of input & output
def getData(games):
    moves = [a for a in dataGenerator(games)]

    # Insert em into matrices
    set_dim = len(moves)
    X = np.zeros((set_dim, 8, 4, 1))
    Y = np.zeros((set_dim, 128))

    print("Size of input is:", (X.nbytes + Y.nbytes)/1000000, "MB.")

    for ix, (x, y) in enumerate(moves):
        x = np.expand_dims(x, axis=2) # Cause conv2d
        y = y.flatten()

        X[ix] = x
        Y[ix] = y

    return X, Y

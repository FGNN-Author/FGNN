import numpy as np

# (8,4) matrix to board (8,8) coordinates
def matToBoard(x,y):
    return x*2 + ((y+1)%2), y

# Board (8,8) coordinates to (8,4) mat coords
def boardToMat(x,y):
    return x//2, y

# PEN index to matrix coordinates.
def ixToMatPos(i):
    i -= 1
    return i % 4, i // 4  # x (0-3), y (0-7)

# Convert x,y coord (x,y both 0-7) to PEN index
def coordToIx(row, col):
    return row*4 + col + (row+1)%2

# Create pieces! (Setup initial pos)
def getDefaultPosition():
    position = np.zeros((8,4))

    position[:3]  = 1  # First 3 rows are black
    position[-3:] = -1 # Last 3 red (or whatever)

    return position

# Generate the heatmap for a given start and end square.
def oneHotMove(st, end):
    # Positions in the matrix
    mx1, my1 = ixToMatPos(st)
    mx2, my2 = ixToMatPos(end)

    bx1, by1 = matToBoard(mx1, my1)
    bx2, by2 = matToBoard(mx2, my2)

    # Vector
    vx, vy = np.sign(bx2 - bx1), np.sign(by2 - by1)

    # MUST be symmetrical for symmnet...
    # 0 | 2
    #---|---
    # 1 | 3

    # Index within subarray to put it in.
    ix = (vx + 1) + (vy + 1)//2

    raw_moves = np.zeros((8,4,4))

    # Onehot the one move.
    raw_moves[my1, mx1, ix] = 1

    return raw_moves

# Makes move in place in the given matrix
def makeMove(move, pos):

    doesCapture, sqrs = g_mvs[i]

    # Create pairs of moves in list.
    # E.g. axbxc -> [(a,b), (b,c)]
    for st, end in zip(sqrs, sqrs[1:]):
        makeSubMove(doesCapture, st, end, pos)

# Make a single hop in place in a given matrix
def makeSubMove(doesCapture, st, end, pos):
    x1,y1 = ixToMatPos(st)
    x2,y2 = ixToMatPos(end)
    # print("from", st, (x1, y1), "to", end, (x2, y2))

    # Remove counter in middle if captured.
    if doesCapture:
        # Find middle of both points in board coordinates.
        bx1, by1 = matToBoard(x1, y1)
        bx2, by2 = matToBoard(x2, y2)
        mx, my = (bx1 + bx2) // 2, (by1 + by2) // 2
        # Convert back to mat coords
        mx, my = boardToMat(mx, my)
        pos[my, mx] = 0

    # Move piece
    piece = pos[y1, x1]
    pos[y2, x2] = piece
    pos[y1, x1] = 0

    # If it's a regular piece and it moved into top or bottom row make it a king.
    # TODO: Display & test
    if abs(piece) == 1 and (x2 == 0 or x2 == 7):
        pos[y2, x2] = 3*piece

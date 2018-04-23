import numpy as np
import scipy as sp
import scipy.signal
import scipy.ndimage.morphology as morph



class MineSweeper(object):
    def __init__(self, width=10, height=10, mines=0.2):
        """
        width: int, width of the board
        height: int, height of the board
        mines: float or int, proportion of mines, or absolute number of mines
        """
        self._initialized = False
        self._width = width
        self._height = height

        if isinstance(mines, float):
            self._nmines = round(width * height * mines)
        else:
            self._nmines = mines

        if self._nmines >= width * height:
            raise ValueError("Can't put that many mines")

        self._mine_board = np.zeros((height, width), dtype=np.bool)
        self._flag_board = np.zeros((height, width), dtype=np.bool)
        self._hint_board = np.zeros((height, width), dtype=np.int8)
        self._open_board = np.zeros((height, width), dtype=np.bool)



    def _initialize(self):
        self._open_board[:, :] = False
        self._mine_board[:, :] = False
        self._flag_board[:, :] = False

        self._mine_board.flat[:self._nmines] = True
        np.random.shuffle(self._mine_board.flat)

        mask = np.ones((3, 3), dtype=np.int8)
        self._hint_board = sp.signal.convolve2d(self._mine_board, mask, mode='same')

        self._initialized = True



    @property
    def finished(self):
        return self.won or self.lost

    @property
    def won(self):
        return np.all(np.logical_xor(self._open_board, self._mine_board))

    @property
    def lost(self):
        return np.any(np.logical_and(self._open_board, self._mine_board))

    @property
    def open(self):
        """Board of opened cells"""
        return self._open_board

    @property
    def hints(self):
        """
        Board of known hints.
        0 is either a known 0 or an unopened cell or an opened cell with a mine.
        """
        return self._hint_board * self._open_board * (1 - self._mine_board)

    @property
    def board(self):
        """
        Full representation of the board.
        0 is an opened cell without mines.
        -1 is a closed cell.
        -2 is a flag.
        -3 is an opened cell with a mine.
        """
        closed = 1 - self._open_board
        openmine = self._mine_board * self._open_board
        return self.hints - closed - self._flag_board - 3 * openmine



    def __str__(self):
        s = "+" + "-" * self._width + "+\n"
        for row in self.board:
            s += "|"
            for cell in row:
                if cell == -1:
                    s += "#"
                elif cell == 0:
                    s += " "
                elif cell > 0:
                    s += str(cell)
                elif cell == -2:
                    s += "F"
                elif cell == -3:
                    s += "O"
                else:
                    raise ValueError("Unexpected value in board: %d" % cell)

            s += "|\n"
        s += "+" + "-" * self._width + "+"

        return s



    def restart(self):
        self._open_board[:, :] = False
        self._mine_board[:, :] = False
        self._flag_board[:, :] = False
        self._hint_board[:, :] = 0
        self._initialized = False



    def click(self, x, y):
        if x not in range(self._width):
            raise ValueError("Value for %d for x is out of the board" % x)
        if y not in range(self._height):
            raise ValueError("Value for %d for y is out of the board" % y)

        if not self._initialized:
            self._initialize()
            while self._mine_board[y, x]:
                self._initialize()

        if self.finished:
            raise ValueError("Can't play a game already finished")

        if self._open_board[y, x]:
            raise ValueError("Cell at %d, %d is already open" % (x, y))

        if self._flag_board[y, x]:
            raise ValueError("Can't open flagged cell at %d, %d" % (x, y))

        self._open_board[y, x] = True

        if self.lost:
            return

        if self._hint_board[y, x] == 0:
            # Only the zeros are to be filled
            mask_fillable = (self._hint_board == 0)

            # We start from cell at x, y and propagate from there
            seed = np.zeros_like(mask_fillable)
            kern = np.ones((3, 3), dtype=np.bool)
            seed[y, x] = True
            mask = morph.binary_propagation(seed, structure=kern, mask=mask_fillable)

            # Propagate once more to open the adjacent hints
            mask = morph.binary_dilation(mask, structure=kern)

            self._open_board |= mask

        if self.won:
            self._flag_board[:, :] = self._mine_board[:, :]



    def flag(self, x, y):
        if x not in range(self._width):
            raise ValueError("Value for %d for x is out of the board" % x)
        if y not in range(self._height):
            raise ValueError("Value for %d for y is out of the board" % y)

        if not self._initialized:
            self._initialize()
            while self._mine_board[y, x]:
                self._initialize()

        self._flag_board[y, x] ^= 1

        if self.finished:
            raise ValueError("Can't play a game already finished")

        if self._open_board[y, x]:
            raise ValueError("Can't flag an open cell at %d, %d" % (x, y))

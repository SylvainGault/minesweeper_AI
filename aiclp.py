import sys
import numpy as np
import scipy.ndimage.morphology as morph
import clpfd



def printover(s):
    sys.stdout.write("\r" + " " * printover.lastlen + "\r")
    sys.stdout.write(s)
    sys.stdout.flush()
    printover.lastlen = len(s)

printover.lastlen = 0



class AI(object):
    def __init__(self):
        self.width = None
        self.height = None
        self.lastmove = None
        self.known_mines = None



    def new_game(self, width, height):
        self.width = width
        self.height = height
        self.lastmove = (np.random.randint(width), np.random.randint(height))
        self.known_mines = np.zeros((height, width), np.bool)



    def _hint_constraints(self, varmines, board):
        """
        Build and return a matrix of expressions that represents the number of
        mines around each cell.
        varmines: Matrix of clpfd Variables representing the presence of mine
        in each cell.
        board: Matrix of booleans integers representing the board.
        """

        h = self.height
        w = self.width
        closeboard = (board < 0)

        # Add a padding around to simplify the summation just below
        summines = np.empty((h + 2, w + 2), np.object).view(clpfd.Variables)
        summines[:, :] = 0

        for a in [-1, 0, 1]:
            for b in [-1, 0, 1]:
                summines[1+a:h+1+a, 1+b:w+1+b] += varmines

        # Remove the padding that was added just to make the sums
        summines = summines[1:-1, 1:-1]

        # Set the constraint that the sum of mines is equal to the hint
        hintsconst = (summines == board)
        hintsconst[closeboard] = None
        return hintsconst



    def _check_coords(self, openboard):
        """
        Return a matrix of cells to check and the list of their coordinate in
        the order in which they should be checked.
        """
        # Cells to check for mines
        kern = np.ones((3, 3), dtype=np.bool)
        checkboard = morph.binary_dilation(openboard, structure=kern)
        checkboard = np.logical_xor(checkboard, openboard)
        checkcoords = np.argwhere(checkboard)

        # Check the cells in order from the closest to the farthest from the
        # last move. That's where the interesting stuff is more likely to have
        # happened.
        np.random.shuffle(checkcoords)
        dist = checkcoords - np.array([self.lastmove[1], self.lastmove[0]])
        dist = dist.max(axis=1)
        checkcoords = checkcoords[dist.argsort()]

        return checkboard, checkcoords



    def next_move(self, board):
        h = board.shape[0]
        w = board.shape[1]
        openboard = (board >= 0)

        varmines = clpfd.Variables(board.shape, range(2), "mines")
        hintsconst = self._hint_constraints(varmines, board)

        # Where there is a hint, there is no mine
        nomineconst = (varmines[openboard] == 0)

        checkboard, checkcoords = self._check_coords(openboard)

        for c in checkcoords:
            y, x = c

            # check that we don't already know there's a mine there
            if self.known_mines[y, x]:
                continue

            solver = clpfd.solver()
            solver.add_constraint(hintsconst)
            solver.add_constraint(nomineconst)
            solver.add_constraint(varmines[self.known_mines == True] == 1)
            solver.add_constraint(varmines[y, x] == 1)

            printover("checking if %d, %d is free" % (x, y))
            sol = solver.solve()
            if sol.status == 'Infeasible':
                self.lastmove = x, y
                printover("")
                return x, y

            if sol.status == 'Optimal':
                # There exist a solution that might put a mine there.
                # Make sure there's necessarily one and mark it.
                solver = clpfd.solver()
                solver.add_constraint(hintsconst)
                solver.add_constraint(nomineconst)
                solver.add_constraint(varmines[self.known_mines == True] == 1)
                solver.add_constraint(varmines[y, x] == 0)
                printover("checking if %d, %d is a mine" % (x, y))
                if solver.solve().status == 'Infeasible':
                    self.known_mines[y, x] = True


        # Just choose a random cell
        # But preferably one far away
        closeboard = (board < 0)
        randboard = np.logical_xor(closeboard, checkboard)
        randcoord = np.argwhere(randboard)

        if randcoord.shape[0] == 0:
            randcoord = np.argwhere(closeboard)

        y, x = randcoord[np.random.randint(randcoord.shape[0])]
        self.lastmove = x, y
        printover("")
        return x, y

"""
This file defines a few Board subclasses. Boards are gamestate objects, that
store information about the board, check for winners, and provide next boards

It also contains Parsers, which are closely linked to boards, and are
responsible for drawing boards and announcing the winner of a game
"""

import curses
from curses import wrapper
from copy import deepcopy

# TODO: rename board internal data structrue to data

class Board(object):
    """Base Board class. Subclasses must define the following methods:
     - new_board(self) -> initial game state
     - check_winner(self) -> -1, 1, 0 (draw), or None
     - validate_move(self, player, *move_info) -> True/False
     - move(self, player, *move_info) -> new Board instance with updated state
     - get_next_boards(self, player)
    """

    def __init__(self, board = None):
        if board is None:
            board = self.new_board()

        self.board = board
        self.winner = self.check_winner()

class TTTBoard(Board):
    """Board class for a Tic-Tac-Toe game."""

    def __init__(self, board = None):
        super().__init__(board)

    def new_board(self):
        return [[0, 0, 0],
                [0, 0, 0],
                [0, 0 ,0]]

    def check_winner(self):
        threes = self.get_all_threes()

        for three in threes:
            Xs = three.count(1)
            Os = three.count(-1)

            if Xs == 3:
                return 1
            elif Os == 3:
                return -1

        if self.full():
            return 0

        return None

    def validate_move(self, player, row, column):
        return self.board[row][column] == 0

    def move(self, player, row, column):
        new_board = deepcopy(self.board)

        new_board[row][column] = player

        return TTTBoard(new_board)

    def get_next_boards(self, player):
        next_boards = []

        for row in range(0, 3):
            for column in range(0,3):
                if self.board[row][column] == 0:
                    next_boards.append(self.move(player, row, column))

        return next_boards

    def full(self):
        """Checks if the board is full by counting the number of blanks"""

        blanks = sum(row.count(0) for row in self.board)

        return blanks == 0

    def get_all_threes(self):
        """Retreivies all possible three-in-a-row combos"""

        return self.get_rows() + self.get_columns() + self.get_diagonals()

    def get_rows(self):
        return self.board

    def get_columns(self):
        columns = []

        for i in range(0, 3):
            columns.append([row[i] for row in self.board])

        return columns

    def get_diagonals(self):
        diagonals = []
        b = self.board

        diagonals.append([b[0][0], b[1][1], b[2][2]])
        diagonals.append([b[2][0], b[1][1], b[0][2]])

        return diagonals

class Parser(object):
    """
    Object that interprets the Board class data.

    Subclasses must define:
     - draw_board(self, board) -> None (print/display board's)
     - announce_winner(self, winner) -> None (message stating winner)
    """

class TTTParser(Parser):
    def __init__(self):
        def colors(*unusued):
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_RED)

        wrapper(colors)

    def draw_board(self, board):
        wrapper(self.draw_complete, board)

    def draw_complete(self, stdscr, board):
        stdscr.clear()
        curses.curs_set(0) # hide the cursor

        self.draw_frame(stdscr)

        for i, row in enumerate(board):
            for j, val in enumerate(row):
                self.draw_token(stdscr, board[i][j], i, j)

        stdscr.refresh()

    def draw_frame(self, stdscr):
        stdscr.addstr(1, 1, "   |   |   ")
        stdscr.addstr(2, 1, "---+---+---")
        stdscr.addstr(3, 1, "   |   |   ")
        stdscr.addstr(4, 1, "---+---+---")
        stdscr.addstr(5, 1, "   |   |   ")

    def draw_token(self, stdscr, num, row, column, color = 0):
        stdscr.addstr(1 + 2*row, 2 + 4*column, self.decode(num), curses.color_pair(color))

    def announce_winner(self, winner):
        wrapper(self.announce_winner_wrapper, winner)

    def announce_winner_wrapper(self, stdscr, winner):
        stdscr.addstr('\n\n')

        if winner == 1:
            stdscr.addstr(' X wins!')
        elif winner == -1:
            stdscr.addstr(' O wins!')
        else:
            stdscr.addstr(' It\'s a draw!')

        stdscr.addstr('\n Press Enter')
        stdscr.getkey()

    def decode(self, num):
        if num == 1:
            return 'X'
        elif num == -1:
            return 'O'
        elif num == 0:
            return ' '
        else:
            return '?'

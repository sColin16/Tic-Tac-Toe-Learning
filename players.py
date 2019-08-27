"""
This file defines all the players (objects that accept a game state as input,
and output a corresponding board) that interact with the GameDelegate to
simulate the game.
"""
import curses
import re
import random
from curses import wrapper
from math import inf

from mappers import FeatureExtractor

# TODO: implement a Look up player
# TODO: extract HumanPlayer class

class Player(object):
    """Base class that defines all players.

    Arguments:
     - player (1 or -1): determines if it's player is player 1 or player 2

    Subclasses must define:
     - get_move(self, board, board_parser) -> new Board instance
    """

    def __init__(self, player):
        self.player = player

class RandomPlayer(Player):
    """Player that selects and returns a random valid move

    Arguments
     - seed (any number): seed for random number generator for reproducibility
    """

    def __init__(self, player, seed = None):
        super().__init__(player)
        self.generator = random.Random()
        self.generator.seed(seed)

    def get_move(self, board, *args):
        next_boards = board.get_next_boards(self.player)
        choice = self.generator.choice(next_boards)

        return choice

class TrainablePlayer(Player):
    """
    A computer player that sometimes plays randomly to help learning agents
    explore more of the game space.

    Arguments:
     - epsilon: (number 0-1) Chance the player chooses a random move

    Subclasses must define:
     - smart_move(self, board) -> new Board subclass instance
    """

    def __init__(self, player, epsilon = 0.2, seed = None):
        super().__init__(player)

        self.random_player = RandomPlayer(player, seed)
        self.epsilon = epsilon

    def get_move(self, board, *unused):
        if random.random() < self.epsilon:
            return self.random_player.get_move(board)

        else:
            return self.smart_move(board)

# TODO: only recurisvely call the top n high scoring boards
# TODO: return a list of all boards that are the best

class LookAheadPlayer(TrainablePlayer):
    """
    Uses minimax, a heuristic function, and an optional board feature extractor
    to determine the next best move.
    """

    def __init__(self, player, depth, heuristic_model, feature_extractor = FeatureExtractor(), epsilon = 0.2, seed = None):
        super().__init__(player, epsilon, seed)

        self.heuristic_model = heuristic_model
        self.depth = depth
        self.feature_extractor = feature_extractor

    def smart_move(self, board):
        _, best_board = self.minimax(board, self.depth, -inf, inf)

        return best_board

    def minimax(self, board, depth, alpha, beta, maximizer = True):
        """
        Returns a tuple of the best score and the best next board.
        """

        if depth == 0 or board.winner is not None:
            return (self.get_heuristic(board), board)

        elif maximizer:
            best_score = -inf
            best_board = None

            for next_board in board.get_next_boards(self.player):
                score, _ = self.minimax(next_board, depth - 1, alpha, beta, False)

                if score > best_score:
                    best_board = next_board

                best_score = max(best_score, score)
                alpha = max(alpha, best_score)

                if alpha > beta:
                    break

            return (best_score, best_board)

        elif not maximizer:
            best_score = inf

            for next_board in board.get_next_boards(self.player * -1):
                score, _ = self.minimax(next_board, depth - 1, alpha, beta, True)
                best_score = min(best_score, score)
                beta = min(beta, best_score)

                if alpha > beta:
                    break

            # No need to return best_board, only upper level (maximizer) call
            # needs that
            return (best_score, None)

    def get_heuristic(self, board):
        features = self.feature_extractor.extract(board)

        return self.heuristic_model.score(features)

"""----------------
Specialized Players
----------------"""

class HumanTTTPlayer(Player):
    """
    Terminal-based player for Tic-Tac-Toe using the curses library
    """

    def __init__(self, player):
        super().__init__(player)

    def get_move(self, board, board_parser):
        row, column = self.get_first_open(board.board)
        submit = False

        while True:
            row, column, submit = wrapper(self.get_input, board, board_parser, row, column)

            if submit:
                return board.move(self.player, row, column)

    def get_input(self, stdscr, board, board_parser, row, column):
        submit = False

        board_parser.draw_complete(stdscr, board.board)
        self.draw_selection(stdscr, board, row, column, board_parser)

        c = stdscr.getkey()

        if c == '\n' and board.validate_move(self.player, row, column):
            submit = True

        elif c =='KEY_UP':
            row = (row - 1) % 3

        elif c == 'KEY_DOWN':
            row = (row + 1) % 3

        elif c == 'KEY_LEFT':
            column = (column - 1) % 3

        elif c == 'KEY_RIGHT':
            column = (column + 1) % 3

        return (row, column, submit)

    def draw_selection(self, stdscr, board, row, column, board_parser):
        valid = board.validate_move(self.player, row, column)

        if valid:
            token = self.player
            color = 1
        else:
            token = board.board[row][column]
            color = 2

        board_parser.draw_token(stdscr, token, row, column, color)

    def get_first_open(self, board):
        for i, row in enumerate(board):
            for j, num in enumerate(row):
                if num == 0:
                    return (i, j)

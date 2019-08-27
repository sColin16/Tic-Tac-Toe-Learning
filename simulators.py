"""
This file contains classes that simulate gameplay:
 - GameDelegate: simulates a game between any two players
 - Trainer (LinearPlayer only): Simulates training games
"""

import numpy as np
import time
import tqdm

# TODO: make trainer epsilon value slowly go down over time
# TODO: change printing parameters at end to better design
# TODO: allow user to set verbosity for the trainer

class GameDelegate(object):
    """Runs the game between two players, and records the history of the game"""

    def __init__(self, player1, player2, board_callable, board_parser,
            verbosity = 1):
        self.players = [player1, player2]
        self.board_callable = board_callable
        self.board = board_callable()
        self.history = [self.board]
        self.board_parser = board_parser
        self.verbosity = verbosity

    def reset(self):
        self.board = self.board_callable()
        self.history = [self.board]

    def run_game(self):
        self.reset()

        turn = 0

        while self.board.winner is None:
            new_board = self.players[turn].\
                get_move(self.board, self.board_parser)

            self.board = new_board
            self.history.append(new_board)

            if self.verbosity >= 2:
                self.board_parser.draw_board(self.board.board)

            if turn == 0:
                turn = -1
            else:
                turn = 0

        if self.verbosity >= 1:
            self.board_parser.draw_board(self.board.board)
            self.board_parser.announce_winner(self.board.winner)

    def play_games(self, number):
        for i in range(number):
            self.run_game()

class Trainer(object):
    """
    Simulates games using the GameDelegate to train a heuristic model
    """

    def __init__(self, game_delegate, players):
        self.game_delegate = game_delegate
        self.players = players

    def train(self, iterations):
        default = self.game_delegate.verbosity
        self.game_delegate.verbosity = 0

        print('Training....')
        for i in tqdm.trange(iterations):
            self.game_delegate.run_game()

            for player in self.players:
                training_data = self.get_training_data(player)

                player.heuristic_model.train(training_data)

        print('\nTraining completed. Player parameters:')
        for player in self.players:
            print(player.heuristic_model.parameters)

        self.game_delegate.verbosity = default

    def get_training_data(self, player):
        """Uses game history from the GameDelegate to generate training data
        Final game states scored according to win/loss, following ones scored
        according to the estimated value of the next board state"""

        data = []
        history = self.game_delegate.history
        features = [player.feature_extractor.extract(board) for board in history]

        for i in range(len(history)):
            data_point = [features[i]]

            if i == len(history) - 1:
                data_point.append(self.get_end_label(history[i], player.player))

            elif i == len(history) - 2:
                succesor_value  = player.heuristic_model.score(features[-1])
                data_point.append(succesor_value)

            else:
                succesor_value  = player.heuristic_model.score(features[i + 2])
                data_point.append(succesor_value)

            data.append(data_point)

        return data

    def get_end_label(self, board, player):
        """Computes the training label for a board state that is a final
        game state."""

        if board.winner == player:
            return 100
        elif board.winner == player * -1:
            return -100
        else:
            return 0

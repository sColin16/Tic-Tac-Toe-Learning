import unittest
import numpy as np
import curses
from unittest.mock import call, patch, MagicMock
from io import StringIO
from curses import wrapper
from math import inf

from players import RandomPlayer, LookAheadPlayer, HumanTTTPlayer, TrainablePlayer
from boards import TTTBoard, TTTParser
from simulators import GameDelegate, Trainer
from mappers import LinearRegression, TTTFeatureExtractor, RegressionModel

class MockStdscr(object):
    """A simple class that can be used to mock the window class of curses.
    It could not be mocked directly."""
    def __init__(self):
        self.clear = MagicMock()
        self.refresh = MagicMock()
        self.addstr = MagicMock()
        self.getkey = MagicMock()

class TestGameDelegate(unittest.TestCase):

    # This test class uses the Tic-Tac-Toe board and parser to simplify testing

    def setUp(self):
        p1 = RandomPlayer(player = 1, seed = 42)
        p2 = RandomPlayer(player = -1, seed = 42)
        self.GD = GameDelegate(p1, p2, TTTBoard, TTTParser(), 0)

    @patch('boards.TTTParser')
    def test_run_game(self, mock_parser):
        self.GD.run_game()

        self.assertEqual(self.GD.board.winner, 1)
        self.assertEqual(len(self.GD.history), 10)

    @patch('boards.TTTParser')
    def test_reset(self, mock_parser):
        self.GD.run_game()
        self.GD.reset()

        self.assertEqual(self.GD.board.board, TestTTTBoard.blank_board)
        self.assertEqual(len(self.GD.history), 1)

    @patch.object(GameDelegate, 'run_game')
    def test_play_game(self, mock_run_game):
        calls = [call()] * 42
        self.GD.play_games(42)

        mock_run_game.assert_has_calls(calls)

    @patch.object(TTTParser, 'draw_board')
    @patch.object(TTTParser, 'announce_winner')
    def test_verbosity_one(self, mock_announce, mock_draw):
        self.GD.verbosity = 1
        self.GD.run_game()

        self.assertEqual(mock_draw.call_count, 1)
        self.assertEqual(mock_announce.call_count, 1)

    @patch.object(TTTParser, 'draw_board')
    @patch.object(TTTParser, 'announce_winner')
    def test_verbosity_two(self, mock_announce, mock_draw):
        self.GD.verbosity = 2
        self.GD.run_game()

        self.assertEqual(mock_draw.call_count, 10)
        self.assertEqual(mock_announce.call_count, 1)

class TestTrainer(unittest.TestCase):
    """
    This test case also uses Tic-Tac-Toe boards to simplify training
    """

    sample_data = [[np.array([1, 0, 0, 0, 0, 0, 0]), 17],
                   [np.array([1, 0, 0, 0, 0, 2, 0]), 19],
                   [np.array([1, 0, 0, 0, 0, 1, 2]), 23],
                   [np.array([1, 0, 0, 0, 0, 3, 2]), 19],
                   [np.array([1, 0, 0, 0, 0, 2, 3]), 17],
                   [np.array([1, 0, 0, 0, 0, 3, 2]), 20],
                   [np.array([1, 0, 0, 0, 0, 1, 2]), 17],
                   [np.array([1, 0, 0, 1, 0, 0, 1]), 15],
                   [np.array([1, 0, 0, 1, 1, 0, 0]), 15],
                   [np.array([1, 1, 0, 0, 0, 0, 0]), 100]]

    def setUp(self):
        LR = LinearRegression([6, 9, 2, 9, 2, 1, 5])
        player1 = LookAheadPlayer(1, 1, LR, TTTFeatureExtractor(), epsilon = 1, seed = 42)
        player2 = LookAheadPlayer(-1, 1, LR, TTTFeatureExtractor(), epsilon = 1, seed = 42)

        self.game_delegate = GameDelegate(player1, player2, TTTBoard, TTTParser(), False)
        self.trainer = Trainer(self.game_delegate, [player1, player2])

    def tearDown(self):
        self.trainer = None

    def test_get_end_label(self):
        self.confirm_end_label(TestTTTBoard.draw_board, 0, 1)
        self.confirm_end_label(TestTTTBoard.X_win_board, 100, 1)
        self.confirm_end_label(TestTTTBoard.O_win_board, -100, 1)

        self.confirm_end_label(TestTTTBoard.draw_board, 0, -1)
        self.confirm_end_label(TestTTTBoard.X_win_board, -100, -1)
        self.confirm_end_label(TestTTTBoard.O_win_board, 100, -1)

    def test_get_training_data(self):
        self.game_delegate.run_game()

        correct = TestTrainer.sample_data
        test = self.trainer.get_training_data(self.trainer.players[0])

        for i in range(len(correct)):
            self.assertEqual(correct[i][0].all(), test[i][0].all())
            self.assertEqual(correct[i][1], test[i][1])

    @patch.object(LinearRegression, 'train')
    @patch('tqdm.trange', range)
    def test_train(self, mock_train):
        self.trainer.train(1)

        self.assertTrue(mock_train.called)

    def confirm_end_label(self, board, label, player):
        self.assertEqual(self.trainer.get_end_label(TTTBoard(board), player), label)

class TestLinearRegression(unittest.TestCase):
    sample_data = [[np.array([5, 5, 7, 3, 1]), 4],
                   [np.array([3, 4, 3, 7, 5]), 3],
                   [np.array([5, 7, 9, 2, 7]), 4],
                   [np.array([8, 6, 6, 7, 5]), 5],
                   [np.array([4, 5, 5, 2, 3]), 8]]

    def test_score(self):
        LR = LinearRegression([1, 2, 3])

        self.assertEqual(LR.score([1, 2, 3]), 14)

    def test_get_gradients(self):
        self.LR = LinearRegression.new(5)

        actual = self.LR.get_gradients(TestLinearRegression.sample_data)
        estimated = self.estimate_gradients(TestLinearRegression.sample_data)

        for i in range(len(actual)):
            self.assertAlmostEqual(actual[i], estimated[i], delta = 0.01)

    @patch.object(LinearRegression, 'get_gradients', side_effect =
        lambda *unused:np.array([1, 2, 3]))
    def test_train(self, mock_gradients):
        LR = LinearRegression([4, 5, 6])
        LR.train(None)

        self.assertEqual(LR.parameters.tolist(), [5, 7, 9])

    def estimate_gradients(self, training_data, learning_rate = 0.01):
        gradients = np.array([0] * 7, dtype = 'float64')

        for data_point in training_data:
            features = data_point[0]
            label = data_point[1]

            for i in range(len(features)):
                gradients[i] += learning_rate * self.estimate_gradient(features, label, i)

        return gradients

    def estimate_gradient(self, features, label, i, h = 0.001):
        cost_0 = self.cost_function(features, label)
        self.LR.parameters[i] += h

        cost_1 = self.cost_function(features, label)
        self.LR.parameters[i] -= h

        return (cost_0 - cost_1)/h

    def cost_function(self, features, label):
        estimate = self.LR.score(features)

        return 0.5 * (label - estimate) **2

class TestRandomPlayer(unittest.TestCase):
    def test_get_move(self):
        board = TTTBoard(TestTTTBoard.blank_board)
        player1 = RandomPlayer(1, seed = 42)
        player2 = RandomPlayer(1, seed = 42)

        a = player1.get_move(board)
        b = player2.get_move(board)

        self.assertEqual(a.board, b.board)

        a = player2.get_move(board)
        b = player1.get_move(board)

        self.assertEqual(a.board, b.board)

class TestTrainablePlayer(unittest.TestCase):
    @patch.object(RandomPlayer, 'get_move')
    def test_get_move_random(self, mock_move):
        player = TrainablePlayer(1, 1)
        player.get_move(None)

        self.assertTrue(mock_move.called)

    @patch.object(TrainablePlayer, 'smart_move', create=True)
    def test_get_move_smart(self, mock_move):
            player = TrainablePlayer(1, 0)
            player.get_move(None)

            self.assertTrue(mock_move.called)

class TestLookAheadPlayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.player = LookAheadPlayer(1, 0, RegressionModel)
        self.board = TTTBoard(TestTTTBoard.blank_board)

        # Mock the minimax function to prevent recursion calls
        self.minimax_copy = self.player.minimax
        self.player.minimax = MagicMock(return_value = (42, None))

    @patch.object(RegressionModel, 'score', create=True)
    def test_get_heuristic(self, mock_score):
        self.player.get_heuristic(self.board)

        mock_score.assert_has_calls([call(TestTTTBoard.blank_board)])

    @patch.object(RegressionModel, 'score', create=True, return_value=42)
    def test_minimax_terminate(self, mock_score):
        test_output = self.minimax_copy(self.board, 0, 0, 0, True)

        self.assertEqual(test_output, (42, self.board))

    def test_minimax_maximizer(self):
        # This patch is necessary otherwise seperate instances of the boards
        # will not be considered equal by mock.assert_has_calls
        with patch.object(TTTBoard, 'get_next_boards',
            return_value = self.board.get_next_boards(1)):
            next_boards = self.board.get_next_boards(1)
            correct_calls = [call(board, 0, 42, inf, False) for board in next_boards]

            test_output = self.minimax_copy(self.board, 1, 42, inf, True)

            self.player.minimax.assert_has_calls(correct_calls)
            self.assertEqual(test_output, (42, next_boards[0]))

    def test_minimax_minimizer(self):
        with patch.object(TTTBoard, 'get_next_boards',
            return_value = self.board.get_next_boards(1)):
            next_boards = self.board.get_next_boards(1)
            correct_calls = [call(board, 0, -inf, 42, True) for board in next_boards]

            test_output = self.minimax_copy(self.board, 1, -inf, 42, False)

            # make sure correct recurssion calls occured
            self.player.minimax.assert_has_calls(correct_calls)
            self.assertEqual(test_output, (42, None))

    def test_minimax_alpha_beta(self):
        # alpha > beta already, so cutoff should call minimax once
        test_output = self.minimax_copy(self.board, 1, inf, -inf, True)
        self.assertEqual(self.player.minimax.call_count, 1)

"""--------------
Specialized Tests
--------------"""

class TestTTTBoard(unittest.TestCase):
    blank_board = [[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]

    mid_game_board = [[-1,  0,  0],
                      [ 1,  1, -1],
                      [ 0,  0,  0]]

    draw_board = [[ 1,  1, -1],
                  [-1, -1,  1],
                  [ 1,  1, -1]]

    X_win_board = [[ 0, -1,  1],
                   [-1,  1,  1],
                   [-1,  0,  1]]

    O_win_board = [[-1,  -1, -1],
                   [ 1,  -1,  1],
                   [ 0,   1,  1]]

    board_A = [[0, -1,  1],
               [0,  1,  0],
               [0,  0,  0]]

    board_B = [[1,  1, -1],
               [0,  0, -1],
               [0,  0,  0]]

    board_C = [[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]]

    board_D = [[-1, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]

    def test_check_winner(self):
        self.assertIsNone(TTTBoard(TestTTTBoard.blank_board).winner)
        self.assertIsNone(TTTBoard(TestTTTBoard.mid_game_board).winner)
        self.assertEqual(TTTBoard(TestTTTBoard.draw_board).winner, 0)
        self.assertEqual(TTTBoard(TestTTTBoard.X_win_board).winner, 1)
        self.assertEqual(TTTBoard(TestTTTBoard.O_win_board).winner, -1)

    def test_validate_move(self):
        TB = TTTBoard(TestTTTBoard.mid_game_board)

        self.assertTrue(TB.validate_move(1, 2, 0))
        self.assertFalse(TB.validate_move(1, 1, 1))

    def test_move(self):
        TB  = TTTBoard(TestTTTBoard.mid_game_board)

        correct_A = [[-1,  0,  0],
                     [ 1,  1, -1],
                     [ 1,  0,  0]]

        correct_B = [[-1,  -1,  0],
                     [ 1,  1, -1],
                     [ 0,  0,  0]]

        self.assertEqual(TB.move(1, 2, 0).board, correct_A)
        self.assertEqual(TB.move(-1, 0, 1).board, correct_B)

    def test_get_next_boards(self):
        correct = [
            [[-1,  1,  0],
             [ 1,  1, -1],
             [ 0,  0,  0]],

            [[-1,  0,  1],
             [ 1,  1, -1],
             [ 0,  0,  0]],

            [[-1,  0,  0],
             [ 1,  1, -1],
             [ 1,  0,  0]],

            [[-1,  0,  0],
             [ 1,  1, -1],
             [ 0,  1,  0]],

            [[-1,  0,  0],
             [ 1,  1, -1],
             [ 0,  0,  1]]
        ]

        test_boards = TTTBoard(TestTTTBoard.mid_game_board).get_next_boards(1)

        for i in range(len(test_boards)):
            self.assertEqual(test_boards[i].board, correct[i])

    def test_full(self):
        self.assertTrue(TTTBoard(TestTTTBoard.draw_board).full())
        self.assertFalse(TTTBoard(TestTTTBoard.blank_board).full())
        self.assertFalse(TTTBoard(TestTTTBoard.mid_game_board).full())

    def test_get_all_threes(self):
        correct = [[-1, 0, 0], [1, 1, -1], [0, 0, 0], [-1, 1, 0],
                    [0, 1, 0], [0, -1, 0], [-1, 1, 0], [0, 1, 0]]

        self.assertEqual(TTTBoard(TestTTTBoard.mid_game_board).
            get_all_threes(), correct)

class TestTTTFeatureExtractor(unittest.TestCase):
    def test_extract(self):
        self.confirm_features(TestTTTBoard.blank_board, [1] + [0] * 6)
        self.confirm_features(TestTTTBoard.mid_game_board, [1, 0, 0, 0, 0, 2, 2])
        self.confirm_features(TestTTTBoard.draw_board, [1] + [0] * 6)
        self.confirm_features(TestTTTBoard.X_win_board, [1, 1, 0, 1, 1, 0, 0])
        self.confirm_features(TestTTTBoard.O_win_board, [1, 0, 1, 1, 1, 0, 0])
        self.confirm_features(TestTTTBoard.board_A, [1, 0, 0, 1, 0, 3, 0])
        self.confirm_features(TestTTTBoard.board_B, [1, 0, 0, 0, 1, 3, 2])
        self.confirm_features(TestTTTBoard.board_C, [1, 0, 0, 0, 0, 4, 0])
        self.confirm_features(TestTTTBoard.board_D, [1, 0, 0, 0, 0, 0, 3])

    def confirm_features(self, board, features):
        test_features = TTTFeatureExtractor().extract(TTTBoard(board))

        self.assertEqual(test_features.all(), np.array(features).all())

class TestTTTParser(unittest.TestCase):
    def test_decode(self):
        parser = TTTParser()

        self.assertEqual(parser.decode(1), 'X')
        self.assertEqual(parser.decode(-1), 'O')
        self.assertEqual(parser.decode(0), ' ')
        self.assertEqual(parser.decode('X'), '?')

    def test_draw_token(self):
        self.assert_token_call(1, 0, 0, call(1, 2, 'X', 0))
        self.assert_token_call(-1, 1, 1, call(3, 6, 'O', 0))
        self.assert_token_call(0, 2, 2, call(5, 10, ' ', 0))

    @patch.object(TTTParser, 'draw_frame')
    @patch.object(TTTParser, 'draw_token')
    @patch.object(curses, 'curs_set')
    def test_draw_complete(self, mock_curs_set, mock_token, mock_frame):
        parser = TTTParser()
        mock_stdscr = MockStdscr()

        correct_calls = [
            call(mock_stdscr, 1, 0, 0),
            call(mock_stdscr, 1, 0, 1),
            call(mock_stdscr, -1, 0, 2),
            call(mock_stdscr, -1, 1, 0),
            call(mock_stdscr, -1, 1, 1),
            call(mock_stdscr, 1, 1, 2),
            call(mock_stdscr, 1, 2, 0),
            call(mock_stdscr, 1, 2, 1),
            call(mock_stdscr, -1, 2, 2)
        ]

        parser.draw_complete(mock_stdscr, TestTTTBoard.draw_board)

        self.assertTrue(mock_stdscr.clear.called)
        self.assertTrue(mock_stdscr.refresh.called)
        self.assertTrue(mock_frame.called)
        mock_curs_set.assert_has_calls([call(0)])
        mock_token.assert_has_calls(correct_calls)

    def test_announce_winner(self):
        self.assert_announce_winner_call(1, ' X wins!')
        self.assert_announce_winner_call(-1, ' O wins!')
        self.assert_announce_winner_call(0, ' It\'s a draw!')

    def assert_token_call(self, player, row, column, call):
        parser = TTTParser()
        mock_stdscr = MockStdscr()

        parser.draw_token(mock_stdscr, player, row, column)
        mock_stdscr.addstr.assert_has_calls([call])

    def assert_announce_winner_call(self, winner, output):
        parser = TTTParser()
        mock_stdscr = MockStdscr()
        calls = [call('\n\n'), call('\n Press Enter')] + [call(output)]

        parser.announce_winner_wrapper(mock_stdscr, winner)

        mock_stdscr.addstr.assert_has_calls(calls, any_order = True)
        self.assertTrue(mock_stdscr.getkey.called)

class TestHumanTTTPlayer(unittest.TestCase):
    def test_get_input(self):
        # Test each arrow key
        self.assert_get_input_call(TestTTTBoard.blank_board, 0, 0, 'KEY_UP', 2, 0, False)
        self.assert_get_input_call(TestTTTBoard.blank_board, 2, 2, 'KEY_DOWN', 0, 2, False)
        self.assert_get_input_call(TestTTTBoard.blank_board, 0, 0, 'KEY_LEFT', 0, 2, False)
        self.assert_get_input_call(TestTTTBoard.blank_board, 2, 2, 'KEY_RIGHT', 2, 0, False)

        # Test the submit key
        self.assert_get_input_call(TestTTTBoard.mid_game_board, 0, 0, '\n', 0, 0, False)
        self.assert_get_input_call(TestTTTBoard.mid_game_board, 0, 1, '\n', 0, 1, True)

    @patch.object(TTTParser, 'draw_token')
    def test_draw_selection(self, mock_draw):
        self.assert_draw_selection(TestTTTBoard.blank_board, 0, 0, 1, 1)
        self.assert_draw_selection(TestTTTBoard.draw_board, 1, 1, -1, 2)

    def test_get_first_open(self):
        self.assert_first_open(TestTTTBoard.blank_board, 0, 0)
        self.assert_first_open(TestTTTBoard.mid_game_board, 0, 1)
        self.assert_first_open(TestTTTBoard.O_win_board, 2, 0)
        self.assert_first_open(TestTTTBoard.board_B, 1, 0)

    @patch.object(TTTParser, 'draw_complete')
    @patch.object(HumanTTTPlayer, 'draw_selection')
    def assert_get_input_call(self, board, start_row, start_column, key,
            end_row, end_column, submit, mock_selection, mock_draw):

        player = HumanTTTPlayer(1)
        parser = TTTParser()
        mock_stdscr = MockStdscr()

        mock_stdscr.getkey.side_effect = lambda: key

        test_output = player.get_input(mock_stdscr, TTTBoard(board), parser, start_row, start_column)

        self.assertEqual(test_output, (end_row, end_column, submit))
        self.assertTrue(mock_draw.called)
        self.assertTrue(mock_selection.called)

    def assert_first_open(self, board, row, column):
        player = HumanTTTPlayer(1)
        test_output = player.get_first_open(board)

        self.assertEqual(test_output, (row, column))

    @patch.object(TTTParser, 'draw_token')
    def assert_draw_selection(self, board, row, column, token, color, mock_draw):
        parser = TTTParser()
        mock_stdscr = MockStdscr()
        player = HumanTTTPlayer(1)

        player.draw_selection(mock_stdscr, TTTBoard(board), row, column, parser)

        mock_draw.assert_has_calls([call(mock_stdscr, token, row, column, color)])

if __name__ == '__main__':
    unittest.main()

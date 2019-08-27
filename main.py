from boards import TTTBoard, TTTParser
from players import HumanTTTPlayer, LookAheadPlayer
from mappers import TTTFeatureExtractor, LinearRegression
from simulators import GameDelegate, Trainer

perfect_O = [5.84737142, -97.96467256, 103.59219359, -29.53656259, 26.45196385
,-15.40886535, 10.69894667]

simple_X = [0, 100, -100, 30, -30, 10, -10]
simple_O = [0, -100, 100, -30, 30, -10, 10]

if __name__ == '__main__':
    x_model = LinearRegression.new(7)
    o_model = LinearRegression.new(7)

    player1 = LookAheadPlayer(1, 1, x_model, TTTFeatureExtractor(), epsilon = 0.2)
    player2 = LookAheadPlayer(-1, 1, o_model, TTTFeatureExtractor(), epsilon = 0.2)

    game_delegate = GameDelegate(player1, player2, TTTBoard, TTTParser())
    trainer = Trainer(game_delegate, [player1, player2])

    trainer.train(1000)

    game_delegate.players[0].epsilon = 0
    game_delegate.players[0].depth = 4
    game_delegate.players[1] = HumanTTTPlayer(-1)

    game_delegate.play_games(10)

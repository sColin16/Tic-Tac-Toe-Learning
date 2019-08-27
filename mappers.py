"""
This file contains classes that map various objects:

 - FeatureExtractors (map boards to a list of features).
 - RegressionModels (map features to a number),
"""

import numpy as np
import random

# TODO: implement Neural Player
# TODO: use convolution neural net

"""-----------------------
FeatureExtractors
-----------------------"""

class FeatureExtractor(object):
    """Takes a Board instance as input, and outputs a numpy array of features.
    Subclasses must define the following methods:
    - extract(self, board) -> representation of features (list, numpy array, etc)
    """

    def extract(self, board):
        return board.board

class TTTFeatureExtractor(FeatureExtractor):
    """
    Extracts 6 features of a Tic-Tac-Toe board, and 1 bias:

    0 - bias (always 1)
    1 - # of 3 Xs in a row
    2 - # of 3 Os in a row
    3 - # of 2 Xs in a row (w/ blank)
    4 - # of 2 Os in a row
    5 - # of 1 X in a row (w/ 2 blanks)
    6 - # of 1 O in a row
    """

    def extract(self, board):
        features = [1] + [0] * 6
        threes = board.get_all_threes()

        for three in threes:
            Xs = three.count(1)
            Os = three.count(-1)

            if Os == 0:
                if Xs == 3:
                    features[1] += 1
                elif Xs == 2:
                    features[3] += 1
                elif Xs == 1:
                    features[5] += 1

            elif Xs == 0:
                if Os == 3:
                    features[2] += 1
                elif Os == 2:
                    features[4] += 1
                elif Os == 1:
                    features[6] += 1

        return np.array(features)

"""-----------------------
RegressionModels
-----------------------"""

class RegressionModel(object):
    """Accepts a numpy array of numbers, and returns a single number.
    Subclasses must define:
    - score(self, features) -> number
    - train(training_data) -> Train one iteration of the model
    """

    pass

class LinearRegression(RegressionModel):
    """Transforms an array of features into a single number using a linear
    combination."""

    def __init__(self, parameters = None):
        self.parameters = np.array(parameters)

    @classmethod
    def new(cls, param_num):
        return cls(np.array([random.uniform(-100, 100) for i in range(param_num)]))

    def score(self, features):
        return np.matmul(np.array(features), self.parameters)

    def train(self, training_data, learning_rate=0.01):
        gradients = self.get_gradients(training_data, learning_rate)
        #print(gradients)
        self.parameters += gradients

    def get_gradients(self, training_data, learning_rate=0.01):
        gradients = np.array([0] * len(self.parameters), dtype = 'float64')

        for data_point in training_data:
            features = data_point[0]
            label = data_point[1]
            estimate = self.score(features)

            gradients += (learning_rate * (label - estimate)) * features

        return gradients

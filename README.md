# Tic-Tac-Toe-Learning

This repo provides implements a temporal-learning agent that learns to play Tic-Tac-Toe by playing against itself,
as well as a terminal-based interface to play against this AI opponent.

## Usage
1. Install dependencies:
```pip3 install -r requirements.txt```

2. To play a game against the AI, run:
```python3 main.py```.
The AI agent will train by playing 1000 games against another AI opponent before the interface displays

3. Use your arrow keys to move your piece (highlighted white), and press enter to submit your move. 
A piece highlighted red means you are trying to place the piece where one already exists

## How it Works

The AI agent learns to play through temporal difference learning, a type of reinforcement learning.
The agent determines a score for any state of the board, based on features of the board (specifically,
the number of 2 and 3-in-a-rows), and weights for those features it learns through playing.

The AI learns weights for each feature through gradient descent. At the end of each game, the board is given
a score (-100, 0, or 100), based on the outcome of the game. The AI updates its weights so that they match
the score for that board more closely. For all other intermediate board states, the AI updates its weights
to more closely match its own estimated score for the following board. The result is a boot-strapping process,
where the final score for a game "trickles down" to the intermediate states. When training is complete, the
score the AI assigns to any board is the expected value of that state, in terms of the final outcome.

When it is it's turn to select a move, it iterates through all possible moves, scores the corresponding boards,
and then selects the board with the highest score.

## Tests
This repo also includes tests for various components. To run the tests, do: ```python3 tests.py```

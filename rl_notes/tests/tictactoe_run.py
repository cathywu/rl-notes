from extensive_form_game import ExtensiveFormGame
from tictactoe import TicTacToe

tictactoe = TicTacToe()
state = tictactoe.get_initial_state()
state = tictactoe.get_transition(state, (0,0))
state = tictactoe.get_transition(state, (1,2))
state = tictactoe.get_transition(state, (1,1))
state = tictactoe.get_transition(state, (2,1))
assert tictactoe.get_winner(state) is None
assert tictactoe.get_reward(state) == {CROSS:0, NOUGHT:0}

state = tictactoe.get_transition(state, (2,2))
print(tictactoe.to_string(state))
assert tictactoe.get_winner(state) == CROSS
assert tictactoe.get_reward(state) == {CROSS:1, NOUGHT:-1}

# play a random game
import random
state = tictactoe.get_initial_state()
while not tictactoe.is_terminal(state):
    actions = tictactoe.get_actions(state)
    state = tictactoe.get_transition(state, random.choice(actions))
    print(tictactoe.to_string(state) + "\n")
print("winner is %s" % tictactoe.get_winner(state))

from extensive_form_game import GameNode
from tictactoe import TicTacToe
from graph_visualisation import GraphVisualisation


tictactoe = TicTacToe()
initial_state = tictactoe.get_initial_state()
initial_state = [['x', 'o', 'o'],
                [' ', ' ', 'x'],
                [' ', ' ', ' ']]
next_state = tictactoe.get_transition(initial_state, (1, 1))
backward_induction = BackwardInduction(tictactoe, do_cache = False)
solution = backward_induction.backward_induction(next_state)

gv = GraphVisualisation()
graph = gv.node_to_graph(tictactoe, solution, print_state = True, print_value = True)
graph

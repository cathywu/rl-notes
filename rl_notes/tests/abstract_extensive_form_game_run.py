from extensive_form_game import ExtensiveFormGame
from backward_induction import BackwardInduction
from graph_visualisation import GraphVisualisation

game = AbstractExtensiveFormGame()
backward_induction = BackwardInduction(game)
solution = backward_induction.backward_induction(game.get_initial_state())

gv = GraphVisualisation(max_level = 5)
graph = gv.node_to_graph(game, game.game_tree(), print_value = False)
graph

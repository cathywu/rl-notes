from gridworld import GridWorld
from graph_visualisation import GraphVisualisation
from qtable import QTable
from single_agent_mcts import SingleAgentMCTS
from q_policy import QPolicy
from multi_armed_bandit.ucb import UpperConfidenceBounds


mdp = GridWorld()
qfunction = QTable()
root_node = SingleAgentMCTS(mdp, qfunction, UpperConfidenceBounds()).mcts(timeout=0.1)
mdp.visualise_q_function(qfunction)

policy = QPolicy(qfunction)
mdp.visualise_policy(policy)

gv = GraphVisualisation(max_level=6)
graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
graph.view()

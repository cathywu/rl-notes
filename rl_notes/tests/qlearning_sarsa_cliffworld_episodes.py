from gridworld import CliffWorld
from tests.compare_convergence_curves import qlearning_vs_sarsa
from gridworld import CliffWorld
from tests.plot import Plot

mdp_q = CliffWorld()
mdp_s = CliffWorld()

qlearning_vs_sarsa(mdp_q, mdp_s, episodes=2000)
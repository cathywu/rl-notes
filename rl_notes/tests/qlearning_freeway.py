from ale_wrapper import ALEWrapper
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

print("==========\nTabular Q-learning: Freeway\n==========")

qfunction = QTable()
mdp = ALEWrapper(version="ALE/Freeway-ram-v5", render_mode="human")
QLearning(mdp, EpsilonGreedy(epsilon=1.0), qfunction).execute(episodes=2000)

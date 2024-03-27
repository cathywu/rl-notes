from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

print("==========\nTabular Q-learning: Gridworld\n==========")

mdp = GridWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))
mdp.visualise_q_function(qfunction)

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))

qfunction = QTable()
mdp = GridWorld()

rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=2000)

Plot.plot_cumulative_rewards(["Q-learning"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Q-learning"], [rewards])

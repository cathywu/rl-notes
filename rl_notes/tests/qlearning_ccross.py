from contested_crossing import ContestedCrossing
from gridworld import GridWorld
from qtable import QTable
from stochastic_q_policy import StochasticQPolicy
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

print("==========\nTabular Q-learning: Contested crossing\n==========")


episodes = 2000
episodes_per_evaluation = 20
qfunction = QTable()
mdp = ContestedCrossing()
policy = StochasticQPolicy(qfunction, EpsilonGreedy())
rewards = mdp.execute_policy(policy, episodes=1)
for _ in range(int(episodes / episodes_per_evaluation)):
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=episodes_per_evaluation)
    policy = StochasticQPolicy(qfunction, EpsilonGreedy())
    rewards += mdp.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(
    ["Q-learning"],
    [rewards],
    smoothing_factor=0.0,
    episodes_per_evaluation=episodes_per_evaluation,
)
Plot.plot_cumulative_rewards(
    ["Q-learning"], [rewards], episodes_per_evaluation=episodes_per_evaluation
)

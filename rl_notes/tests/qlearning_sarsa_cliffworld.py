from gridworld import CliffWorld
from qtable import QTable
from qlearning import QLearning
from sarsa import SARSA
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot
from q_policy import QPolicy

# Train using Q-learning
mdp = CliffWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)

# Exrract the policy
policy = QPolicy(qfunction)

# Execute the policy and get all rewards: 2000 training and 2000 test
mdp.execute_policy(policy, episodes=2000)
q_learning_rewards = mdp.get_rewards()

# Train using SARSA
mdp = CliffWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)

# Execute the policy
policy = QPolicy(qfunction)

mdp.execute_policy(policy, episodes=2000)
sarsa_rewards = mdp.get_rewards()

Plot.plot_rewards_per_episode(["Q-learning", "SARSA"], [q_learning_rewards, sarsa_rewards])

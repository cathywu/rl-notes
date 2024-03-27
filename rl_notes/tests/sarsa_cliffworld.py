from gridworld import CliffWorld
from qtable import QTable
from sarsa import SARSA
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = CliffWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)
#print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))

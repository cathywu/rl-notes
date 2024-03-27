from gridworld import GridWorld
from qtable import QTable
from sarsa import SARSA
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(), qfunction).execute()
print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))

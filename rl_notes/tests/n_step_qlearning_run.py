from gridworld import GridWorld
from qtable import QTable
from n_step_qlearning import NStepQLearning
from qlearning import QLearning
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

# Illustrate the n-step feedback from the first episode
mdp = GridWorld(noise=0.0, goals=[((3, 2), 1)])
qfunction = QTable()
NStepQLearning(mdp, EpsilonGreedy(), qfunction, 5, alpha=0.4).execute(episodes=1)
print(mdp.q_function_to_string(qfunction))

mdp = GridWorld(noise=0.0, goals=[((3, 2), 1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction, alpha=0.4).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

mdp = GridWorld(noise=0.0, goals=[((3, 2), 1)])
qfunction = QTable()
NStepQLearning(mdp, EpsilonGreedy(), qfunction, 5, alpha=0.4).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))

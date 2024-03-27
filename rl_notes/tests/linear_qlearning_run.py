from gridworld import GridWorld
from qlearning import QLearning
from linear_qfunction import LinearQFunction
from gridworld_feature_extractor import GridWorldFeatureExtractor
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld()
features = GridWorldFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))


from gridworld_better_feature_extractor import GridWorldBetterFeatureExtractor

mdp = GridWorld()
features = GridWorldBetterFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))

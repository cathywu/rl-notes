from contested_crossing import ContestedCrossing
from qtable import QTable
from sarsa import SARSA
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = ContestedCrossing()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

#mdp.visualise_q_function(qfunction)
policy = QPolicy(qfunction)
mdp.visualise_policy(policy, "Policy plot", mode=0)
mdp.visualise_policy(policy, "Path plot", mode=1)
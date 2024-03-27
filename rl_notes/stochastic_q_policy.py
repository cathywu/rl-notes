from policy import StochasticPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

""" Make a stochastic policy from a qfunction and a mutli-armed bandit.
    This helps to avoid e.g. loops in policies.
    This policy cannot be updated -- it is only for execution.
"""


class StochasticQPolicy(StochasticPolicy):
    def __init__(self, qfunction, bandit=EpsilonGreedy(epsilon=.05)):
        self.qfunction = qfunction
        self.bandit = bandit

    def select_action(self, state, actions):
        return self.bandit.select(state, actions, self.qfunction)

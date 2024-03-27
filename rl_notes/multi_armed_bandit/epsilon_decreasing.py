from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy


class EpsilonDecreasing(MultiArmedBandit):
    def __init__(self, epsilon=1.0, alpha=0.999, lower_bound=0.1):
        self.epsilon_greedy_bandit = EpsilonGreedy(epsilon)
        self.initial_epsilon = epsilon
        self.alpha = alpha
        self.lower_bound = lower_bound

    def reset(self):
        self.epsilon_greedy_bandit = EpsilonGreedy(self.initial_epsilon)

    def select(self, state, actions, qfunction):
        result = self.epsilon_greedy_bandit.select(state, actions, qfunction)
        self.epsilon_greedy_bandit.epsilon = max(
            self.epsilon_greedy_bandit.epsilon * self.alpha, self.lower_bound
        )
        return result

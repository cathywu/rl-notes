import math
import random

from policy import StochasticPolicy


""" A two-action policy implemented using logistic regression from first principles """


class LogisticRegressionPolicy(StochasticPolicy):

    """ Create a new policy, with given parameters theta (randomly if theta is None)"""

    def __init__(self, actions, num_params, theta=None):
        assert len(actions) == 2

        self.actions = actions

        if theta is None:
            theta = [0.0 for _ in range(num_params)]
        self.theta = theta

    """ Select one of the two actions using the logistic function for the given state """

    def select_action(self, state):
        # Get the probability of selecting the first action
        probability = self.get_probability(state, self.actions[0])

        # With a probability of 'probability' take the first action
        if random.random() < probability:
            return self.actions[0]
        return self.actions[1]

    """ Update our policy parameters according using the gradient descent formula:
          theta <- theta + alpha * G * nabla J(theta), 
          where G is the future discounted reward
    """

    def update(self, states, actions, deltas):
        for t in range(len(states)):
            gradient_log_pi = self.gradient_log_pi(states[t], actions[t])
            # Update each parameter
            for i in range(len(self.theta)):
                self.theta[i] += deltas[t] * gradient_log_pi[i]

    """ Get the probability of applying an action in a state """

    def get_probability(self, state, action):
        # Calculate y as the linearly weight product of the 
        # policy parameters (theta) and the state
        y = self.dot_product(state, self.theta)

        # Pass y through the logistic regression function to convert it to a probability
        probability = self.logistic_function(y)

        if action == self.actions[0]:
            return probability
        return 1 - probability

    """ Computes the gradient of the log of the policy (pi),
    which is needed to get the gradient of the objective (J).
    Because the policy is a logistic regression, using the policy parameters (theta).
        pi(actions[0] | state)= 1 / (1 + e^(-theta * state))
        pi(actions[1] | state) = 1 / (1 + e^(theta * state))
    When we apply a logarithmic transformation and take the gradient we end up with:
        grad_log_pi(left | state) = state - state * pi(left|state)
        grad_log_pi(right | state) = - state * pi(0|state)
    """

    def gradient_log_pi(self, state, action):
        y = self.dot_product(state, self.theta)
        if action == self.actions[0]:
            return [s_i - s_i * self.logistic_function(y) for s_i in state]
        return [-s_i * self.logistic_function(y) for s_i in state]

    """ Standard logistic function """

    @staticmethod
    def logistic_function(y):
        return 1 / (1 + math.exp(-y))

    """ Compute the dot product between two vectors """

    @staticmethod
    def dot_product(vec1, vec2):
        return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])

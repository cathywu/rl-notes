from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction
from qtable import QTable

class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001):

        while True:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                old_value = values.get_value(state)
                new_value = values.get_q_value(self.mdp, state, policy.select_action(state))
                values.update(state, new_value)
                delta = max(delta, abs(old_value - new_value))

            # terminate if the value function has converged
            if delta < theta:
                break

        return values

    """ Implmentation of policy iteration iteration. Returns the number of iterations exected """

    def policy_iteration(self, max_iterations=100, theta=0.001):

        # create a value function to hold details
        values = TabularValueFunction()

        for i in range(1, max_iterations + 1):
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta)
            for state in self.mdp.get_states():
                old_action = self.policy.select_action(state)

                q_values = QTable()
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = values.get_q_value(self.mdp, state, action)
                    q_values.update(state, action, new_value)

                # V(s) = argmax_a Q(s,a)
                (new_action, _) = q_values.get_max_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)

                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )

            if not policy_changed:
                return i

        return max_iterations

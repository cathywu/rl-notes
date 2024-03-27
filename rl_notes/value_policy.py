from policy import DeterministicPolicy
from qtable import QTable

""" Make a deterministic policy from a value function.
    This policy cannot be updated -- it is only for execution.
"""


class ValuePolicy(DeterministicPolicy):
    def __init__(self, mdp, values):
        self.mdp = mdp
        self.values = values

    def select_action(self, state, actions):
        qfunction = QTable()
        for action in actions:
            q_value = self.values.get_q_value(self.mdp, state, action)
            qfunction.update(state, action, q_value)
        return qfunction.get_max_q(state, actions)[0]
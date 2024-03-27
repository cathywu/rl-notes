from policy import DeterministicPolicy

""" Make a deterministic policy from a qfunction.
    This policy cannot be updated -- it is only for execution.
"""

class QPolicy(DeterministicPolicy):
    def __init__(self, qfunction):
        self.qfunction = qfunction

    def select_action(self, state, actions):
        return self.qfunction.get_max_q(state, actions)[0]

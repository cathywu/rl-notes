from temporal_difference_learner import TemporalDifferenceLearner


class SARSA(TemporalDifferenceLearner):
    def state_value(self, state, action):
        return self.qfunction.get_q_value(state, action)

from model_free_reinforcement_learner import ModelFreeReinforcementLearner


class SARSA(ModelFreeReinforcementLearner):
    def state_value(self, state, action):
        return self.qfunction.get_q_value(state, action)

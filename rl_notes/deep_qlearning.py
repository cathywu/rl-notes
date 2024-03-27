from qlearning import QLearning


class DeepQLearning(QLearning):
    """
    We introduce this class because, unlike non-deep versions of QLearning, the neural network
    optimiser (we used ADAM) uses the learning rate. This needs to get instantiated when the q-network gets
    instantiated. Therefore, we shouldn't technically pass alpha * delta to the deep q-function update procedure, since
    it is already accounted for by the optimiser.
    """

    def get_delta(self, reward, q_value, state, next_state, next_action):
        next_state_value = self.state_value(next_state, next_action)
        delta = reward + self.mdp.discount_factor * next_state_value - q_value
        return delta

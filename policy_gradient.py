class PolicyGradient:
    def __init__(self, mdp, policy, alpha) -> None:
        super().__init__()
        self.alpha = alpha  # Learning rate (gradient update step-size)
        self.mdp = mdp
        self.policy = policy

    """ Generate and store an entire episode trajectory to use to update the policy """

    def execute(self, episodes=100):
        for _ in range(episodes):
            actions = []
            states = []
            rewards = []

            state = self.mdp.get_initial_state()
            episode_reward = 0
            while not self.mdp.is_terminal(state):
                action = self.policy.select_action(state)
                next_state, reward = self.mdp.execute(state, action)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            deltas = self.calculate_deltas(rewards)
            self.policy.update(states=states, actions=actions, deltas=deltas)

    def calculate_deltas(self, rewards):
        """
        Generate a list of the discounted future rewards at each step of an episode
        Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        We can use that pattern to populate the discounted_rewards array.
        """
        T = len(rewards)
        discounted_future_rewards = [0 for _ in range(T)]
        # The final discounted reward is the reward you get at that step
        discounted_future_rewards[T - 1] = rewards[T - 1]
        for t in reversed(range(0, T - 1)):
            discounted_future_rewards[t] = (
                rewards[t]
                + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
            )
        deltas = []
        for t in range(len(discounted_future_rewards)):
            deltas += [
                self.alpha
                * (self.mdp.get_discount_factor() ** t)
                * discounted_future_rewards[t]
            ]
        return deltas

class NStepReinforcementLearner:
    def __init__(self, mdp, bandit, qfunction, n, alpha=0.1):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction
        self.n = n

    def execute(self, episodes=100):
        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            rewards = []
            states = [state]
            actions = [action]

            while len(states) > 0:
                if not self.mdp.is_terminal(state):
                    (next_state, reward) = self.mdp.execute(state, action)
                    rewards += [reward]
                    next_actions = self.mdp.get_actions(next_state)

                    if not self.mdp.is_terminal(next_state):
                        next_action = self.bandit.select(
                            next_state, next_actions, self.qfunction
                        )
                        states += [next_state]
                        actions += [next_action]

                if len(rewards) == self.n or self.mdp.is_terminal(state):
                    n_step_rewards = sum(
                        [
                            self.mdp.discount_factor ** i * rewards[i]
                            for i in range(len(rewards))
                        ]
                    )

                    if not self.mdp.is_terminal(state):
                        next_state_value = self.state_value(next_state, next_action)
                        n_step_rewards = (
                            n_step_rewards
                            + self.mdp.discount_factor ** self.n * next_state_value
                        )

                    q_value = self.qfunction.get_q_value(
                        states[0], actions[0]
                    )

                    self.qfunction.update(
                        states[0],
                        actions[0],
                        self.alpha * (n_step_rewards - q_value),
                    )

                    rewards = rewards[1 : self.n + 1]
                    states = states[1 : self.n + 1]
                    actions = actions[1 : self.n + 1]

                state = next_state
                action = next_action

    """ Get the value of a state """

    def state_value(self, state, action):
        abstract

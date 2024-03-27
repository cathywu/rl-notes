class ActorCritic:
    def __init__(self, mdp, actor, critic, alpha=0.1):
        self.alpha = alpha  # Learning rate (gradient update step-size)
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100):
        for _ in range(episodes):
            actions = []
            states = []
            rewards = []
            next_states = []

            state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                action = self.actor.select_action(state, self.mdp.get_actions(state))
                next_state, reward = self.mdp.execute(state, action)
                self.update_critic(reward, state, action, next_state)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)

                state = next_state

            self.update_actor(
                rewards=rewards, states=states, actions=actions, next_states=next_states
            )

    """ Update the actor using a batch of rewards, states, actions, and next states """

    def update_actor(self, rewards, states, actions, next_states):
        abstract

    """ Update the critc using a reward, state, action, and next state """

    def update_critic(self, reward, state, action, next_state):
        abstract

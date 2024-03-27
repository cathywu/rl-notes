from actor_critic import ActorCritic


class AdvantageActorCritic(ActorCritic):
    """
    This implements the actor critic algorithms using the Advantage Actor Critic method.
    This replaces the reward-to-go (G) in vanilla policy gradient with an advantage:
        advantage = Q[s_t, a_t] - V[s_t]
                  = r + gamma * V[s_{t+1}] - V[s_t]
    """

    def __init__(self, mdp, actor, critic, alpha=0.1):
        super().__init__(mdp, actor, critic, alpha)

    def update_actor(self, rewards, states, actions, next_states):
        # advantage = r + gamma * V[s_{t+1}] - V[s_t]
        value_function = self.critic
        advantages = []
        for t in range(len(states)):
            state = states[t]
            next_state = next_states[t]
            value = value_function.get_value(state)
            next_value = value_function.get_value(next_state)

            reward = rewards[t]

            advantage = reward + self.mdp.get_discount_factor() * next_value - value
            advantages.append(advantage)

        self.actor.update(states, actions, advantages)

    def update_critic(self, reward, state, action, next_state):
        # We want to take the MSE between the state_value and r + gamma * next_state_value
        state_value = self.critic.get_value(state=state)
        next_state_value = self.critic.get_value(state=next_state)
        delta = reward + self.mdp.get_discount_factor() * next_state_value - state_value
        self.critic.update(state=state, delta=delta)

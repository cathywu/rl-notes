from actor_critic import ActorCritic
from deep_agent import DeepAgent


class QActorCritic(ActorCritic, DeepAgent):

    def __init__(self, mdp, actor, critic, alpha=0.1):
        super().__init__(mdp, actor, critic, alpha)

    def update_actor(self, rewards, states, actions, next_states):
        q_values = [
            self.critic.qfunction.get_q_value(state, action)
            for state, action in zip(states, actions)
        ]
        self.actor.update(states, actions, q_values)

    def update_critic(self, reward, state, action, next_state):
        next_state = self.encode_state(next_state)
        next_action = self.actor.select_action(next_state)
        q_value = self.critic.qfunction.get_q_value(state, action)
        delta = self.critic.get_delta(reward, q_value, state, next_state, next_action)
        self.critic.qfunction.update(state=state, action=action, delta=delta)

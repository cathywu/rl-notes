import random
from model_free_learner import ModelFreeLearner
    
class DoubleQLearning(ModelFreeLearner):
    def __init__(self, mdp, bandit, qfunction1, qfunction2, alpha=0.01):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction1 = qfunction1
        self.qfunction2 = qfunction2

    def execute(self, episodes=100):
        rewards = []

        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction1)
            
            episode_reward = 0.0

            while not self.mdp.is_terminal(state):
        
                (next_state, reward) = self.mdp.execute(state, action)

                actions = self.mdp.get_actions(next_state)

                if random.random() < 0.5:
                    next_action = self.bandit.select(next_state, actions, self.qfunction1)
                    q_value = self.qfunction1.get_q_value(state, action)
                    (_, max_q_value) = self.qfunction2.get_max_q(next_state, self.mdp.get_actions(next_state))
                    delta = self.get_delta(reward, q_value, max_q_value)
                    self.qfunction1.update(state, action, delta)
                else:
                    next_action = self.bandit.select(next_state, actions, self.qfunction2)
                    q_value = self.qfunction2.get_q_value(state, action)
                    (_, max_q_value) = self.qfunction1.get_max_q(next_state, self.mdp.get_actions(next_state))
                    delta = self.get_delta(reward, q_value, max_q_value)
                    self.qfunction2.update(state, action, delta)

                state = next_state
                action = next_action
                episode_reward += reward

            rewards.append(episode_reward)

        return rewards

    def get_delta(self, reward, q_value, next_state_value):
        delta = reward + self.mdp.discount_factor * next_state_value - q_value
        return self.alpha * delta

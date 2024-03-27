import random
from collections import namedtuple, deque

from temporal_difference_learner import TemporalDifferenceLearner

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_capacity(self):
        return self.capacity

    def __len__(self):
        return len(self.memory)

class ExperienceReplayLearner(TemporalDifferenceLearner):
    def __init__(self, mdp, bandit, policy_qfunction, target_qfunction, replay_buffer=ReplayBuffer(10000), alpha=0.001, max_buffer_size=5000, replay_period=1000, batch_size=64):
        super().__init__(mdp, bandit, policy_qfunction, alpha=alpha)
        self.replay_buffer = replay_buffer
        self.replay_period = replay_period
        self.policy_qfunction = policy_qfunction
        self.target_qfunction = target_qfunction

    def execute(self, episodes=100):

        rewards = []
        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.policy_qfunction)

            episode_reward = 0.0
            step = 0
            while not self.mdp.is_terminal(state):
                (next_state, reward, done) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.policy_qfunction)
                q_value = self.policy_qfunction.get_q_value(state, action)
                next_state_value = self.state_value(next_state, next_action)

                #delta = self.get_delta(reward, q_value, state, next_state, next_action)

                #experience = (state, action, delta, done)

                self.buffer.push(state, action, next_state, reward)
                if len(self.replay_buffer) >= self.replay_buffer.get_capacity():
                    transitions = self.replay_buffer.sample(self.batch_size)
                    self.policy_qfunction.update(transitions)

                self.target_qfunction.soft_update(self.policy_qfunction)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.discount_factor ** step)
                step += 1

            rewards.append(episode_reward)

        return rewards

    """ Update from a mini batch """
    def update(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        self.policy_qfunction.multi_update(mini_batch)

    def state_value(self, state, action):
        (_, max_q_value) = self.target_qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value

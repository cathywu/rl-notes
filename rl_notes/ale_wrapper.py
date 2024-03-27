import gymnasium as gym
from mdp import MDP


"""
A wrapper class around the gymnasium class for the Arcade Learning Environment
(https://gymnasium.farama.org/environments/atari/)
to meet the requirements for the MDP class interface.
"""


class ALEWrapper(MDP):

    def __init__(self, version, render_mode="rgb_array", discount_factor=1.):
        self.env = gym.make(version, render_mode=render_mode)
        observation, info = self.env.reset()
        self.terminated = False
        self.discount_factor = discount_factor
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def get_actions(self, state=None):
        num_actions = self.env.action_space.n
        return list(range(num_actions))

    def get_initial_state(self):
        observation, info = self.env.reset()
        return tuple(observation)

    def reset(self):
        observation, info = self.env.reset()
        #**** return tuple(observation)
        return tuple(observation)

    def step(self, action):
        return self.env.step(action)

    """ Return true if and only if state is a terminal state of this MDP """

    def is_terminal(self, state):
        # This hacks the gym interface by recording termination status during 'execute'
        if self.terminated:
            self.env.reset()
            self.terminated = False
            return True
        return self.terminated

    """ Return the discount factor for this MDP """

    def get_discount_factor(self):
        return self.discount_factor

    def execute(self, state, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated or truncated
        return (tuple(observation), reward, terminated)

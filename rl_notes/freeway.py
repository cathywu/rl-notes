import gymnasium as gym
from mdp import MDP
from ale_wrapper import ALEWrapper

"""
An wrapper for the Freeway gymnasium problem that extracts
just the relevant state information based on the study
from https://arxiv.org/abs/2109.01220
"""


class Freeway(ALEWrapper):
    
    # The actions available
    ACTIONS = [0, 1, 2]

    # Meaningful names for actions (for debugging)
    MEANINGFUL_ACTIONS=("STAY", "UP", "DOWN")

    # The highest value in the original is 177 (https://arxiv.org/abs/2109.01220)
    ORIGINAL_Y_MAX=177

    # Byte 14 contains y position of agent (https://arxiv.org/abs/2109.01220)
    ORIGINAL_Y_POSITION_INDEX=14

    # The index for the Y position in the abstract state
    ABSTRACT_Y_POSITION_INDEX=0

    # The maximum Y position in the abstract state
    ABSTRACT_Y_MAX=ORIGINAL_Y_MAX

    def __init__(self, version="Freeway-ramDeterministic-v4", render_mode="human", discount_factor=1.0):
        super().__init__(version=version, render_mode=render_mode, discount_factor=discount_factor)

    def get_initial_state(self):
        observation, info = self.env.reset()
        abstract_state = self.observation_to_abstract_state(observation)
        return abstract_state


    def execute(self, state, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated or truncated
        abstract_state = self.observation_to_abstract_state(observation)
        return (abstract_state, reward)
    
    """ Convert an observation into an abstract state """

    def observation_to_abstract_state(self, observation):
        # y position of the player
        y_position = observation[Freeway.ORIGINAL_Y_POSITION_INDEX]

        # x positions of the cars is 108-117 (https://arxiv.org/abs/2109.01220)
        car_positions = observation[108:117+1]
        return tuple([y_position]+car_positions)

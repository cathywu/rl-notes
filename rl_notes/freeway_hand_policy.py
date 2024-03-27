from policy import Policy
from freeway_abstraction import FreewayAbstraction
""" A hand-coded policy for Freeway.
"""


class FreewayHandPolicy(Policy):
    def __init__(self):
        self.abstract = FreewayAbstraction(render_mode="rgb_array")

    def select_action(self, state):

        x_position = 46
        y_position = state[14]
        lane_height = 17
        lane_start_offset = 13
        
        lane_number = 0
        if y_position >= lane_start_offset:
            adjusted_y_position = y_position - lane_start_offset
            lane_number = y_position // lane_height

        if lane_number <= 5 and not (30 <= state[108+lane_number+1] <= 49):
            return 1
        elif lane_number > 5 and not (40 <= state[108+lane_number+1] <= 60):
            return 1
        elif lane_number <= 5 and (30 <= state[108+lane_number] <= 49):
            return 2
        elif lane_number > 5 and (40 <= state[108+lane_number] <= 60):
            return 2
        return 0
        
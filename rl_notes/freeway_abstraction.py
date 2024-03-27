from freeway import Freeway

"""
An abstraction for the Freeway gymnasium problem.
"""


class FreewayAbstraction(Freeway):

    # The X position of the player is 46 (https://arxiv.org/abs/2109.01220)
    X_POSITION = 46

    # if a car is a certain THRESHOLD away, it becomes irrelevant, so use the threshold value
    THRESHOLD = 30

    def __init__(
        self,
        version="Freeway-ramDeterministic-v4",
        render_mode="human",
        discount_factor=1.0,
    ):
        super().__init__(
            version=version, render_mode=render_mode, discount_factor=discount_factor
        )

    def observation_to_abstract_state(self, observation):

        # get the concrete state
        state = super().observation_to_abstract_state(observation)
        
        # abstract the position of the player into a lane
        # Lane i lane starts at 16i + 13, and ends at 16(i + 1) + 13 (https://arxiv.org/abs/2109.01220)
        lane_height = 17
        lane_start_offset = 13
        y_position = state[Freeway.ABSTRACT_Y_POSITION_INDEX]

        lane_number = 0
        if y_position >= lane_start_offset:
            adjusted_y_position = y_position - lane_start_offset
            lane_number = y_position // lane_height


        # the relative position of the cars in the same lane, above lane, and below lane
        car_positions_left_to_right = [
            FreewayAbstraction.X_POSITION - car_y
            if 0 <= FreewayAbstraction.X_POSITION - car_y <= THRESHOLD
            else THRESHOLD
            for car_y in state[lane_number-1:lane_number+1]
        ]
        car_positions_right_to_left = [
            car_y - FreewayAbstraction.X_POSITION
            if 0 <= car_y - FreewayAbstraction.X_POSITION <= THRESHOLD
            else THRESHOLD
            for car_y in state[lane_number-1:lane_number+1]
        ]
        '''
        car_positions_left_to_right = [
            True
            if 0 <= FreewayAbstraction.X_POSITION - car_y <= THRESHOLD
            else False
            for car_y in state[lane_number-1:lane_number+1]
        ]
        car_positions_right_to_left = [
            True
            if 0 <= car_y - FreewayAbstraction.X_POSITION <= THRESHOLD
            else False
            for car_y in state[lane_number-1:lane_number+1]
        ]    
        '''
        return tuple(
            [y_position] + car_positions_left_to_right + car_positions_right_to_left
        )

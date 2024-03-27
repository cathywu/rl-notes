from potential_function import PotentialFunction
from gridworld import GridWorld


class GridWorldPotentialFunction(PotentialFunction):
    def __init__(self, mdp):
        self.mdp = mdp

    def get_potential(self, state):
        if state != GridWorld.TERMINAL:
            goal = (self.mdp.width, self.mdp.height)
            x = 0
            y = 1
            return 0.1 * (
                1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y]))
            )
        else:
            return 0.0

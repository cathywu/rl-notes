from potential_function import PotentialFunction
from freeway_abstraction import FreewayAbstraction

class FreewayPotentialFunction(PotentialFunction):
    def __init__(self, mdp):
        self.mdp = mdp
        self.visited = set()

    def get_potential(self, state):
        y_position = state[FreewayAbstraction.ABSTRACT_Y_POSITION_INDEX]
        if y_position in self.visited:
            return 0.0
        self.visited.add(y_position)
        return 1 / FreewayAbstraction.ABSTRACT_Y_MAX
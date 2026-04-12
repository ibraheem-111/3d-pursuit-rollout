from .base import Agent

class RandomWalkEvaderAgent(Agent):
    def __init__(self, name, position):
        super().__init__(name, position)

    def choose_action(self, grid):
        pass

class EvasiveEvaderAgent(Agent):
    def __init__(self, name, position):
        super().__init__(name, position)

    def choose_action(self, grid, pursuers):
        """ Moves away from the closest pursuer. """
        pass

    def _find_closest_pursuer(self, pursuers):
        """ Helper method to find the closest pursuer. """
        l2_distances = 
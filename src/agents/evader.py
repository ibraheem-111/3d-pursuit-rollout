from src.agents.base import Agent
import numpy as np

class RandomWalkEvaderAgent(Agent):
    def __init__(self, name, position):
        super().__init__(name, position)

    def choose_action(self, grid, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)

        valid_moves = grid.get_valid_moves_array(self.position)
        if valid_moves.shape[0] == 0:
            return self.position

        move_idx = int(rng.integers(0, valid_moves.shape[0]))
        move = valid_moves[move_idx]
        return type(self.position)(x=int(move[0]), y=int(move[1]), z=int(move[2]))

class EvasiveEvaderAgent(Agent):
    def __init__(self, name, position):
        super().__init__(name, position)

    def choose_action(self, grid, pursuers):
        """ Moves away from the closest pursuer. """
        pass

    def _find_closest_pursuer(self, pursuers):
        """ Helper method to find the closest pursuer. """
        l2_distances = [pursuer.position.l2_distance(self.position) for pursuer in pursuers]
        closest_index = l2_distances.index(min(l2_distances))
        return pursuers[closest_index]
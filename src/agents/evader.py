from src.agents.base import Agent
import numpy as np
from src.utils.math_utils import distance_matrix

class RandomWalkEvaderAgent(Agent):
    def __init__(self, name, position, agent_id):
        super().__init__(name, position, agent_id)

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
    def __init__(self, name, position, agent_id):
        super().__init__(name, position, agent_id)

    def choose_action(self, grid, pursuers):
        """ Moves away from the closest pursuer. """
        distances = distance_matrix([p.position for p in pursuers], [self.position])
        closest_pursuer_idx = np.argmin(distances)
        closest_pursuer = pursuers[closest_pursuer_idx]

        valid_moves = grid.get_valid_moves_array(self.position)

        target_distances = distance_matrix([closest_pursuer.position], valid_moves)
        best_move_idx = np.argmax(target_distances)
        best_move = valid_moves[best_move_idx]
        return type(self.position)(x=int(best_move[0]), y=int(best_move[1]), z=int(best_move[2]))

    
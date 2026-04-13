import numpy as np
from src.agents.base import Agent
from src.data_types.postion import Position
from src.utils.math_utils import distance_matrix


class GreedyAgent(Agent):
    def __init__(self, name, position: Position, agent_id):
        super().__init__(name, position, agent_id)

    def choose_action(self, grid, target_position: Position):
        """ Chooses the move that minimizes the L2 distance to the target. """
        valid_moves = grid.get_valid_moves_array(self.position, self.agent_id)
        if valid_moves.shape[0] == 0:
            return self.position

        target_distances = distance_matrix([np.array(target_position.as_tuple())], valid_moves)

        best_move_idx = np.argmin(target_distances)
        best_move = valid_moves[best_move_idx]
        return type(self.position)(x=int(best_move[0]), y=int(best_move[1]), z=int(best_move[2]))
import numpy as np
from src.agents.base import Agent
from src.data_types import Position, GameState
from src.utils.math_utils import distance_matrix


class GreedyAgent(Agent):
    def __init__(self, name, position: Position, agent_id, role):
        super().__init__(name, position, agent_id, role)

    def choose_action(self, grid, **kwargs):
        """ Chooses the move that minimizes the L2 distance to the target. """
        target_position: Position = kwargs["target_position"]
        valid_moves = grid.get_valid_moves_array(self.position, self.agent_id)
        if valid_moves.shape[0] == 0:
            return self.position

        target_distances = distance_matrix([np.array(target_position.as_tuple())], valid_moves)

        best_move_idx = np.argmin(target_distances)
        best_move = valid_moves[best_move_idx]
        return type(self.position)(x=int(best_move[0]), y=int(best_move[1]), z=int(best_move[2]))
            
    def choose_action_from_state(self, current_position, target_position, grid_model, pursuer_positions):
        valid_moves = grid_model.get_valid_moves(
            position=current_position,
            agent_id=self.agent_id,
            occupied_positions=pursuer_positions,
            evader_position=target_position,
        )
        if len(valid_moves) == 0:
            return current_position

        valid_moves_array = np.array([p.as_tuple() for p in valid_moves], dtype=int)
        target_distances = distance_matrix([np.array(target_position.as_tuple())], valid_moves_array)
        best_move_idx = np.argmin(target_distances)
        best_move = valid_moves_array[best_move_idx]
        return Position(x=int(best_move[0]), y=int(best_move[1]), z=int(best_move[2]))
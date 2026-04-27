from src.agents.base import Agent
import numpy as np
from src.utils.math_utils import distance_matrix
from src.data_types.postion import Position

class RandomWalkEvaderAgent(Agent):
    def __init__(self, name, position, agent_id, role):
        super().__init__(name, position, agent_id, role)

    def choose_action(self, grid, **kwargs):
        rng = kwargs.get("rng")
        if rng is None:
            raise ValueError("RandomWalkEvaderAgent requires rng in choose_action kwargs")

        valid_moves = grid.get_valid_moves_array(self.position, self.agent_id)
        if valid_moves.shape[0] == 0:
            return self.position

        move_idx = int(rng.integers(0, valid_moves.shape[0]))
        move = valid_moves[move_idx]
        return Position(x=int(move[0]), y=int(move[1]), z=int(move[2]))
    def choose_action_from_state(
        self,
        current_position,
        grid_model,
        pursuer_positions,
        rng=None,
        evader_positions=None,
        pursuer_agent_ids=None,
    ):
        if rng is None:
            rng = np.random.default_rng(0)

        valid_moves = grid_model.get_valid_moves(
            position=current_position,
            agent_id=self.agent_id,
            occupied_positions=pursuer_positions,
            evader_position=current_position,
            evader_positions=evader_positions,
            occupied_agent_ids=pursuer_agent_ids,
        )
        if len(valid_moves) == 0:
            return current_position

        move_idx = int(rng.integers(0, len(valid_moves)))
        return valid_moves[move_idx]

class EvasiveEvaderAgent(Agent):
    def __init__(self, name, position, agent_id, role):
        super().__init__(name, position, agent_id, role)

    def _choose_random_move(self, grid):
        rng = np.random.default_rng()
        valid_moves = grid.get_valid_moves_array(self.position, self.agent_id)
        if valid_moves.shape[0] == 0:
            return self.position

        move_idx = int(rng.integers(0, valid_moves.shape[0]))
        move = valid_moves[move_idx]
        return Position(x=int(move[0]), y=int(move[1]), z=int(move[2]))

    def choose_action(self, grid, **kwargs):
        """ Moves away from the closest pursuer. """
        pursuers = kwargs["pursuers"]
        if len(pursuers) == 0:
            return self._choose_random_move(grid)

        valid_moves = grid.get_valid_moves_array(self.position, self.agent_id)
        if valid_moves.shape[0] == 0:
            return self.position

        distances = distance_matrix(
            [np.array(p.position.as_tuple()) for p in pursuers],
            [np.array(self.position.as_tuple())],
        )
        closest_pursuer_idx = np.argmin(distances)
        closest_pursuer = pursuers[closest_pursuer_idx]

        target_distances = distance_matrix(
            [np.array(closest_pursuer.position.as_tuple())],
            valid_moves,
        )
        best_move_idx = np.argmax(target_distances)
        best_move = valid_moves[best_move_idx]
        return Position(x=int(best_move[0]), y=int(best_move[1]), z=int(best_move[2]))

    def choose_action_from_state(
        self,
        current_position,
        grid_model,
        pursuer_positions,
        rng=None,
        evader_positions=None,
        pursuer_agent_ids=None,
    ):
        valid_moves = grid_model.get_valid_moves(
            position=current_position,
            agent_id=self.agent_id,
            occupied_positions=pursuer_positions,
            evader_position=current_position,
            evader_positions=evader_positions,
            occupied_agent_ids=pursuer_agent_ids,
        )
        if len(valid_moves) == 0:
            return current_position

        valid_moves_array = np.array([p.as_tuple() for p in valid_moves], dtype=int)
        distances = distance_matrix(
            [np.array(pursuer_position.as_tuple()) for pursuer_position in pursuer_positions],
            [np.array(current_position.as_tuple())],
        )
        closest_pursuer_idx = np.argmin(distances)
        closest_pursuer_position = pursuer_positions[closest_pursuer_idx]

        target_distances = distance_matrix(
            [np.array(closest_pursuer_position.as_tuple())],
            valid_moves_array,
        )
        best_move_idx = np.argmax(target_distances)
        best_move = valid_moves_array[best_move_idx]
        return Position(x=int(best_move[0]), y=int(best_move[1]), z=int(best_move[2]))

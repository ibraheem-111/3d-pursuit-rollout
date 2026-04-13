import numpy as np
from src.data_types.postion import Position


CARDINAL_MOVES = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=int,
)

class Grid3D:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.grid = np.zeros((width, height, depth), dtype=int)

    def is_within_bounds(self, position: Position):
        return (0 <= position.x < self.width) and (0 <= position.y < self.height) and (0 <= position.z < self.depth)

    def is_occupied(self, position: Position):
        return self.grid[position.x, position.y, position.z] != 0

    def get_valid_moves_array(self, position: Position, agent_id: int):
        current = np.array([position.x, position.y, position.z], dtype=int)
        candidate_positions = current + CARDINAL_MOVES

        in_bounds_mask = (
            (candidate_positions[:, 0] >= 0)
            & (candidate_positions[:, 0] < self.width)
            & (candidate_positions[:, 1] >= 0)
            & (candidate_positions[:, 1] < self.height)
            & (candidate_positions[:, 2] >= 0)
            & (candidate_positions[:, 2] < self.depth)
        )

        bounded_candidates = candidate_positions[in_bounds_mask]
        if bounded_candidates.size == 0:
            return np.empty((0, 3), dtype=int)

        occupancy = self.grid[
            bounded_candidates[:, 0],
            bounded_candidates[:, 1],
            bounded_candidates[:, 2],
        ]
        if agent_id == 1:
            return bounded_candidates[occupancy == 0]
        if agent_id > 1:
            return bounded_candidates[(occupancy == 0) | (occupancy == 1)]

        raise ValueError(f"Invalid agent_id: {agent_id}")

    def get_valid_moves(self, position: Position, agent_id: int):
        valid_moves_array = self.get_valid_moves_array(position, agent_id=agent_id)
        return [
            Position(x=int(x), y=int(y), z=int(z))
            for x, y, z in valid_moves_array
        ]
    
    def place_agent(self, position: Position, agent_id: int):
        if self.is_within_bounds(position) and not self.is_occupied(position):
            self.grid[position.x, position.y, position.z] = agent_id
            return True
        return False

    def move_agent(self, old_position: Position, new_position: Position, agent_id: int):
        if not self.is_within_bounds(new_position):
            return False

        target_occupancy = self.grid[new_position.x, new_position.y, new_position.z]

        if agent_id == 1:
            can_move = target_occupancy == 0
        elif agent_id > 1:
            can_move = (target_occupancy == 0) or (target_occupancy == 1)
        else:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        if can_move:
            self.grid[old_position.x, old_position.y, old_position.z] = 0  # Clear old position
            self.grid[new_position.x, new_position.y, new_position.z] = agent_id  # Place agent in new position
            return True

        return False

    def remove_agent(self, position: Position):
        if self.is_within_bounds(position):
            self.grid[position.x, position.y, position.z] = 0  # Clear the position
            return True
        return False

    def get_agent_id(self, position: Position):
        if self.is_within_bounds(position):
            return self.grid[position.x, position.y, position.z]
        return None
import numpy as np
from src.agents.base import AgentRole
from src.data_types import Position, GameState


CARDINAL_MOVES = np.array(
    [
        [0, 0, 0],
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
        self.agent_roles_by_id: dict[int, AgentRole] = {}

    def _get_agent_role(self, agent_id: int) -> AgentRole:
        role = self.agent_roles_by_id.get(int(agent_id))
        if role is None:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        return role  

    def is_within_bounds(self, position: Position):
        return (0 <= position.x < self.width) and (0 <= position.y < self.height) and (0 <= position.z < self.depth)

    def is_occupied(self, position: Position):
        return self.grid[position.x, position.y, position.z] != 0

    def get_valid_moves_array(self, position: Position, agent_id: int):
        mover_role = self._get_agent_role(agent_id)

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

        if mover_role == AgentRole.EVADER:
            return bounded_candidates[occupancy == 0]
        if mover_role == AgentRole.PURSUER:
            valid_mask = occupancy == 0
            occupied_mask = occupancy != 0
            if np.any(occupied_mask):
                occupied_ids = occupancy[occupied_mask].astype(int)
                occupied_roles = np.array(
                    [self._get_agent_role(occupied_id) for occupied_id in occupied_ids],
                    dtype=object,
                )
                valid_mask[occupied_mask] = occupied_roles == AgentRole.EVADER
            return bounded_candidates[valid_mask]

        raise ValueError(f"Unknown mover role for agent_id={agent_id}: {mover_role}")

    def get_valid_moves(self, position: Position, agent_id: int):
        valid_moves_array = self.get_valid_moves_array(position, agent_id=agent_id)
        return [
            Position(x=int(x), y=int(y), z=int(z))
            for x, y, z in valid_moves_array
        ]
    
    def place_agent(self, position: Position, agent_id: int, role: AgentRole):
        known_role = self.agent_roles_by_id.get(int(agent_id))
        if known_role is not None and known_role != role:
            raise ValueError(f"Agent {agent_id} cannot change role from {known_role} to {role}")

        if self.is_within_bounds(position) and not self.is_occupied(position):
            self.agent_roles_by_id[int(agent_id)] = role
            self.grid[position.x, position.y, position.z] = agent_id
            return True
        return False

    def move_agent(self, old_position: Position, new_position: Position, agent_id: int):
        mover_role = self._get_agent_role(agent_id)

        if self.grid[old_position.x, old_position.y, old_position.z] != agent_id:
            raise ValueError(f"Agent {agent_id} is not at position {old_position}")

        if not self.is_within_bounds(new_position):
            return False

        target_agent_id = int(self.grid[new_position.x, new_position.y, new_position.z])

        if mover_role == AgentRole.EVADER:
            can_move = target_agent_id == 0
        elif mover_role == AgentRole.PURSUER:
            if target_agent_id == 0:
                can_move = True
            else:
                target_role = self._get_agent_role(target_agent_id)
                can_move = target_role == AgentRole.EVADER
        else:
            raise ValueError(f"Unknown mover role for agent_id={agent_id}: {mover_role}")

        if can_move:
            if target_agent_id != 0:
                self.agent_roles_by_id.pop(target_agent_id, None)
            self.grid[old_position.x, old_position.y, old_position.z] = 0  # Clear old position
            self.grid[new_position.x, new_position.y, new_position.z] = agent_id  # Place agent in new position
            return True

        return False

    def remove_agent(self, position: Position):
        if self.is_within_bounds(position):
            agent_id = int(self.grid[position.x, position.y, position.z])
            if agent_id != 0:
                self.agent_roles_by_id.pop(agent_id, None)
            self.grid[position.x, position.y, position.z] = 0  # Clear the position
            return True
        return False

    def get_agent_id(self, position: Position):
        if self.is_within_bounds(position):
            return self.grid[position.x, position.y, position.z]
        return None
from __future__ import annotations
from typing import List
from src.grid import Grid3D
from src.data_types.postion import Position
from src.data_types.state import GameState
from src.agents.base import AgentRole



class GridModel:
    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth

    def is_capture(self, state: GameState) -> bool:
        return any(p == state.evader_position for p in state.pursuer_positions)

    def stage_cost(self, state: GameState) -> float:
        return 0.0 if self.is_capture(state) else 1.0

    def get_valid_moves(self, position: Position, agent_id: int, occupied_positions: List[Position], evader_position: Position):
        """
        Reconstruct a temporary grid to reuse your Grid3D move logic.
        """
        temp_grid = Grid3D(self.width, self.height, self.depth)

        # place evader
        temp_grid.place_agent(evader_position, agent_id=1, role=AgentRole.EVADER)

        # place pursuers
        next_id = 2
        for p in occupied_positions:
            if p != evader_position:
                placed = temp_grid.place_agent(p, agent_id=next_id, role=AgentRole.PURSUER)
                if not placed:
                    temp_grid.agent_roles_by_id[int(next_id)] = AgentRole.PURSUER
                next_id += 1

        mover_agent_id = int(agent_id)
        if mover_agent_id not in temp_grid.agent_roles_by_id:
            temp_grid.agent_roles_by_id[mover_agent_id] = AgentRole.PURSUER if mover_agent_id != 1 else AgentRole.EVADER

        return temp_grid.get_valid_moves(position, agent_id=agent_id)

    def transition(
        self,
        state: GameState,
        pursuer_next_positions: List[Position],
        evader_next_position: Position,
    ) -> GameState:
        return GameState(
            pursuer_positions=tuple(pursuer_next_positions),
            evader_position=evader_next_position,
            step_idx=state.step_idx + 1,
        )
from __future__ import annotations
from typing import List, Optional
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
        if len(state.evader_positions) == 0:
            return True
        return all(
            any(p == evader_position for p in state.pursuer_positions)
            for evader_position in state.evader_positions
        )

    def stage_cost(self, state: GameState) -> float:
        return 0.0 if self.is_capture(state) else 1.0

    def get_valid_moves(
        self,
        position: Position,
        agent_id: int,
        occupied_positions: List[Position],
        evader_position: Optional[Position] = None,
        evader_positions: Optional[List[Position]] = None,
        occupied_agent_ids: Optional[List[int]] = None,
    ):
        """
        Reconstruct a temporary grid to reuse your Grid3D move logic.
        """
        temp_grid = Grid3D(self.width, self.height, self.depth)

        if evader_positions is None:
            evader_positions = [] if evader_position is None else [evader_position]

        # Place active evaders. Use the mover's real id for its current cell so
        # no-op moves stay valid; negative ids safely stand in for other evaders.
        for idx, active_evader_position in enumerate(evader_positions):
            evader_agent_id = int(agent_id) if active_evader_position == position else -(idx + 1)
            placed = temp_grid.place_agent(
                active_evader_position,
                agent_id=evader_agent_id,
                role=AgentRole.EVADER,
            )
            if not placed:
                temp_grid.agent_roles_by_id[int(evader_agent_id)] = AgentRole.EVADER

        # place pursuers
        next_id = len(evader_positions) + 1
        used_ids = set(temp_grid.agent_roles_by_id)
        for idx, p in enumerate(occupied_positions):
            if any(p == active_evader_position for active_evader_position in evader_positions):
                continue

            if occupied_agent_ids is not None and idx < len(occupied_agent_ids):
                pursuer_agent_id = int(occupied_agent_ids[idx])
            else:
                while next_id in used_ids:
                    next_id += 1
                pursuer_agent_id = next_id
                next_id += 1

            used_ids.add(pursuer_agent_id)
            placed = temp_grid.place_agent(p, agent_id=pursuer_agent_id, role=AgentRole.PURSUER)
            if not placed:
                temp_grid.agent_roles_by_id[int(pursuer_agent_id)] = AgentRole.PURSUER

        mover_agent_id = int(agent_id)
        if mover_agent_id not in temp_grid.agent_roles_by_id:
            temp_grid.agent_roles_by_id[mover_agent_id] = AgentRole.PURSUER if mover_agent_id != 1 else AgentRole.EVADER

        return temp_grid.get_valid_moves(position, agent_id=agent_id)

    def transition(
        self,
        state: GameState,
        pursuer_next_positions: List[Position],
        evader_next_positions,
    ) -> GameState:
        if isinstance(evader_next_positions, Position):
            evader_next_positions = [evader_next_positions]

        active_evader_positions = tuple(
            evader_position
            for evader_position in evader_next_positions
            if not any(pursuer_position == evader_position for pursuer_position in pursuer_next_positions)
        )

        return GameState(
            pursuer_positions=tuple(pursuer_next_positions),
            evader_positions=active_evader_positions,
            step_idx=state.step_idx + 1,
        )

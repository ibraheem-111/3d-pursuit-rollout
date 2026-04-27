from dataclasses import dataclass
from typing import Optional, Tuple

from src.data_types.postion import Position

@dataclass(frozen=True, slots=True, init=False)
class GameState:
    pursuer_positions: Tuple[Position, ...]
    evader_positions: Tuple[Position, ...]
    step_idx: int = 0

    def __init__(
        self,
        pursuer_positions: Tuple[Position, ...],
        evader_position: Optional[Position] = None,
        evader_positions: Optional[Tuple[Position, ...]] = None,
        step_idx: int = 0,
    ):
        if evader_positions is None:
            if evader_position is None:
                raise ValueError("GameState requires evader_position or evader_positions")
            evader_positions = (evader_position,)

        object.__setattr__(self, "pursuer_positions", tuple(pursuer_positions))
        object.__setattr__(self, "evader_positions", tuple(evader_positions))
        object.__setattr__(self, "step_idx", step_idx)

    @property
    def evader_position(self) -> Position:
        if len(self.evader_positions) == 0:
            raise ValueError("GameState has no active evaders")
        return self.evader_positions[0]

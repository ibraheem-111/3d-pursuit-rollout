from src.data_types.postion import Position
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class GameState:
    pursuer_positions: Tuple[Position, ...]
    evader_position: Position
    step_idx: int = 0
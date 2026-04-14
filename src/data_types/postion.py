import math
from typing import Any
import numpy as np
from dataclasses import dataclass

@dataclass
class Position:
    x: int
    y: int
    z: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)
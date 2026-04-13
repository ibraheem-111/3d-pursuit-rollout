import math
from pydantic import BaseModel
from typing import Any
import numpy as np


class Position(BaseModel):
    x: int
    y: int
    z: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)
import math
from pydantic import BaseModel


class Position(BaseModel):
    x: int
    y: int
    z: int

    def l2_distance(self, other: "Position") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)
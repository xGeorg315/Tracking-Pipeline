from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class LaneBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "LaneBox":
        x_min, x_max, y_min, y_max, z_min, z_max = values
        return cls(
            x_min=min(x_min, x_max),
            x_max=max(x_min, x_max),
            y_min=min(y_min, y_max),
            y_max=max(y_min, y_max),
            z_min=min(z_min, z_max),
            z_max=max(z_min, z_max),
        )

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        if len(xyz) == 0:
            return np.zeros((0,), dtype=bool)
        return (
            (xyz[:, 0] >= self.x_min)
            & (xyz[:, 0] <= self.x_max)
            & (xyz[:, 1] >= self.y_min)
            & (xyz[:, 1] <= self.y_max)
            & (xyz[:, 2] >= self.z_min)
            & (xyz[:, 2] <= self.z_max)
        )

    def to_list(self) -> list[float]:
        return [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]

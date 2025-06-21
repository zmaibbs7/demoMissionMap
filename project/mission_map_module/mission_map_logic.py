"""Core logic for coverage marking and path tracking."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class MapLogic:
    """Coordinate transform and cell marking."""

    def __init__(self, resolution: float, origin: Tuple[float, float, float]) -> None:
        self.resolution = resolution
        self.origin = origin

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        return px, py

    def mark_cell(self, covered: np.ndarray, cell: Tuple[int, int]) -> None:
        x, y = cell
        if 0 <= y < covered.shape[0] and 0 <= x < covered.shape[1]:
            covered[y, x] = 255

    def add_path(self, path: List[Tuple[int, int]], cell: Tuple[int, int]) -> None:
        path.append(cell)

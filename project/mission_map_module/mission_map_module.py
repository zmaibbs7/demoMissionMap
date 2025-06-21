"""MissionMap module: manage map data and control interfaces."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from .mission_map_logic import MapLogic
from .mission_map_output import MapOutput


@dataclass
class MapMeta:
    """Store map resolution and origin."""

    resolution: float
    origin: Tuple[float, float, float]
    size: Tuple[int, int]


class MissionMap:
    """Load map, maintain data, and control recording."""

    def __init__(self) -> None:
        self.map_meta: Optional[MapMeta] = None
        self.origin_map: Optional[np.ndarray] = None
        self.covered: Optional[np.ndarray] = None
        self.path: List[Tuple[int, int]] = []
        self.logic: Optional[MapLogic] = None
        self.output = MapOutput()
        self.running = False
        logging.basicConfig(level=logging.INFO)

    # ---- Initialization & control ----
    def load_map(self, pgm_path: str, json_path: str) -> None:
        logging.info("Loading map from %s and %s", pgm_path, json_path)
        with Image.open(pgm_path) as pgm:
            self.origin_map = np.array(pgm)

        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        resolution = float(info.get("resolution", 0.05))
        origin = tuple(info.get("origin", [0.0, 0.0, 0.0]))  # type: ignore
        size = (int(info.get("width", self.origin_map.shape[1])),
                int(info.get("height", self.origin_map.shape[0])))

        self.map_meta = MapMeta(resolution=resolution, origin=origin, size=size)
        self.covered = np.zeros(self.origin_map.shape, dtype=np.uint8)
        self.path = []
        self.logic = MapLogic(resolution, origin)
        logging.info("Map loaded: size=%s", self.map_meta.size)

    def start(self) -> None:
        if not self.map_meta:
            raise RuntimeError("Map not loaded")
        self.covered = np.zeros_like(self.origin_map)
        self.path = []
        self.running = True
        logging.info("Recording started")

    def pause(self) -> None:
        if self.running:
            self.running = False
            logging.info("Recording paused")

    def resume(self) -> None:
        if not self.running:
            self.running = True
            logging.info("Recording resumed")

    def stop(self) -> None:
        """Stop recording without saving results."""
        self.running = False
        logging.info("Recording stopped")

    def export_results(self, output_dir: str) -> None:
        """Save maps and coverage report to *output_dir*."""
        if (
            self.origin_map is None
            or self.covered is None
            or self.map_meta is None
        ):
            raise RuntimeError("Map not loaded")
        os.makedirs(output_dir, exist_ok=True)
        self.output.save_coverage_map(
            os.path.join(output_dir, "sim_real_mowing.pgm"),
            self.origin_map,
            self.covered,
        )
        self.output.save_path_map(
            os.path.join(output_dir, "sim_robot_path.pgm"),
            self.path,
            self.map_meta.size,
        )
        self.output.save_combined_map(
            os.path.join(output_dir, "combined_map.jpg"),
            self.origin_map,
            self.covered,
            self.path,
        )
        self.output.save_report(
            os.path.join(output_dir, "coverage_report.json"), self.get_stats()
        )

    # ---- Pose update ----
    def update_pose(self, x: float, y: float, theta: float) -> None:
        if not self.running or not self.logic or self.covered is None:
            return
        pixel = self.logic.world_to_pixel(x, y)
        self.logic.mark_cell(self.covered, pixel)
        self.logic.add_path(self.path, pixel)

    # ---- Data access ----
    def get_stats(self) -> Dict[str, float]:
        if self.covered is None or self.origin_map is None:
            return {}
        covered_cells = np.count_nonzero(self.covered)
        total_cells = np.count_nonzero(self.origin_map != 0)
        coverage = (covered_cells / total_cells * 100) if total_cells else 0
        return {"coverage": coverage, "miss": 100 - coverage}

    def get_map(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.origin_map is None or self.covered is None:
            raise RuntimeError("Map not loaded")
        return self.origin_map.copy(), self.covered.copy()

    def get_path(self) -> List[Tuple[int, int]]:
        return list(self.path)

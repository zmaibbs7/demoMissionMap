"""Map output utilities for images and reports."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


class MapOutput:
    """Generate and save map images and JSON report."""

    def save_coverage_map(self, pgm_file: str,
                          base_map: np.ndarray,
                          covered: np.ndarray) -> None:
        img = Image.fromarray(np.maximum(base_map, covered).astype(np.uint8))
        img.save(pgm_file)

    def save_path_map(self, pgm_file: str,
                      path: List[Tuple[int, int]],
                      size: Tuple[int, int]) -> None:
        img = Image.new("L", size, 0)
        draw = ImageDraw.Draw(img)
        if path:
            if len(path) > 1:
                draw.line(path, fill=255, width=1)
            else:
                draw.point(path[0], fill=255)
        img.save(pgm_file)

    def save_combined_map(self, jpg_file: str,
                          base_map: np.ndarray,
                          covered: np.ndarray,
                          path: List[Tuple[int, int]]) -> None:
        base_rgb = Image.fromarray(base_map).convert("RGB")
        mask = Image.fromarray(covered)
        base_rgb.paste((0, 255, 0), mask=mask)
        draw = ImageDraw.Draw(base_rgb)
        if path:
            if len(path) > 1:
                draw.line(path, fill=(255, 0, 0), width=2)
            else:
                draw.point(path[0], fill=(255, 0, 0))
        base_rgb.save(jpg_file)

    def save_report(self, json_file: str,
                    stats: Dict[str, float]) -> None:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

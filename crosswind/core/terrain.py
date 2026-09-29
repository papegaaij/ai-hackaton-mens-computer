"""Destructible terrain stored as a boolean mask (mask[y, x] == True means solid)."""
from __future__ import annotations

import numpy as np

from crosswind import config


class Terrain:
    def __init__(self, width: int, height: int, rng: np.random.Generator):
        self.width = width
        self.height = height
        self.mask = self._generate(rng)
        self.version = 0  # bumped on every change so renderers can cache

    def _generate(self, rng: np.random.Generator) -> np.ndarray:
        xs = np.arange(self.width)
        base = self.height * 0.62
        heights = np.full(self.width, base)
        # Sum of random sine waves gives rolling alien hills.
        for amp, freq in ((90, 1.0), (45, 2.5), (18, 6.0), (6, 15.0)):
            phase = rng.uniform(0, 2 * np.pi)
            f = freq * rng.uniform(0.8, 1.2)
            heights += amp * np.sin(xs / self.width * 2 * np.pi * f + phase)
        heights = np.clip(heights, self.height * 0.3, self.height - config.BEDROCK - 20)
        ys = np.arange(self.height)[:, None]
        return ys >= heights[None, :]

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_solid(self, x: float, y: float) -> bool:
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.width):
            return False
        if yi >= self.height:
            return True
        if yi < 0:
            return False
        return bool(self.mask[yi, xi])

    def surface_y(self, x: float) -> int:
        """Y of the topmost solid pixel in column x."""
        xi = int(np.clip(x, 0, self.width - 1))
        col = self.mask[:, xi]
        idx = np.argmax(col)
        return int(idx) if col[idx] else self.height

    def carve_circle(self, cx: float, cy: float, r: float) -> None:
        x0, x1 = max(0, int(cx - r)), min(self.width, int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(self.height - config.BEDROCK, int(cy + r) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        hole = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        self.mask[y0:y1, x0:x1] &= ~hole
        self.version += 1

"""Wind: a base value that drifts constantly toward random targets, stronger at altitude."""
import math

import numpy as np

from crosswind import config


class Wind:
    """Base wind acceleration at ground level (px/s^2). Positive blows to the right.

    Every WIND_SHIFT_MIN..WIND_SHIFT_MAX seconds a new random target is picked; the wind eases toward
    it, so it never holds still for long and can turn while a shot is in the air.
    """

    def __init__(self, rng: np.random.Generator, strength: float = 1.0):
        self.rng = rng
        self.strength = strength  # fraction of WIND_MAX the wind can reach (AI training starts with less)
        self.value = self._random_value()
        self.target = self._random_value()
        self._shift_in = self._random_interval()

    def _random_value(self) -> float:
        return float(self.rng.uniform(-config.WIND_MAX, config.WIND_MAX)) * self.strength

    def _random_interval(self) -> float:
        return float(self.rng.uniform(config.WIND_SHIFT_MIN, config.WIND_SHIFT_MAX))

    def step(self, dt: float) -> None:
        self._shift_in -= dt
        if self._shift_in <= 0:
            self.target = self._random_value()
            self._shift_in = self._random_interval()
        self.value += (self.target - self.value) * (1.0 - math.exp(-dt / config.WIND_RESPONSE))


def wind_at(base: float, y: float, height: int) -> float:
    altitude = min(max(1.0 - y / height, 0.0), 1.0)
    return base * (1.0 + config.WIND_ALTITUDE_BOOST * altitude)

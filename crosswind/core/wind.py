"""Wind: one base value per turn, stronger at altitude."""
import numpy as np

from crosswind import config


def new_wind(rng: np.random.Generator) -> float:
    """Base wind acceleration at ground level (px/s^2). Positive blows to the right."""
    return float(rng.uniform(-config.WIND_MAX, config.WIND_MAX))


def wind_at(base: float, y: float, height: int) -> float:
    altitude = np.clip(1.0 - y / height, 0.0, 1.0)
    return base * (1.0 + config.WIND_ALTITUDE_BOOST * altitude)

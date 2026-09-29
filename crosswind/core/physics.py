"""Point-mass projectile integration with gravity, altitude-scaled wind and drag."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from crosswind import config
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import Weapon
from crosswind.core.wind import wind_at


@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    weapon: Weapon
    owner: int
    trail: list[tuple[float, float]] = field(default_factory=list)


def launch(x: float, y: float, angle: float, power: float, weapon: Weapon, owner: int) -> Projectile:
    speed = power / 100.0 * config.MAX_LAUNCH_SPEED * weapon.speed_factor
    rad = math.radians(angle)
    # Screen y grows downward, so "up" is negative vy.
    return Projectile(x, y, speed * math.cos(rad), -speed * math.sin(rad), weapon, owner)


def step(p: Projectile, base_wind: float, height: int, dt: float) -> None:
    w = p.weapon
    ax = wind_at(base_wind, p.y, height) / w.mass - w.drag * p.vx
    ay = config.GRAVITY - w.drag * p.vy
    p.vx += ax * dt
    p.vy += ay * dt
    p.x += p.vx * dt
    p.y += p.vy * dt


def hits_terrain(p: Projectile, terrain: Terrain) -> bool:
    return terrain.is_solid(p.x, p.y)


def out_of_bounds(p: Projectile, terrain: Terrain) -> bool:
    # Allow flying above the screen; lost when leaving left/right or below.
    return p.x < -200 or p.x > terrain.width + 200 or p.y > terrain.height + 50

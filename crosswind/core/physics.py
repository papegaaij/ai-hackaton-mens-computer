"""Point-mass projectile integration with gravity, altitude-scaled wind and drag."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from crosswind import config
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import Weapon
from crosswind.core.wind import wind_at

DRILL_DRAG = 3.0        # extra drag (1/s) while a driller bores through rock
BOUNCE_FRICTION = 0.6   # fraction of the speed along the ground kept on a real bounce
ROLL_FRICTION = 3.0     # drag (1/s) along the ground while rolling


@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    weapon: Weapon
    owner: int
    trail: list[tuple[float, float]] = field(default_factory=list)
    age: float = 0.0                # seconds since launch
    timer: float | None = None      # driller/bouncer: seconds left until it explodes, once it touched ground


def launch(x: float, y: float, angle: float, power: float, weapon: Weapon, owner: int) -> Projectile:
    speed = power / 100.0 * config.MAX_LAUNCH_SPEED * weapon.speed_factor
    rad = math.radians(angle)
    # Screen y grows downward, so "up" is negative vy.
    return Projectile(x, y, speed * math.cos(rad), -speed * math.sin(rad), weapon, owner)


def step(p: Projectile, base_wind: float, height: int, dt: float) -> None:
    w = p.weapon
    drag = w.drag + (DRILL_DRAG if w.drill and p.timer is not None else 0.0)
    ax = wind_at(base_wind, p.y, height) / w.mass - drag * p.vx
    ay = config.GRAVITY - drag * p.vy
    if p.age < w.burn:
        speed = math.hypot(p.vx, p.vy)
        if speed > 0:
            ax += w.thrust * p.vx / speed
            ay += w.thrust * p.vy / speed
    p.vx += ax * dt
    p.vy += ay * dt
    p.x += p.vx * dt
    p.y += p.vy * dt
    p.age += dt


def split(p: Projectile) -> list[Projectile]:
    """Break a cluster projectile into its children, fanned out horizontally."""
    w = p.weapon
    assert w.split_into is not None
    mid = (w.split - 1) / 2
    return [Projectile(p.x, p.y, p.vx + w.split_spread * (i - mid), p.vy, w.split_into, p.owner, list(p.trail), p.age)
            for i in range(w.split)]


def bounce(p: Projectile, terrain: Terrain, prev_x: float, prev_y: float, dt: float) -> None:
    """Move back out of the ground and reflect the velocity off the local surface."""
    p.x, p.y = prev_x, prev_y
    slope = (terrain.surface_y(p.x + 3) - terrain.surface_y(p.x - 3)) / 6
    length = math.hypot(slope, 1.0)
    nx, ny = slope / length, -1.0 / length  # surface normal, pointing out of the ground
    vn = p.vx * nx + p.vy * ny
    if vn >= 0:
        return
    tx, ty = p.vx - vn * nx, p.vy - vn * ny
    keep = BOUNCE_FRICTION if -vn > 30 else max(0.0, 1.0 - ROLL_FRICTION * dt)
    p.vx = tx * keep - p.weapon.bounce * vn * nx
    p.vy = ty * keep - p.weapon.bounce * vn * ny


def hits_terrain(p: Projectile, terrain: Terrain) -> bool:
    return terrain.is_solid(p.x, p.y)


def out_of_bounds(p: Projectile, terrain: Terrain) -> bool:
    # Allow flying above the screen; lost when leaving left/right or below.
    return p.x < -200 or p.x > terrain.width + 200 or p.y > terrain.height + 50

"""The simulation. Pure Python + numpy, no pygame: runs headless for tests and future RL."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from crosswind import config
from crosswind.core import physics
from crosswind.core.actions import FireAction
from crosswind.core.player import Player
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import WEAPONS
from crosswind.core.wind import new_wind


class Phase(Enum):
    AIMING = auto()
    IN_FLIGHT = auto()
    RESOLVING = auto()
    GAME_OVER = auto()


@dataclass
class Explosion:
    x: float
    y: float
    radius: float


class Game:
    def __init__(self, seed: int | None = None, width: int = config.WIDTH, height: int = config.HEIGHT):
        self.rng = np.random.default_rng(seed)
        self.terrain = Terrain(width, height, self.rng)
        self.players = [
            self._spawn(0, width * 0.15, 45.0),
            self._spawn(1, width * 0.85, 135.0),
        ]
        self.current = 0
        self.turn = 1
        self.wind = new_wind(self.rng)
        self.phase = Phase.AIMING
        self.projectile: physics.Projectile | None = None
        self.winner: int | None = None
        self.events: list[Explosion] = []  # consumed by the renderer
        self._resolve_timer = 0.0
        self._accum = 0.0

    def _spawn(self, index: int, x: float, angle: float) -> Player:
        p = Player(index, x, 0.0, angle)
        self._settle(p)
        return p

    # ---- queries -------------------------------------------------------
    @property
    def active(self) -> Player:
        return self.players[self.current]

    def barrel_tip(self, p: Player) -> tuple[float, float]:
        rad = math.radians(p.angle)
        length = config.PLAYER_RADIUS + 8
        return p.x + math.cos(rad) * length, p.y - config.PLAYER_RADIUS * 0.5 - math.sin(rad) * length

    # ---- input API (used by controllers) ---------------------------------
    def adjust_aim(self, d_angle: float, d_power: float) -> None:
        if self.phase is not Phase.AIMING:
            return
        p = self.active
        p.angle = float(np.clip(p.angle + d_angle, 0, 180))
        p.power = float(np.clip(p.power + d_power, 5, 100))

    def cycle_weapon(self, step: int) -> None:
        if self.phase is Phase.AIMING:
            self.active.weapon = (self.active.weapon + step) % len(WEAPONS)

    def move(self, direction: int, dt: float) -> None:
        """Drive the active tank along the ground; costs fuel, blocked by steep slopes."""
        if self.phase is not Phase.AIMING or direction == 0:
            return
        p = self.active
        dist = min(config.PLAYER_SPEED * dt, p.fuel)
        if dist <= 0:
            return
        nx = float(np.clip(p.x + direction * dist, config.PLAYER_RADIUS, self.terrain.width - config.PLAYER_RADIUS))
        new_ground = self.terrain.surface_y(nx)
        if (p.y + config.PLAYER_RADIUS) - new_ground > config.MAX_CLIMB * max(dist, 1):
            return
        p.fuel -= abs(nx - p.x)
        p.x = nx
        self._settle(p)

    def fire(self, action: FireAction) -> bool:
        if self.phase is not Phase.AIMING:
            return False
        p = self.active
        p.angle = float(np.clip(action.angle, 0, 180))
        p.power = float(np.clip(action.power, 5, 100))
        p.weapon = action.weapon % len(WEAPONS)
        tx, ty = self.barrel_tip(p)
        self.projectile = physics.launch(tx, ty, p.angle, p.power, WEAPONS[p.weapon], p.index)
        self.phase = Phase.IN_FLIGHT
        return True

    # ---- simulation ----------------------------------------------------
    def update(self, dt: float) -> None:
        """Advance real time; physics runs in fixed PHYSICS_DT substeps."""
        self._accum += dt
        while self._accum >= config.PHYSICS_DT:
            self._accum -= config.PHYSICS_DT
            self._tick(config.PHYSICS_DT)

    def _tick(self, dt: float) -> None:
        if self.phase is Phase.IN_FLIGHT:
            self._tick_projectile(dt)
        elif self.phase is Phase.RESOLVING:
            self._resolve_timer -= dt
            if self._resolve_timer <= 0:
                self._next_turn()

    def _tick_projectile(self, dt: float) -> None:
        p = self.projectile
        assert p is not None
        physics.step(p, self.wind, self.terrain.height, dt)
        if not p.trail or math.dist(p.trail[-1], (p.x, p.y)) > 4:
            p.trail.append((p.x, p.y))
        hit_player = any(
            pl.alive and math.dist((pl.x, pl.y), (p.x, p.y)) <= config.PLAYER_RADIUS
            and not (pl.index == p.owner and len(p.trail) < 6)
            for pl in self.players
        )
        if hit_player or physics.hits_terrain(p, self.terrain):
            self._explode(p.x, p.y, p.weapon.blast_radius, p.weapon.damage)
        elif physics.out_of_bounds(p, self.terrain):
            self._begin_resolve()

    def _explode(self, x: float, y: float, radius: float, damage: float) -> None:
        self.terrain.carve_circle(x, y, radius)
        self.events.append(Explosion(x, y, radius))
        for pl in self.players:
            d = math.dist((pl.x, pl.y), (x, y))
            reach = radius + config.PLAYER_RADIUS
            if d < reach:
                pl.hp = max(0.0, pl.hp - damage * (1 - d / reach))
        for pl in self.players:
            self._settle(pl)
        self._begin_resolve()

    def _settle(self, p: Player) -> None:
        """Drop the tank onto the ground beneath it."""
        p.y = self.terrain.surface_y(p.x) - config.PLAYER_RADIUS * 0.5

    def _begin_resolve(self) -> None:
        self.projectile = None
        self.phase = Phase.RESOLVING
        self._resolve_timer = config.RESOLVE_DELAY

    def _next_turn(self) -> None:
        alive = [p for p in self.players if p.alive]
        if len(alive) <= 1:
            self.phase = Phase.GAME_OVER
            self.winner = alive[0].index if alive else None
            return
        self.current = (self.current + 1) % len(self.players)
        self.turn += 1
        self.wind = new_wind(self.rng)
        self.active.fuel = config.PLAYER_FUEL
        self.phase = Phase.AIMING

    # ---- RL hook (stub for later) --------------------------------------
    def observe(self, index: int, samples: int = 32) -> np.ndarray:
        """Numeric observation from `index`'s point of view, normalised to ~[-1, 1]."""
        me, them = self.players[index], self.players[1 - index]
        w, h = self.terrain.width, self.terrain.height
        xs = np.linspace(0, w - 1, samples)
        ground = np.array([self.terrain.surface_y(x) for x in xs]) / h
        head = np.array([
            me.x / w, me.y / h, me.hp / config.PLAYER_HP, me.angle / 180, me.power / 100,
            them.x / w, them.y / h, them.hp / config.PLAYER_HP,
            self.wind / config.WIND_MAX,
        ])
        return np.concatenate([head, ground]).astype(np.float32)

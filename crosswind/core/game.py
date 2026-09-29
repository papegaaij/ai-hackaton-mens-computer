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
from crosswind.core.wind import Wind


class Phase(Enum):
    PLAYING = auto()
    GAME_OVER = auto()


@dataclass
class Explosion:
    x: float
    y: float
    radius: float
    dirt: bool = False  # a mound of earth was built instead of a crater


class Game:
    """A live duel: both players aim, move and fire at the same time, paying for each shot with energy."""

    def __init__(self, seed: int | None = None, width: int = config.WIDTH, height: int = config.HEIGHT,
                 countdown: float = config.START_COUNTDOWN):
        self.rng = np.random.default_rng(seed)
        self.terrain = Terrain(width, height, self.rng)
        self.players = [
            self._spawn(0, width * 0.15, 45.0),
            self._spawn(1, width * 0.85, 135.0),
        ]
        self._wind = Wind(self.rng)
        self.phase = Phase.PLAYING
        self.countdown = countdown  # seconds before anyone may fire
        self.projectiles: list[physics.Projectile] = []
        self.winner: int | None = None
        self.events: list[Explosion] = []  # consumed by the renderer
        self._end_timer: float | None = None
        self._accum = 0.0

    def _spawn(self, index: int, x: float, angle: float) -> Player:
        p = Player(index, x, 0.0, angle)
        self._settle(p)
        return p

    # ---- queries -------------------------------------------------------
    @property
    def wind(self) -> float:
        """Current base wind at ground level (px/s^2)."""
        return self._wind.value

    def can_act(self, index: int) -> bool:
        return self.phase is Phase.PLAYING and self.players[index].alive

    def can_fire(self, index: int, weapon: int | None = None) -> bool:
        """Not during the countdown or cooldown, and only with enough energy for the (selected) weapon."""
        p = self.players[index]
        w = WEAPONS[(p.weapon if weapon is None else weapon) % len(WEAPONS)]
        return self.can_act(index) and self.countdown <= 0 and p.cooldown <= 0 and p.energy >= w.energy

    def barrel_tip(self, p: Player) -> tuple[float, float]:
        rad = math.radians(p.angle)
        length = config.PLAYER_RADIUS * 1.6
        return p.x + math.cos(rad) * length, p.y - config.PLAYER_RADIUS * 0.5 - math.sin(rad) * length

    # ---- input API (used by controllers) ---------------------------------
    def adjust_aim(self, index: int, d_angle: float, d_power: float) -> None:
        if not self.can_act(index):
            return
        p = self.players[index]
        p.angle = float(np.clip(p.angle + d_angle, 0, 180))
        p.power = float(np.clip(p.power + d_power, 5, 100))

    def cycle_weapon(self, index: int, step: int) -> None:
        if self.can_act(index):
            p = self.players[index]
            p.weapon = (p.weapon + step) % len(WEAPONS)

    def move(self, index: int, direction: int, dt: float) -> None:
        """Drive a tank along the ground, or steer it in the air; costs fuel, blocked by steep slopes and walls."""
        if not self.can_act(index) or direction == 0:
            return
        p = self.players[index]
        dist = min(config.PLAYER_SPEED * dt, p.fuel)
        if dist <= 0:
            return
        nx = float(np.clip(p.x + direction * dist, config.PLAYER_RADIUS, self.terrain.width - config.PLAYER_RADIUS))
        max_step = config.MAX_CLIMB * max(dist, 1)
        if p.airborne:
            if self.terrain.surface_y(nx) < self._feet(p):  # ground above our feet there: a wall
                return
        elif self.terrain.surface_y(p.x) - self.terrain.surface_y(nx) > max_step:  # too steep to climb
            return
        p.fuel -= abs(nx - p.x)
        p.x = nx
        if not p.airborne:
            self._settle_or_fall(p, max_step)

    def jump(self, index: int) -> bool:
        """Hop up to get over steep sections (or dodge). Only from the ground; costs JUMP_FUEL."""
        p = self.players[index]
        if not self.can_act(index) or p.airborne or p.fuel < config.JUMP_FUEL:
            return False
        p.fuel -= config.JUMP_FUEL
        p.vy = -config.JUMP_SPEED
        p.airborne = True
        return True

    def fire(self, index: int, action: FireAction) -> bool:
        """Launch a shot for player `index`; costs the weapon's energy. Refused when `can_fire` is False."""
        if not self.can_fire(index, action.weapon):
            return False
        p = self.players[index]
        p.angle = float(np.clip(action.angle, 0, 180))
        p.power = float(np.clip(action.power, 5, 100))
        p.weapon = action.weapon % len(WEAPONS)
        weapon = WEAPONS[p.weapon]
        tx, ty = self.barrel_tip(p)
        self.projectiles.append(physics.launch(tx, ty, p.angle, p.power, weapon, p.index))
        p.energy -= weapon.energy
        p.cooldown = config.FIRE_COOLDOWN
        return True

    # ---- simulation ----------------------------------------------------
    def update(self, dt: float) -> None:
        """Advance real time; physics runs in fixed PHYSICS_DT substeps."""
        self._accum += dt
        while self._accum >= config.PHYSICS_DT:
            self._accum -= config.PHYSICS_DT
            self._tick(config.PHYSICS_DT)

    def _tick(self, dt: float) -> None:
        self._wind.step(dt)
        if self.phase is Phase.GAME_OVER:
            return
        self.countdown = max(0.0, self.countdown - dt)
        for pl in self.players:
            pl.cooldown = max(0.0, pl.cooldown - dt)
            pl.energy = min(config.ENERGY_MAX, pl.energy + config.ENERGY_REGEN * dt)
            pl.fuel = min(config.PLAYER_FUEL, pl.fuel + config.FUEL_REGEN * dt)
            if pl.airborne:
                self._tick_airborne(pl, dt)
        self.projectiles = [q for p in self.projectiles for q in self._tick_projectile(p, dt)]
        if self._end_timer is not None:
            self._end_timer -= dt
            if self._end_timer <= 0:
                self._finish()

    def _tick_airborne(self, p: Player, dt: float) -> None:
        p.vy += config.GRAVITY * dt
        p.y += p.vy * dt
        if p.vy >= 0 and self._feet(p) >= self.terrain.surface_y(p.x):
            p.airborne, p.vy = False, 0.0
            self._settle(p)

    def _tick_projectile(self, p: physics.Projectile, dt: float) -> list[physics.Projectile]:
        """Advance one projectile; returns what is still flying afterwards: itself, its bomblets, or nothing."""
        w = p.weapon
        prev_x, prev_y = p.x, p.y
        physics.step(p, self.wind, self.terrain.height, dt)
        if not p.trail or math.dist(p.trail[-1], (p.x, p.y)) > 4:
            p.trail.append((p.x, p.y))
        if self._touches_tank(p):
            self._impact(p)
            return []
        if w.split and p.vy >= 0:  # top of the arc
            return physics.split(p)
        in_rock = physics.hits_terrain(p, self.terrain)
        if p.timer is not None:  # driller boring on, or bouncer bouncing around
            p.timer -= dt
            if p.timer <= 0:
                self._impact(p)
                return []
        elif in_rock and (w.drill or w.fuse):
            p.timer = w.drill or w.fuse
        elif in_rock:
            self._impact(p)
            return []
        if in_rock and w.fuse:
            physics.bounce(p, self.terrain, prev_x, prev_y, dt)
        return [] if physics.out_of_bounds(p, self.terrain) else [p]

    def _touches_tank(self, p: physics.Projectile) -> bool:
        return any(
            pl.alive and math.dist((pl.x, pl.y), (p.x, p.y)) <= config.PLAYER_RADIUS
            and not (pl.index == p.owner and len(p.trail) < 6)  # don't hit yourself on the way out of the barrel
            for pl in self.players
        )

    def _impact(self, p: physics.Projectile) -> None:
        w = p.weapon
        if w.dirt:
            self._build(p.x, p.y, w.blast_radius)
        else:
            self._explode(p.x, p.y, w.blast_radius, w.damage)

    def _build(self, x: float, y: float, radius: float) -> None:
        """Dirt Bomb: raise a mound of earth; tanks inside it are pushed up on top."""
        self.terrain.fill_circle(x, y, radius)
        self.events.append(Explosion(x, y, radius, dirt=True))
        for pl in self.players:
            if not pl.airborne:
                self._settle(pl)

    def _explode(self, x: float, y: float, radius: float, damage: float) -> None:
        self.terrain.carve_circle(x, y, radius)
        self.events.append(Explosion(x, y, radius))
        for pl in self.players:
            d = math.dist((pl.x, pl.y), (x, y))
            reach = radius + config.PLAYER_RADIUS
            if d < reach:
                pl.hp = max(0.0, pl.hp - damage * (1 - d / reach))
        for pl in self.players:
            if not pl.airborne:
                self._settle_or_fall(pl, 2)
        if self._end_timer is None and sum(pl.alive for pl in self.players) <= 1:
            self._end_timer = config.END_DELAY

    def _settle(self, p: Player) -> None:
        """Drop the tank onto the ground beneath it."""
        p.y = self.terrain.surface_y(p.x) - config.PLAYER_RADIUS * 0.5

    def _feet(self, p: Player) -> float:
        """Y where the tank touches the ground (the inverse of `_settle`)."""
        return p.y + config.PLAYER_RADIUS * 0.5

    def _settle_or_fall(self, p: Player, tolerance: float) -> None:
        """Put a grounded tank on the ground, or let it fall if the ground dropped away more than `tolerance`."""
        if self.terrain.surface_y(p.x) - self._feet(p) > tolerance:
            p.airborne, p.vy = True, 0.0
        else:
            self._settle(p)

    def _finish(self) -> None:
        alive = [p for p in self.players if p.alive]
        self.phase = Phase.GAME_OVER
        self.winner = alive[0].index if len(alive) == 1 else None
        self.projectiles.clear()

    # ---- RL hook (stub for later) --------------------------------------
    def observe(self, index: int, samples: int = 32) -> np.ndarray:
        """Numeric observation from `index`'s point of view, normalised to ~[-1, 1]."""
        me, them = self.players[index], self.players[1 - index]
        w, h = self.terrain.width, self.terrain.height
        xs = np.linspace(0, w - 1, samples)
        ground = np.array([self.terrain.surface_y(x) for x in xs]) / h
        head = np.array([
            me.x / w, me.y / h, me.hp / config.PLAYER_HP, me.angle / 180, me.power / 100,
            me.energy / config.ENERGY_MAX, me.weapon / len(WEAPONS),
            them.x / w, them.y / h, them.hp / config.PLAYER_HP, them.energy / config.ENERGY_MAX,
            self.wind / config.WIND_MAX,
        ])
        return np.concatenate([head, ground]).astype(np.float32)

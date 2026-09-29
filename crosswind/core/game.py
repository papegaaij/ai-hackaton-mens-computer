"""The simulation. Pure Python + numpy, no pygame: runs headless for tests and AI training."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from crosswind import config
from crosswind.core import physics
from crosswind.core.player import Player
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import WEAPONS
from crosswind.core.wind import Wind


def _clamp(v: float, lo: float, hi: float) -> float:
    """Scalar clip (np.clip is slow on plain floats, and this runs many times per frame)."""
    return float(min(max(v, lo), hi))


class Phase(Enum):
    PLAYING = auto()
    GAME_OVER = auto()


@dataclass
class Explosion:
    x: float
    y: float
    radius: float
    dirt: bool = False  # a mound of earth was built instead of a crater


@dataclass(frozen=True)
class Impact:
    x: float
    y: float
    time: float   # Game.time when it came down
    weapon: int   # index into WEAPONS of the weapon that was fired
    angle: float  # the aim it was fired with
    power: float


@dataclass
class SoundEvent:
    """Something audible happened. Pure data: the audio layer decides what it sounds like."""
    name: str                 # launch, explode, dirt, split, bounce, drill, hit, down, win, draw,
                              # count, go, ready, no_energy, switch, jump, land, gust
    x: float = 0.0
    y: float = 0.0
    player: int | None = None
    weapon: str = ""
    size: float = 0.0         # blast radius, damage, bounce count or countdown number, depending on name


class Game:
    """A live duel: both players aim, move and fire at the same time, paying for each shot with energy."""

    def __init__(self, seed: int | None = None, width: int = config.WIDTH, height: int = config.HEIGHT,
                 countdown: float = config.START_COUNTDOWN, wind_strength: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.terrain = Terrain(width, height, self.rng)
        self.players = [
            self._spawn(0, width * 0.15, 45.0),
            self._spawn(1, width * 0.85, 135.0),
        ]
        self._wind = Wind(self.rng, wind_strength)
        self.phase = Phase.PLAYING
        self.countdown = countdown  # seconds before anyone may fire
        self.projectiles: list[physics.Projectile] = []
        self.winner: int | None = None
        self.events: list[Explosion] = []  # consumed by the renderer
        self.sounds: deque[SoundEvent] = deque(maxlen=256)  # consumed by the audio layer (bounded when nobody listens)
        if countdown > 0:
            self._sound("count", size=math.ceil(countdown))
        self._end_timer: float | None = None
        self._accum = 0.0
        self.time = 0.0  # simulated seconds since the start
        # per player: where their latest shot came down (what a player sees on screen), or None
        self.last_impact: list[Impact | None] = [None, None]

    def _spawn(self, index: int, x: float, angle: float) -> Player:
        p = Player(index, x, 0.0, angle)
        self._settle(p)
        return p

    def _sound(self, name: str, x: float = 0.0, y: float = 0.0, player: int | None = None,
               weapon: str = "", size: float = 0.0) -> None:
        self.sounds.append(SoundEvent(name, x, y, player, weapon, size))

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
    def adjust_aim(self, index: int, d_angle: float, d_power: float, dt: float) -> None:
        """Turn the barrel and the power dial; never faster than ANGLE_SPEED / POWER_SPEED allow."""
        if not self.can_act(index):
            return
        p = self.players[index]
        max_angle, max_power = config.ANGLE_SPEED * dt, config.POWER_SPEED * dt
        p.angle = _clamp(p.angle + _clamp(d_angle, -max_angle, max_angle), 0, 180)
        p.power = _clamp(p.power + _clamp(d_power, -max_power, max_power), 5, 100)

    def cycle_weapon(self, index: int, step: int = 1) -> None:
        """Step to the next (or previous) weapon: one weapon per call, like one key press."""
        if self.can_act(index) and step:
            p = self.players[index]
            p.weapon = (p.weapon + (1 if step > 0 else -1)) % len(WEAPONS)
            self._sound("switch", p.x, p.y, index, WEAPONS[p.weapon].name, p.weapon)

    def move(self, index: int, direction: int, dt: float) -> None:
        """Drive a tank along the ground, or steer it in the air; costs fuel, blocked by steep slopes and walls."""
        if not self.can_act(index) or direction == 0:
            return
        p = self.players[index]
        dist = min(config.PLAYER_SPEED * dt, p.fuel)
        if dist <= 0:
            return
        nx = _clamp(p.x + direction * dist, config.PLAYER_RADIUS, self.terrain.width - config.PLAYER_RADIUS)
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
        self._sound("jump", p.x, p.y, index)
        return True

    def fire(self, index: int) -> bool:
        """Launch a shot with the tank's current aim and weapon; costs the weapon's energy.

        Refused when `can_fire` is False.
        """
        if not self.can_fire(index):
            p = self.players[index]
            if self.can_act(index) and self.countdown <= 0 and p.cooldown <= 0:  # refused only for lack of energy
                self._sound("no_energy", p.x, p.y, index)
            return False
        p = self.players[index]
        weapon = WEAPONS[p.weapon]
        tx, ty = self.barrel_tip(p)
        shot = physics.launch(tx, ty, p.angle, p.power, weapon, p.index)
        shot.weapon_index, shot.aim = p.weapon, (p.angle, p.power)
        self.projectiles.append(shot)
        p.energy -= weapon.energy
        p.cooldown = config.FIRE_COOLDOWN
        self._sound("launch", tx, ty, index, weapon.name, weapon.blast_radius)
        return True

    # ---- simulation ----------------------------------------------------
    def update(self, dt: float) -> None:
        """Advance real time; physics runs in fixed PHYSICS_DT substeps."""
        self._accum += dt
        while self._accum >= config.PHYSICS_DT:
            self._accum -= config.PHYSICS_DT
            self._tick(config.PHYSICS_DT)

    def _tick(self, dt: float) -> None:
        self.time += dt
        target = self._wind.target
        self._wind.step(dt)
        if self._wind.target != target:
            self._sound("gust", size=self._wind.target)
        if self.phase is Phase.GAME_OVER:
            return
        before = self.countdown
        self.countdown = max(0.0, self.countdown - dt)
        if math.ceil(before) != math.ceil(self.countdown):
            self._sound("count" if self.countdown > 0 else "go", size=math.ceil(self.countdown))
        for pl in self.players:
            pl.cooldown = max(0.0, pl.cooldown - dt)
            cost = WEAPONS[pl.weapon].energy
            was_short = pl.energy < cost
            pl.energy = min(config.ENERGY_MAX, pl.energy + config.ENERGY_REGEN * dt)
            if was_short and pl.energy >= cost and pl.alive and self.countdown <= 0:
                self._sound("ready", pl.x, pl.y, pl.index)
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
            self._sound("land", p.x, p.y, p.index)

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
            self._sound("split", p.x, p.y, p.owner, w.name)
            return physics.split(p)
        in_rock = physics.hits_terrain(p, self.terrain)
        if p.timer is not None:  # driller boring on, or bouncer bouncing around
            p.timer -= dt
            if p.timer <= 0:
                self._impact(p)
                return []
        elif in_rock and (w.drill or w.fuse):
            p.timer = w.drill or w.fuse
            if w.drill:
                self._sound("drill", p.x, p.y, p.owner, w.name)
        elif in_rock:
            self._impact(p)
            return []
        if in_rock and w.fuse:
            speed = math.hypot(p.vx, p.vy)
            physics.bounce(p, self.terrain, prev_x, prev_y, dt)
            if speed > 60 and p.age - p.last_bounce > 0.15:  # a real bounce, not rolling
                p.last_bounce = p.age
                self._sound("bounce", p.x, p.y, p.owner, w.name, p.bounces)
                p.bounces += 1
        return [] if physics.out_of_bounds(p, self.terrain) else [p]

    def _touches_tank(self, p: physics.Projectile) -> bool:
        return any(
            pl.alive and math.dist((pl.x, pl.y), (p.x, p.y)) <= config.PLAYER_RADIUS
            and not (pl.index == p.owner and len(p.trail) < 6)  # don't hit yourself on the way out of the barrel
            for pl in self.players
        )

    def _impact(self, p: physics.Projectile) -> None:
        w = p.weapon
        self.last_impact[p.owner] = Impact(p.x, p.y, self.time, p.weapon_index, *p.aim)
        if w.dirt:
            self._build(p.x, p.y, w.blast_radius)
            self._sound("dirt", p.x, p.y, p.owner, w.name, w.blast_radius)
        else:
            self._explode(p.x, p.y, w.blast_radius, w.damage, p.owner, w.name)

    def _build(self, x: float, y: float, radius: float) -> None:
        """Dirt Bomb: raise a mound of earth; tanks inside it are pushed up on top."""
        self.terrain.fill_circle(x, y, radius)
        self.events.append(Explosion(x, y, radius, dirt=True))
        for pl in self.players:
            if not pl.airborne:
                self._settle(pl)

    def _explode(self, x: float, y: float, radius: float, damage: float,
                 owner: int | None = None, weapon: str = "") -> None:
        self.terrain.carve_circle(x, y, radius)
        self.events.append(Explosion(x, y, radius))
        self._sound("explode", x, y, owner, weapon, radius)
        for pl in self.players:
            d = math.dist((pl.x, pl.y), (x, y))
            reach = radius + config.PLAYER_RADIUS
            if d < reach and pl.alive:
                dmg = damage * (1 - d / reach)
                pl.hp = max(0.0, pl.hp - dmg)
                if dmg > 0:
                    self._sound("hit", pl.x, pl.y, pl.index, weapon, dmg)
                if not pl.alive:
                    self._sound("down", pl.x, pl.y, pl.index)
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
        self._sound("win" if self.winner is not None else "draw", player=self.winner)
        self.projectiles.clear()

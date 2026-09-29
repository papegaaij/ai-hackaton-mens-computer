"""What the AI sees, as numbers: the HUD, the terrain, both tanks, shots in flight and its own last impact.

Everything is mirrored when the enemy is on the left, so the AI always sees its enemy ahead, to the right:
x is flipped, angles become 180 - angle and the wind changes sign. One network plays both sides, and a
tank that has driven past its enemy simply turns around (the game is left-right symmetric).
"""
from __future__ import annotations

import numpy as np

from crosswind import config
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS

N_WEAPONS = len(WEAPONS)
ANGLE_BINS = np.arange(10.0, 171.0, 10.0)  # the angles (mirrored) the AI picks from
POWER_STEPS = np.array([-15.0, -10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0, 15.0])  # added to the aimer
TERRAIN_SAMPLES = 32
AIM_TARGETS = ("enemy", "halfway", "near")  # where the aimer aims: the enemy tank, halfway there, or close by
NEAR = 120.0          # px in front of the tank for the "near" target (e.g. a Dirt Bomb for cover)
WIND_TREND_SPAN = 0.1  # s over which the AI watches the wind bar move
INCOMING = 3          # nearest enemy shots in flight that are seen
IMPACT_MEMORY = 5.0   # s after which the last impact counts as old news


FLIP_MARGIN = 40.0  # px the enemy must be past us before the AI turns around (no flapping while overlapping)


def facing_flip(game: Game, index: int, flip: bool | None = None) -> bool:
    """Whether to mirror player `index`'s view: True when the enemy is to its left.

    `flip` is the current setting; it only changes once the enemy is clearly on the other side.
    """
    dx = game.players[1 - index].x - game.players[index].x
    if flip is None or abs(dx) > FLIP_MARGIN:
        return dx < 0
    return flip


def canon_angle(angle: float, flip: bool) -> float:
    """A game angle as the AI sees it (or back again: the mapping is its own inverse)."""
    return 180.0 - angle if flip else angle


def canon_dx(dx: float, flip: bool) -> float:
    return -dx if flip else dx


def one_hot(i: int, n: int = N_WEAPONS) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


def aimer_inputs(weapon, angles, dx, dy, wind, trend) -> np.ndarray:
    """Aimer features in the mirrored frame, one row per shot (scalars are broadcast).

    dx, dy: px from the tank centre to the target; wind: the HUD wind; trend: how fast the HUD wind
    bar is moving (per second).
    """
    weapon, angles, dx, dy, wind, trend = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=np.float32)) for v in (weapon, angles, dx, dy, wind, trend)))
    rad = np.radians(angles)
    return np.column_stack([
        np.eye(N_WEAPONS, dtype=np.float32)[weapon.astype(int)],
        np.cos(rad), np.sin(rad), dx / config.WIDTH, dy / config.HEIGHT, wind / config.WIND_MAX,
        trend / config.WIND_MAX,
    ]).astype(np.float32)


def aim_target(game: Game, index: int, flip: bool, target: int) -> tuple[float, float]:
    """(dx, dy) from the tank to the chosen aim target (see AIM_TARGETS), dx in the mirrored frame."""
    me, them = game.players[index], game.players[1 - index]
    dx, dy = canon_dx(them.x - me.x, flip), them.y - me.y
    if AIM_TARGETS[target] == "halfway":
        dx /= 2
        x = me.x + canon_dx(dx, flip)
        dy = game.terrain.surface_y(x) - me.y
    elif AIM_TARGETS[target] == "near":
        dx = NEAR if dx >= 0 else -NEAR
        x = me.x + canon_dx(dx, flip)
        dy = game.terrain.surface_y(x) - me.y
    return dx, dy


def observe(game: Game, index: int, flip: bool, wind_trend: float, aim_suggestions: np.ndarray,
            target_angle: float | None, target_power: float | None) -> np.ndarray:
    """The observation for player `index`, mirrored if `flip`. `aim_suggestions`: aimer power per ANGLE_BINS."""
    me, them = game.players[index], game.players[1 - index]
    w, h = game.terrain.width, game.terrain.height
    m = flip

    own = [me.x / w if not m else (w - me.x) / w, me.y / h, me.hp / config.PLAYER_HP, me.fuel / config.PLAYER_FUEL,
           me.energy / config.ENERGY_MAX, me.cooldown / config.FIRE_COOLDOWN, float(me.airborne),
           canon_angle(me.angle, flip) / 180, me.power / 100, float(game.countdown > 0),
           float(game.can_fire(index))]
    enemy = [canon_dx(them.x - me.x, flip) / w, (them.y - me.y) / h, them.hp / config.PLAYER_HP,
             them.energy / config.ENERGY_MAX, float(them.airborne)]
    wind = [canon_dx(game.wind, flip) / config.WIND_MAX, canon_dx(wind_trend, flip) / config.WIND_MAX]

    xs = np.linspace(0, w - 1, TERRAIN_SAMPLES)
    if m:
        xs = xs[::-1]
    ground = [(game.terrain.surface_y(x) - me.y) / h for x in xs]

    hit = game.last_impact[index]
    if hit is None:
        impact = [0.0] * 6 + [0.0] * N_WEAPONS
    else:
        impact = [1.0, canon_dx(hit.x - them.x, flip) / w, (hit.y - them.y) / h,
                  min(game.time - hit.time, IMPACT_MEMORY) / IMPACT_MEMORY,
                  canon_angle(hit.angle, flip) / 180, hit.power / 100, *one_hot(hit.weapon)]

    targets = [0.0 if target_angle is None else target_angle / 180,
               0.0 if target_power is None else target_power / 100]

    incoming = []
    shots = sorted((p for p in game.projectiles if p.owner != index),
                   key=lambda p: (p.x - me.x) ** 2 + (p.y - me.y) ** 2)[:INCOMING]
    for p in shots:
        incoming += [1.0, canon_dx(p.x - me.x, flip) / w, (p.y - me.y) / h,
                     canon_dx(p.vx, flip) / config.MAX_LAUNCH_SPEED, p.vy / config.MAX_LAUNCH_SPEED]
    incoming += [0.0] * (5 * INCOMING - len(incoming))

    return np.concatenate([own, one_hot(me.weapon), enemy, wind, ground, impact, targets,
                           np.asarray(aim_suggestions) / 100, incoming]).astype(np.float32)


OBS_SIZE = 11 + N_WEAPONS + 5 + 2 + TERRAIN_SAMPLES + 6 + N_WEAPONS + 2 + len(ANGLE_BINS) + 5 * INCOMING

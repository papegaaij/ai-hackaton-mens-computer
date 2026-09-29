"""A rule-based computer player: no training and no models, it brackets its shots like a human would.

It aims toward the enemy at a fixed angle, guesses a first power from the distance and the wind, and
after every shot it sees land it turns the power up (short) or down (long), halving the step each time
the miss flips side. Weapons and dodging follow simple rules of thumb. It never works out where a shot
will fly: it only looks at what is on screen, and it plays through the same keys (see `AIController.drive`).
"""
from __future__ import annotations

from dataclasses import dataclass

from crosswind import config
from crosswind.ai.controller import AIController
from crosswind.ai.obs import canon_angle, canon_dx
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS

SPARK, PLASMA, ROCKET, BOUNCER, DRILLER, CLUSTER, DIRT, MEGATON = range(8)
ANGLE, HIGH_ANGLE = 50.0, 65.0  # aim, in the mirrored frame (enemy to the right); high over a big hill
HILL, BIG_HILL = 60.0, 150.0    # px the ground in between rises above both tanks
FIRST_STEP = 8.0      # power step of the bracketing, halved whenever the miss flips side
MIN_STEP = 0.5
HIT = 12.0            # px: a miss this small counts as on target
ENEMY_MOVED = 60.0    # px the enemy may move before the bracketing starts over
WIND_CHANGED = 0.6 * config.WIND_MAX
# power by feel, relative to a Plasma Orb (from playing: rockets push themselves, sparks fly light)
FEEL = {SPARK: 0.88, ROCKET: 0.7}
WIND_POWER = 35.0     # power added for a full head wind (taken off for a full tail wind)
DODGE_TIME = 0.5      # s of driving to get out of the way
DANGER = 220.0        # px: an enemy shot this close and coming closer is a threat


@dataclass
class Bracket:
    """What it has learned: the power that worked, under which wind, angle and enemy distance."""
    power: float
    dx: float
    wind: float
    step: float = FIRST_STEP
    side: int = 0          # -1 short, +1 long, 0 no miss seen yet
    angle: float = ANGLE
    miss: float | None = None


class RuleBasedController(AIController):
    def __init__(self, index: int, keys=None, label: str = "AI (Rules)", decision_interval: float = 0.25,
                 seed: int | None = None):
        super().__init__(index, policy=None, aimer=None, keys=keys, label=label,
                         decision_interval=decision_interval, seed=seed)
        self.bracket: Bracket | None = None  # shared by all weapons, in Plasma Orb power
        self.misses: list[float] = []   # |miss| of each own impact it learned from, in order
        self._seen_impact: float | None = None
        self._flying: dict[int, tuple] = {}  # own shots in the air: last position, weapon and aim
        self._learned: tuple | None = None  # (weapon, angle, power) of the last shot it learned from
        self._fire_wind: dict[tuple[int, float], float] = {}  # (weapon, power) -> wind when it was fired
        self._choice: int | None = None  # weapon picked for the next shot
        self._dodge = 0.0
        self._last_x: float | None = None
        self._last_hp: float | None = None

    # ---- what it sees ----------------------------------------------------
    def _dx(self, game: Game) -> float:
        me, them = game.players[self.index], game.players[1 - self.index]
        return canon_dx(them.x - me.x, self.index)

    def _hill(self, game: Game) -> float:
        """How far (px) the ground between the tanks rises above the higher of the two."""
        me, them = game.players[self.index], game.players[1 - self.index]
        lo, hi = sorted((me.x, them.x))
        top = min((game.terrain.surface_y(x) for x in range(int(lo) + 40, int(hi) - 40, 8)),
                  default=game.terrain.height)
        return min(me.y, them.y) - top

    def _wind(self, game: Game) -> float:
        return canon_dx(game.wind, self.index)  # positive: blowing toward the enemy (tail wind)

    def _bracket(self, game: Game) -> Bracket:
        """The power that works (for a Plasma Orb), kept up to date when the enemy drives or the wind turns."""
        b = self.bracket
        dx, wind = abs(self._dx(game)), self._wind(game)
        if b is None:  # rule of thumb for the first shot
            b = self.bracket = Bracket(35.0 + 55.0 * dx / game.terrain.width, dx, 0.0, angle=self._angle(game))
        if abs(dx - b.dx) > ENEMY_MOVED:  # they drove off: shift by the rule of thumb and bracket again
            b.power += 55.0 * (dx - b.dx) / game.terrain.width
            b.dx, b.step, b.side = dx, FIRST_STEP, 0
        if abs(wind - b.wind) > WIND_CHANGED or b.angle != self._angle(game):
            b.step, b.side, b.angle = FIRST_STEP, 0, self._angle(game)
        return b

    def _power(self, game: Game, weapon: int) -> float:
        """Power for `weapon`: the bracketed Plasma Orb power, by feel for this weapon and the wind now."""
        b = self._bracket(game)
        gust = WIND_POWER * WEAPONS[PLASMA].mass / WEAPONS[weapon].mass  # heavy shells care less
        return b.power * FEEL.get(weapon, 1.0) - gust * (self._wind(game) - b.wind) / config.WIND_MAX

    def _learn(self, game: Game) -> None:
        """Watch its own shots: where did the latest one come down, or did it fly off the screen?"""
        imp = game.last_impact[self.index]
        if imp is not None and imp.time != self._seen_impact:
            self._seen_impact = imp.time
            self._feedback(game, imp.x, imp.weapon, imp.angle, imp.power)
        w, h = game.terrain.width, game.terrain.height
        flying = {id(s): s for s in game.projectiles if s.owner == self.index}
        for key, (x, y, weapon, aim) in self._flying.items():
            if key not in flying and (x < 0 or x >= w or y >= h):  # gone off the screen
                self._feedback(game, x, weapon, *aim)
        self._flying = {k: (s.x, s.y, s.weapon_index, s.aim) for k, s in flying.items()}

    def _feedback(self, game: Game, x: float, weapon: int, angle: float, power: float) -> None:
        """One shot came down at x: was it short or long of the enemy? Bracket the power."""
        shot = (weapon, angle, power)
        if shot == self._learned:  # another bomblet of a shot it already learned from
            return
        self._learned = shot
        angle = canon_angle(angle, self.index)
        if weapon in (DIRT, SPARK) or abs(angle - self._angle(game)) > 2:  # cover, or erratic little sparks
            return
        them = game.players[1 - self.index]
        miss = canon_dx(x - them.x, self.index)  # negative: short, positive: long
        self.misses.append(abs(miss))
        b = self._bracket(game)
        power /= FEEL.get(weapon, 1.0)  # as a Plasma Orb power
        b.miss = abs(miss)
        b.wind = self._fire_wind.pop((weapon, round(power, 2)), self._wind(game))
        b.dx = abs(self._dx(game))
        if abs(miss) < HIT:
            b.power, b.step = power, min(b.step, 2.0)  # on target: stay, and only nudge from here
            return
        side = 1 if miss > 0 else -1
        if b.side and side != b.side:
            b.step = max(b.step / 2, MIN_STEP)
        elif b.side == side and abs(miss) > 100:
            b.step = min(b.step * 1.5, 20.0)
        b.side = side
        b.power = min(max(power - side * b.step, 5.0), 100.0)

    # ---- deciding --------------------------------------------------------
    def _angle(self, game: Game) -> float:
        return HIGH_ANGLE if self._hill(game) > BIG_HILL else ANGLE

    def _pick_weapon(self, game: Game) -> int:
        me = game.players[self.index]
        last = self.misses[-1] if self.misses else None
        r = self.rng.random()
        if me.energy >= 0.7 * config.ENERGY_MAX and last is not None and last < 60:  # then waits for a full bar
            return MEGATON
        if me.hp < 40 and r < 0.15:
            return DIRT
        if me.energy < WEAPONS[PLASMA].energy:
            return SPARK
        if abs(game.wind) > 0.6 * config.WIND_MAX:
            return ROCKET
        if self._hill(game) > HILL:
            return DRILLER
        if last is not None and last < 100 and r < 0.2:  # close enough to try something different
            return CLUSTER if r < 0.12 else BOUNCER
        return PLASMA

    def _threatened(self, game: Game) -> bool:
        me = game.players[self.index]
        for s in game.projectiles:
            if s.owner == self.index:
                continue
            dx, dy = me.x - s.x, me.y - s.y
            if dx * dx + dy * dy < DANGER * DANGER and dx * s.vx + dy * s.vy > 0:  # close and coming closer
                return True
        return False

    def _decide(self, game: Game) -> None:
        me = game.players[self.index]
        if self._choice is None:
            self._choice = self._pick_weapon(game)
        weapon = self._choice
        self.weapon = weapon
        angle = self._angle(game)
        if weapon == DIRT:  # a mound just in front, for cover
            power = 25.0
        else:
            power = self._power(game, weapon)
        self.target_angle = canon_angle(angle, self.index)
        power = float(min(max(power, 5.0), 100.0))
        if self.target_power is None or abs(power - self.target_power) > 1.5:  # don't chase every gust
            self.target_power = power
        in_flight = any(s.owner == self.index for s in game.projectiles)
        ready = (abs(me.angle - self.target_angle) < 0.5 and abs(me.power - self.target_power) < 0.5
                 and me.weapon == weapon and game.can_fire(self.index))
        # one shot in the air at a time, so each one tells it something (unless energy would go to waste)
        self._fire = ready and (not in_flight or me.energy >= config.ENERGY_MAX - 0.5)

    def update(self, game: Game, dt: float) -> None:
        me = game.players[self.index]
        self._learn(game)
        hit = self._last_hp is not None and me.hp < self._last_hp
        self._last_hp = me.hp
        if (hit or self._threatened(game)) and self._dodge <= 0 and me.fuel > config.JUMP_FUEL:
            self._dodge = DODGE_TIME
            self.move = int(self.rng.choice((-1, 1)))
        if self._dodge > 0:
            self._dodge -= dt
            if self._last_x is not None and me.x == self._last_x and not me.airborne and me.fuel >= config.JUMP_FUEL:
                self._jump = True  # driving is blocked: hop over it
            if self._dodge <= 0:
                self.move = 0
        self._last_x = me.x
        self._decide_in -= dt
        if self._decide_in <= 0 and game.can_act(self.index):
            self._decide_in = self.decision_interval
            self._decide(game)
        fired_before = sum(self.shots)
        self.drive(game, dt)
        if sum(self.shots) > fired_before:  # remember the wind it fired in, and pick the next weapon
            self._fire_wind[(me.weapon, round(me.power, 2))] = self._wind(game)
            self._choice = None

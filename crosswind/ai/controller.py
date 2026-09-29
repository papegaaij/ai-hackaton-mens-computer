"""The computer player: decides a few times a second, and in between works the same keys a human has.

A decision picks an angle, a power correction on top of the aimer's suggestion, a weapon, a driving
direction, and whether to jump or fire. A "servo" then presses the angle, power and weapon keys at human
speed (with the same tap-fine / hold-fast ramp) until the dials read what was decided. What it presses is
reported by `held()`, so the HUD lights up the AI's keys just like a human's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from crosswind import config
from crosswind.ai.aimer import Aimer
from crosswind.ai.nets import MODELS_DIR, MLP
from crosswind.ai.obs import (AIM_TARGETS, ANGLE_BINS, N_WEAPONS, POWER_STEPS, WIND_TREND_SPAN, aim_target,
                              canon_angle, canon_dx, observe)
from crosswind.control.ramp import AimRamp
from crosswind.core.game import Game

# Action: [angle bin, power step, weapon, aim target, move (-1/0/+1 as 0/1/2), jump, fire], mirrored frame
ACTION_NVEC = (len(ANGLE_BINS), len(POWER_STEPS), N_WEAPONS, len(AIM_TARGETS), 3, 2, 2)
JUMP, FIRE = 5, 6  # positions in ACTION_NVEC
TAP_FLASH = 0.15     # s a one-press key (jump, weapon, fire) stays lit in the HUD
WEAPON_PRESS = 0.1   # s between two presses of the weapon key
DEADBAND = 0.05      # dial reading (degrees / power) that counts as "there"

Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (observation, action mask) -> action


def action_mask(game: Game, index: int) -> np.ndarray:
    """Which choices are allowed, flattened over ACTION_NVEC: no firing or jumping when the rules forbid it."""
    p = game.players[index]
    parts = [np.ones(n, dtype=bool) for n in ACTION_NVEC]
    parts[JUMP][1] = game.can_act(index) and not p.airborne and p.fuel >= config.JUMP_FUEL
    parts[FIRE][1] = game.can_fire(index)
    return np.concatenate(parts)


class TacticsPolicy:
    """The trained decision network (exported from the PPO training run)."""

    def __init__(self, net: MLP, deterministic: bool = True, seed: int | None = None):
        self.net = net
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True, seed: int | None = None) -> TacticsPolicy:
        return cls(MLP.load(path)[0], deterministic, seed)

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = np.where(mask, self.net(obs), -np.inf)
        action, start = [], 0
        for n in ACTION_NVEC:
            part = logits[start:start + n]
            start += n
            if self.deterministic:
                action.append(int(np.argmax(part)))
            else:
                prob = np.exp(part - part.max())
                action.append(int(self.rng.choice(n, p=prob / prob.sum())))
        return np.array(action)


class AIController:
    def __init__(self, index: int, policy: Policy | None, aimer: Aimer, keys=None, label: str = "AI",
                 decision_interval: float = 0.2, power_noise: float = 0.0, seed: int | None = None):
        self.index = index
        self.policy = policy  # None: decisions come from outside (the training env calls `act`)
        self.aimer = aimer
        self.keys = keys
        self.label = label
        self.decision_interval = decision_interval
        self.power_noise = power_noise
        self.rng = np.random.default_rng(seed)
        self.target_angle: float | None = None  # game angle the servo turns the barrel to
        self.target_power: float | None = None
        self.weapon = 0
        self.move = 0
        self._jump = self._fire = False
        self._noise = 0.0
        self._ramps = (AimRamp(), AimRamp())
        self._decide_in = 0.0
        self._weapon_in = 0.0
        self._held: set[str] = set()
        self._flash: dict[str, float] = {}
        self._winds: list[tuple[float, float]] = []  # (time, HUD wind) seen over the last moment
        self._trend = 0.0
        self.shots = [0] * N_WEAPONS  # shots fired per weapon, and jumps made (for training stats)
        self.jumps = 0

    # ---- deciding ------------------------------------------------------
    def watch_wind(self, game: Game) -> float:
        """Look at the HUD wind bar (call every frame); returns how fast it is moving (per second)."""
        self._winds.append((game.time, game.wind))
        while len(self._winds) > 2 and game.time - self._winds[1][0] >= WIND_TREND_SPAN:
            self._winds.pop(0)
        (t0, w0), (t1, w1) = self._winds[0], self._winds[-1]
        return (w1 - w0) / (t1 - t0) if t1 > t0 else 0.0

    def _suggest(self, game: Game, weapon: int, angles, target: int) -> np.ndarray:
        dx, dy = aim_target(game, self.index, target)
        return self.aimer.suggest(weapon, angles, dx, dy, canon_dx(game.wind, self.index),
                                  canon_dx(self._trend, self.index))

    def observe(self, game: Game) -> np.ndarray:
        """Observation for the policy."""
        me = game.players[self.index]
        suggestions = self._suggest(game, me.weapon, ANGLE_BINS, 0)
        target = None if self.target_angle is None else canon_angle(self.target_angle, self.index)
        return observe(game, self.index, self._trend, suggestions, target, self.target_power)

    def act(self, action: np.ndarray, game: Game) -> None:
        """Take a decision (see ACTION_NVEC); the keys are then pressed frame by frame in `drive`."""
        a_bin, p_step, weapon, target, move, jump, fire = (int(v) for v in action)
        self.target_angle = canon_angle(float(ANGLE_BINS[a_bin]), self.index)
        # the power for the weapon it will fire, which may not be selected yet
        power = self._suggest(game, weapon, ANGLE_BINS[a_bin], target)[0] + POWER_STEPS[p_step] + self._noise
        self.target_power = float(min(max(power, 5.0), 100.0))
        self.weapon = weapon
        self.move = move - 1 if self.index == 0 else 1 - move
        self._jump, self._fire = bool(jump), bool(fire)

    # ---- pressing keys -------------------------------------------------
    def update(self, game: Game, dt: float) -> None:
        """Per frame in the game: decide when it is time, then work the keys."""
        self._decide_in -= dt
        if self.policy is not None and self._decide_in <= 0 and game.can_act(self.index):
            self._decide_in = self.decision_interval
            self.act(self.policy(self.observe(game), action_mask(game, self.index)), game)
        self.drive(game, dt)

    def drive(self, game: Game, dt: float) -> None:
        self._trend = self.watch_wind(game)
        p = game.players[self.index]
        held = set()
        d_angle = self._servo(0, self.target_angle, p.angle, config.ANGLE_SPEED, dt)
        d_power = self._servo(1, self.target_power, p.power, config.POWER_SPEED, dt)
        game.adjust_aim(self.index, d_angle, d_power, dt)
        if d_angle:
            held.add("angle_left" if d_angle > 0 else "angle_right")  # the same keys a human would press
        if d_power:
            held.add("power_up" if d_power > 0 else "power_down")
        if self.move:
            game.move(self.index, self.move, dt)
            held.add("move_right" if self.move > 0 else "move_left")
        self._weapon_in -= dt
        if self.weapon != p.weapon and self._weapon_in <= 0 and game.can_act(self.index):
            game.cycle_weapon(self.index)
            self._weapon_in = WEAPON_PRESS
            self._flash["weapon"] = TAP_FLASH
        if self._jump:
            self._jump = False
            if game.jump(self.index):
                self._flash["jump"] = TAP_FLASH
                self.jumps += 1
        if self._fire:
            self._fire = False
            if game.fire(self.index):
                self._flash["fire"] = TAP_FLASH
                self.shots[p.weapon] += 1
                self._noise = float(self.rng.uniform(-self.power_noise, self.power_noise))
        for key in list(self._flash):
            self._flash[key] -= dt
            if self._flash[key] <= 0:
                del self._flash[key]
        self._held = held | set(self._flash)

    def _servo(self, axis: int, target: float | None, value: float, speed: float, dt: float) -> float:
        """How far to turn a dial this frame: toward `target`, as fast as a held key would, never past it."""
        diff = 0.0 if target is None else target - value
        direction = 0 if abs(diff) <= DEADBAND else (1 if diff > 0 else -1)
        step = self._ramps[axis].step(direction, dt) * speed * dt
        return float(min(max(step, -abs(diff)), abs(diff)))

    def held(self) -> set[str]:
        return self._held


DIFFICULTIES = {  # name: (model file, seconds between decisions, power noise)
    "Easy": ("tactics_easy.npz", 0.8, 14.0),
    "Medium": ("tactics_medium.npz", 0.4, 7.0),
    "Hard": ("tactics_hard.npz", 0.2, 0.0),
}


def models_available() -> bool:
    return all((MODELS_DIR / f).exists() for f in ["aimer.npz", *(m for m, _, _ in DIFFICULTIES.values())])


def make_ai(index: int, difficulty: str, keys=None) -> AIController:
    model, interval, noise = DIFFICULTIES[difficulty]
    return AIController(index, TacticsPolicy.load(MODELS_DIR / model), Aimer.load(), keys=keys,
                        label=f"AI ({difficulty})", decision_interval=interval, power_noise=noise)

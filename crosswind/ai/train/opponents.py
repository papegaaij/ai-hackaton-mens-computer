"""Opponents to train against: a sitting duck, a random player, a scripted aimer bot, and past selves."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from crosswind.ai.aimer import Aimer
from crosswind.ai.controller import ACTION_NVEC, JUMP, AIController, TacticsPolicy
from crosswind.ai.obs import ANGLE_BINS, POWER_STEPS
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS


class Idle:
    """Stands still and never fires."""
    label = ""
    keys = None

    def __init__(self, index: int):
        self.index = index

    def update(self, game: Game, dt: float) -> None:
        pass

    def held(self) -> set[str]:
        return set()


class RandomPolicy:
    """Random (allowed) choices, but holding a fire button now and then, so it drives and shoots around."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        action, start = [], 0
        for n in ACTION_NVEC:
            allowed = np.flatnonzero(mask[start:start + n])
            start += n
            action.append(int(self.rng.choice(allowed)))
        action[JUMP] = int(action[JUMP] and self.rng.random() < 0.1)
        return np.array(action)


class AimerBot(AIController):
    """Scripted: aims at the enemy with the aimer alone (no correction from its last shot), random weapons."""

    def __init__(self, index: int, aimer: Aimer, power_noise: float = 0.0, seed: int | None = None):
        super().__init__(index, None, aimer, power_noise=power_noise, seed=seed)
        self._next_weapon = int(self.rng.integers(len(WEAPONS) - 1))

    def update(self, game: Game, dt: float) -> None:
        self._decide_in -= dt
        if self._decide_in <= 0 and game.can_act(self.index):
            self._decide_in = self.decision_interval
            me = game.players[self.index]
            weapon = self._next_weapon
            powers = self._suggest(game, weapon, ANGLE_BINS[:8], 0)  # angles toward the enemy
            ok = np.flatnonzero((powers > 10) & (powers < 98))
            a_bin = int(ok[np.argmin(np.abs(ANGLE_BINS[ok] - 50))]) if len(ok) else 4
            ready = (self.target_angle is not None and abs(me.angle - self.target_angle) < 0.5
                     and abs(me.power - (self.target_power or 0)) < 0.5 and me.weapon == weapon)
            fire = ready and game.can_fire(self.index)
            self.act(np.array([a_bin, int(np.flatnonzero(POWER_STEPS == 0)[0]), weapon, 0, 1, 0, int(fire)]), game)
            if fire:  # pick the next weapon (Dirt Bomb excluded: it does no damage)
                self._next_weapon = int(self.rng.integers(len(WEAPONS) - 1))
        self.drive(game, dt)


def pool_policy(pool_dir: Path, rng: np.random.Generator) -> TacticsPolicy | None:
    """A random earlier snapshot of the policy being trained, or None when there is none yet."""
    snaps = sorted(pool_dir.glob("*.npz")) if pool_dir else []
    if not snaps:
        return None
    # favour recent snapshots, but keep old ones in the mix so it doesn't forget how to beat them
    weights = np.linspace(1.0, 3.0, len(snaps))
    path = snaps[int(rng.choice(len(snaps), p=weights / weights.sum()))]
    return TacticsPolicy.load(path, deterministic=False, seed=int(rng.integers(2**31)))


def make_opponent(kind: str, index: int, aimer: Aimer, rng: np.random.Generator, pool_dir: Path | None = None):
    seed = int(rng.integers(2**31))
    if kind == "idle":
        return Idle(index)
    if kind == "random":
        return AIController(index, RandomPolicy(seed), aimer, decision_interval=0.3, seed=seed)
    if kind == "aimer":
        return AimerBot(index, aimer, power_noise=float(rng.uniform(0, 8)), seed=seed)
    if kind == "pool":
        policy = pool_policy(pool_dir, rng)
        if policy is None:
            return AimerBot(index, aimer, seed=seed)
        return AIController(index, policy, aimer, seed=seed)
    raise ValueError(kind)

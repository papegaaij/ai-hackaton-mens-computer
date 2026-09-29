"""Gymnasium environment: one Crosswind duel from the AI's side, a decision every 0.2 s."""
from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from crosswind.ai.aimer import Aimer
from crosswind.ai.controller import ACTION_NVEC, AIController, action_mask
from crosswind.ai.obs import OBS_SIZE
from crosswind.ai.train.opponents import make_opponent
from crosswind.core.game import Game, Phase

FRAME = 1 / 60
FRAMES_PER_STEP = 12  # 0.2 s per decision (the Hard AI's reaction time)
MATCH_LIMIT = 60.0   # s; a match that runs longer is cut off as a draw


class CrosswindEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, wind: float = 1.0, opponents: dict[str, float] | None = None,
                 shaping: float = 1.0, pool_dir: str | None = None):
        self.aimer = Aimer.load()
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (OBS_SIZE,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete(ACTION_NVEC)
        self._seed = seed
        self.set_stage(wind=wind, opponents=opponents or {"idle": 1.0}, shaping=shaping, pool_dir=pool_dir)

    def set_stage(self, wind: float, opponents: dict[str, float], shaping: float, pool_dir: str | None = None):
        """Curriculum knobs: wind strength, opponent mix, and how much miss-distance shaping to give."""
        self.wind = wind
        self.opponents = opponents
        self.shaping = shaping
        self.pool_dir = Path(pool_dir) if pool_dir else None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None  # the constructor's seed, for the very first match
        super().reset(seed=seed)
        rng = self.np_random
        self.game = Game(seed=int(rng.integers(2**31)), countdown=0, wind_strength=self.wind)
        self.side = int(rng.integers(2))
        kinds = list(self.opponents)
        p = np.array([self.opponents[k] for k in kinds])
        self.opponent_kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
        self.me = AIController(self.side, None, self.aimer, seed=int(rng.integers(2**31)))
        self.opp = make_opponent(self.opponent_kind, 1 - self.side, self.aimer, rng, self.pool_dir)
        self._seen_impact = None
        self._stats = {"moved": 0.0, "dealt": 0.0, "taken": 0.0}
        return self.me.observe(self.game), {}

    def action_masks(self) -> np.ndarray:
        return action_mask(self.game, self.side)

    def step(self, action):
        g, me_i = self.game, self.side
        me, them = g.players[me_i], g.players[1 - me_i]
        hp_me, hp_them, x0 = me.hp, them.hp, me.x
        self.me.act(np.asarray(action), g)
        for _ in range(FRAMES_PER_STEP):
            self.me.drive(g, FRAME)
            self.opp.update(g, FRAME)
            g.update(FRAME)
            if g.phase is Phase.GAME_OVER:
                break
        dealt, taken = hp_them - them.hp, hp_me - me.hp
        reward = (dealt - taken) / 100.0
        st = self._stats
        st["dealt"] += dealt
        st["taken"] += taken
        st["moved"] += abs(me.x - x0)
        hit = g.last_impact[me_i]
        if hit is not None and hit is not self._seen_impact:
            self._seen_impact = hit
            miss = math.dist((hit.x, hit.y), (them.x, them.y))
            reward += self.shaping * 0.05 * (1.0 - min(miss, 400.0) / 400.0)
        terminated = g.phase is Phase.GAME_OVER
        truncated = not terminated and g.time >= MATCH_LIMIT
        info = {}
        if terminated or truncated:
            result = 0 if not terminated or g.winner is None else (1 if g.winner == me_i else -1)
            reward += result
            info = {"result": result, "opponent": self.opponent_kind, "shots": list(self.me.shots),
                    "jumps": self.me.jumps, **st}
        return self.me.observe(g), float(reward), terminated, truncated, info

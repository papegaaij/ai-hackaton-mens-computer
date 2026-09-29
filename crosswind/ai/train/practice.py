"""Fire lots of random practice shots and record where each one came down: the aimer's training data.

Every shot is a correct example for the spot it hit ("to land *there*, with this weapon at this angle
in this wind, use this power"), so no shot is wasted.

    python -m crosswind.ai.train.practice --shots 400000
"""
from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from crosswind import config
from crosswind.ai.obs import WIND_TREND_SPAN, canon_angle, canon_dx
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS

MAX_FLIGHT = 12.0  # s after which a shot is given up on


def practice_shot(rng: np.random.Generator) -> tuple | None:
    """One random shot on a random battlefield, in the mirrored frame of the shooter; None if it flew off."""
    g = Game(seed=int(rng.integers(2**31)), countdown=0, wind_strength=1.0)
    g.update(float(rng.uniform(0, 8)))  # let the wind wander to a random state
    before = g.wind
    g.update(WIND_TREND_SPAN)  # and watch which way the wind bar moves, as a player would
    index = int(rng.integers(2))
    me, other = g.players[index], g.players[1 - index]
    other.x = -10_000.0  # out of the way: only the terrain stops the shot
    me.x = float(rng.uniform(config.PLAYER_RADIUS, g.terrain.width - config.PLAYER_RADIUS))
    g._settle(me)
    me.angle = float(rng.uniform(5, 175))
    me.power = float(rng.uniform(15, 100))
    me.weapon = int(rng.integers(len(WEAPONS)))
    wind, trend = g.wind, (g.wind - before) / WIND_TREND_SPAN
    g.events.clear()
    assert g.fire(index)
    t = 0.0
    while g.projectiles and t < MAX_FLIGHT:
        g._tick(config.PHYSICS_DT)
        t += config.PHYSICS_DT
    if not g.events:
        return None
    x = float(np.mean([e.x for e in g.events]))  # a cluster bomb comes down in several places
    y = float(np.mean([e.y for e in g.events]))
    return (me.weapon, canon_angle(me.angle, index), me.power, canon_dx(x - me.x, index), y - me.y,
            canon_dx(wind, index), canon_dx(trend, index), t)


def _worker(args: tuple[int, int]) -> np.ndarray:
    seed, n = args
    rng = np.random.default_rng(seed)
    rows = [r for r in (practice_shot(rng) for _ in range(n)) if r is not None]
    return np.array(rows, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=400_000)
    ap.add_argument("--out", default="runs/practice.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    chunk = 500
    jobs = [(args.seed * 1_000_003 + i, chunk) for i in range(math.ceil(args.shots / chunk))]
    start = time.time()
    parts = []
    with Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_worker, jobs)):
            parts.append(part)
            if i % 50 == 0:
                print(f"{(i + 1) * chunk} shots, {time.time() - start:.0f} s", flush=True)
    data = np.concatenate(parts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, shots=data)
    print(f"{len(data)} usable shots of {len(jobs) * chunk} in {time.time() - start:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()

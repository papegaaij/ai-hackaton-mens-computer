"""Play headless matches and report how an AI does: wins, damage, weapons used, driving and jumping.

    python -m crosswind.ai.train.evaluate models/tactics_hard.npz --vs random aimer
    python -m crosswind.ai.train.evaluate --difficulty Hard --vs Easy
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

from crosswind.ai.aimer import Aimer
from crosswind.ai.controller import DIFFICULTIES, AIController, TacticsPolicy, make_ai
from crosswind.ai.rules import RuleBasedController
from crosswind.ai.train.env import FRAME, MATCH_LIMIT
from crosswind.ai.train.opponents import make_opponent
from crosswind.core.game import Game, Phase
from crosswind.core.weapons import WEAPONS


def make_player(spec: str, index: int, aimer: Aimer, rng: np.random.Generator):
    if spec in DIFFICULTIES:
        return make_ai(index, spec)
    if spec == "Rules":
        return RuleBasedController(index, seed=int(rng.integers(2**31)))
    if spec.endswith(".npz"):
        return AIController(index, TacticsPolicy.load(spec), aimer, seed=int(rng.integers(2**31)))
    return make_opponent(spec, index, aimer, rng)


def cross(g: Game) -> None:
    """Swap the tanks' places, as if they had driven past each other: each now has its enemy behind it."""
    a, b = g.players
    a.x, b.x = b.x, a.x
    for p in g.players:
        g._settle(p)


def play(args: tuple[str, str, int, bool]) -> dict:
    """One match of `me` against `vs`; `me` plays player 1 on even seeds and player 2 on odd ones."""
    me_spec, vs_spec, seed, crossed = args
    rng = np.random.default_rng(seed)
    aimer = Aimer.load()
    g = Game(seed=seed, countdown=0)
    if crossed:
        cross(g)
    side = seed % 2
    me, vs = make_player(me_spec, side, aimer, rng), make_player(vs_spec, 1 - side, aimer, rng)
    ctrls = (me, vs) if side == 0 else (vs, me)
    moved, x = 0.0, g.players[side].x
    while g.phase is Phase.PLAYING and g.time < MATCH_LIMIT:
        for c in ctrls:
            c.update(g, FRAME)
        g.update(FRAME)
        moved += abs(g.players[side].x - x)
        x = g.players[side].x
    result = 0 if g.winner is None or g.phase is not Phase.GAME_OVER else (1 if g.winner == side else -1)
    return {"result": result, "dealt": 100 - g.players[1 - side].hp, "taken": 100 - g.players[side].hp,
            "shots": np.array(me.shots), "jumps": me.jumps, "moved": moved, "time": g.time}


def evaluate(me: str, vs: str, matches: int, workers: int, crossed: bool = False) -> dict:
    with Pool(workers) as pool:
        rows = pool.map(play, [(me, vs, s, crossed) for s in range(matches)])
    res = np.array([r["result"] for r in rows])
    shots = sum(r["shots"] for r in rows)
    return {"win": np.mean(res == 1), "loss": np.mean(res == -1), "draw": np.mean(res == 0),
            "dealt": np.mean([r["dealt"] for r in rows]), "taken": np.mean([r["taken"] for r in rows]),
            "weapons": shots / max(1, shots.sum()), "jumps": np.mean([r["jumps"] for r in rows]),
            "moved": np.mean([r["moved"] for r in rows]), "time": np.mean([r["time"] for r in rows]),
            "drove": np.mean([r["moved"] > 20 for r in rows]), "jumped": np.mean([r["jumps"] > 0 for r in rows])}


def report(me: str, vs: str, r: dict) -> None:
    print(f"{me} vs {vs}: win {r['win']:.0%}  loss {r['loss']:.0%}  draw {r['draw']:.0%}  "
          f"dealt {r['dealt']:.0f}  taken {r['taken']:.0f}  match {r['time']:.0f} s")
    print(f"  drove in {r['drove']:.0%} of matches ({r['moved']:.0f} px avg), jumped in {r['jumped']:.0%} "
          f"({r['jumps']:.1f} avg)")
    print("  weapons: " + ", ".join(f"{w.name} {share:.0%}" for w, share in zip(WEAPONS, r["weapons"])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="a tactics .npz (or use --difficulty)")
    ap.add_argument("--difficulty", choices=[*DIFFICULTIES, "Rules"])
    ap.add_argument("--vs", nargs="+", default=["random", "aimer"],
                    help="idle, random, aimer, Rules, a difficulty name, or a .npz")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--crossed", action="store_true", help="start with the tanks swapped (enemy behind)")
    args = ap.parse_args()
    me = args.difficulty or args.model
    if not me:
        ap.error("give a model or --difficulty")
    for vs in args.vs:
        report(me, vs, evaluate(me, vs, args.matches, args.workers, args.crossed))


if __name__ == "__main__":
    main()

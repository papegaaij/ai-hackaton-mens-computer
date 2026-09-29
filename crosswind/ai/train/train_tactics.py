"""Train the tactics policy with MaskablePPO, through a curriculum that ends in self-play.

    python -m crosswind.ai.train.train_tactics --out runs/tactics

Stages: a sitting duck in light wind, then full wind and a random player, then the scripted aimer bot,
then a league of its own earlier snapshots (plus the bots, so it doesn't forget how to beat them).
Snapshots are exported to <out>/pool/ as they go; they are both the self-play opponents and the
candidates for the Easy / Medium / Hard models.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from crosswind.ai.train.env import CrosswindEnv
from crosswind.ai.train.export import export

STAGES = [  # (million steps of 0.2 s, stage settings); a short run that fits in a hackathon slot
    (0.4, dict(wind=0.3, opponents={"idle": 1.0}, shaping=1.0)),
    (0.3, dict(wind=1.0, opponents={"idle": 0.3, "random": 0.7}, shaping=1.0)),
    (0.4, dict(wind=1.0, opponents={"random": 0.3, "aimer": 0.7}, shaping=0.5)),
    (2.0, dict(wind=1.0, opponents={"random": 0.1, "aimer": 0.3, "pool": 0.6}, shaping=0.2)),
]


class Curriculum(BaseCallback):
    def __init__(self, out: Path, snapshot_every: int):
        super().__init__()
        self.out = out
        self.snapshot_every = snapshot_every
        self.bounds = np.cumsum([m * 1e6 for m, _ in STAGES])
        self.stage = -1
        self.results = defaultdict(lambda: deque(maxlen=300))
        self.shots = np.zeros(8)
        self.moves = deque(maxlen=300)
        self.jumps = deque(maxlen=300)
        self.next_snapshot = snapshot_every

    def _set_stage(self) -> None:
        stage = int(np.searchsorted(self.bounds, self.num_timesteps, side="right"))
        stage = min(stage, len(STAGES) - 1)
        if stage != self.stage:
            self.stage = stage
            settings = dict(STAGES[stage][1], pool_dir=str(self.out / "pool"))
            self.training_env.env_method("set_stage", **settings)
            print(f"stage {stage}: {settings}", flush=True)

    def _on_training_start(self) -> None:
        self._set_stage()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "result" in info:
                self.results[info["opponent"]].append(info["result"])
                self.shots += info["shots"]
                self.moves.append(info["moved"])
                self.jumps.append(info["jumps"])
        if self.num_timesteps >= self.next_snapshot:
            self.next_snapshot += self.snapshot_every
            name = f"step_{self.num_timesteps // 1000:06d}k"
            export(self.model, self.out / "pool" / f"{name}.npz")
            self.model.save(self.out / "checkpoints" / name)
        self._set_stage()
        return True

    def _on_rollout_end(self) -> None:
        for kind, res in self.results.items():
            r = np.array(res)
            self.logger.record(f"win/{kind}", float(np.mean(r == 1)))
            self.logger.record(f"loss/{kind}", float(np.mean(r == -1)))
        if self.shots.sum():
            for i, share in enumerate(self.shots / self.shots.sum()):
                self.logger.record(f"weapons/{i}", float(share))
        if self.moves:
            self.logger.record("play/moved_px", float(np.mean(self.moves)))
            self.logger.record("play/jumps", float(np.mean(self.jumps)))
        self.logger.record("curriculum/stage", self.stage)
        self.shots *= 0.9  # a moving window over recent matches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/tactics")
    ap.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--steps", type=float, default=sum(m for m, _ in STAGES), help="million steps")
    ap.add_argument("--resume", help="checkpoint .zip to continue from")
    ap.add_argument("--snapshot-every", type=int, default=100_000, help="steps between snapshots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)  # a small network: the CPU beats the GPU's per-call overhead here
    out = Path(args.out)
    (out / "pool").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([lambda i=i: CrosswindEnv(seed=args.seed * 1000 + i) for i in range(args.envs)]))
    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device="cpu")
    else:
        model = MaskablePPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=4096, n_epochs=5, gamma=0.99,
            gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, seed=args.seed, device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), verbose=1)
    model.set_logger(configure(str(out / "log"), ["stdout", "csv"]))
    model.learn(int(args.steps * 1e6), callback=Curriculum(out, snapshot_every=args.snapshot_every),
                reset_num_timesteps=not args.resume)
    model.save(out / "final")
    export(model, out / "final.npz")
    print(f"saved {out / 'final.zip'} and {out / 'final.npz'}")


if __name__ == "__main__":
    main()

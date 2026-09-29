"""The aimer: a network that learned, from practice shots, which power lands a shot where you want it.

It never computes a flight path. It was trained on where earlier practice shots came down (see
`crosswind.ai.train.practice`), the way a player gets a feel for their shots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from crosswind.ai.nets import MODELS_DIR, MLP
from crosswind.ai.obs import aimer_inputs


class Aimer:
    def __init__(self, net: MLP):
        self.net = net

    @classmethod
    def load(cls, path: str | Path = MODELS_DIR / "aimer.npz") -> Aimer:
        return cls(MLP.load(path)[0])

    def suggest(self, weapon, angles, dx, dy, wind, trend) -> np.ndarray:
        """Power (unclipped, so > 100 means out of reach) per shot; all inputs in the mirrored frame."""
        return self.net(aimer_inputs(weapon, angles, dx, dy, wind, trend))[:, 0] * 100.0

"""A tiny numpy forward pass for the networks trained in `crosswind.ai.train` (saved as .npz)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0)}


class MLP:
    """Dense layers W0,b0 .. Wn,bn with `activation` between them and a linear output."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray], activation: str = "tanh"):
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]
        self.activation = activation
        self._act = _ACTIVATIONS[activation]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self._act(x @ w + b)
        return x @ self.weights[-1] + self.biases[-1]

    def save(self, path: str | Path, **extra: np.ndarray) -> None:
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)} | {f"b{i}": b for i, b in enumerate(self.biases)}
        np.savez(path, activation=np.array(self.activation), **arrays, **extra)

    @classmethod
    def load(cls, path: str | Path) -> tuple[MLP, dict[str, np.ndarray]]:
        """The network and any extra arrays saved with it."""
        with np.load(path) as data:
            n = sum(1 for k in data.files if k.startswith("W"))
            net = cls([data[f"W{i}"] for i in range(n)], [data[f"b{i}"] for i in range(n)], str(data["activation"]))
            extra = {k: data[k] for k in data.files if k[0] not in "Wb" and k != "activation"}
        return net, extra

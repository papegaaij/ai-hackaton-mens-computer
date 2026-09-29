"""Turn a trained MaskablePPO policy into the small numpy model the game loads.

    python -m crosswind.ai.train.export runs/tactics/final.zip models/tactics_hard.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

from crosswind.ai.nets import MLP


def policy_to_mlp(model) -> MLP:
    """The actor half of the policy: its hidden layers and the action logits layer."""
    policy = model.policy
    layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [policy.action_net]
    acts = {type(m) for m in policy.mlp_extractor.policy_net if not isinstance(m, nn.Linear)}
    assert acts == {nn.Tanh}, acts
    return MLP([m.weight.detach().cpu().numpy().T for m in layers],
               [m.bias.detach().cpu().numpy() for m in layers], "tanh")


def export(model, path: str | Path) -> None:
    policy_to_mlp(model).save(path)


def main() -> None:
    from sb3_contrib import MaskablePPO
    src, dst = sys.argv[1:3]
    export(MaskablePPO.load(src, device="cpu"), dst)
    print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()

"""Controllers decide what a player does. Humans and (later) AI agents are interchangeable."""
from __future__ import annotations

from typing import Protocol

from crosswind.core.actions import FireAction
from crosswind.core.game import Game


class Controller(Protocol):
    def update(self, game: Game, dt: float) -> FireAction | None:
        """Called every frame for this controller's own player. Return an action to fire."""
        ...

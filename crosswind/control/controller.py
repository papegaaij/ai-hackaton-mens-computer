"""Controllers decide what a player does. Humans and AI agents are interchangeable."""
from __future__ import annotations

from typing import Protocol

from crosswind.core.game import Game


class Controller(Protocol):
    index: int
    keys: object  # the player's KeyMap: the HUD shows these key caps for this player
    label: str    # shown next to the player name in the HUD ("" for a human)

    def update(self, game: Game, dt: float) -> None:
        """Called every frame. Acts only through Game's input API: adjust_aim, move, jump, cycle_weapon, fire."""
        ...

    def held(self) -> set[str]:
        """KeyMap field names of the keys this player is pressing right now, for the HUD."""
        ...

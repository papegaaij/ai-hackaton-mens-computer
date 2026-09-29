"""The aim dial feel shared by humans and the AI: a tap nudges it slowly, holding the key speeds it up."""
from __future__ import annotations

from crosswind import config


class AimRamp:
    """Speed of one aim axis (angle or power) as a fraction of full speed, for the direction held."""

    def __init__(self) -> None:
        self.direction = 0
        self.held_for = 0.0

    def step(self, direction: int, dt: float) -> float:
        self.held_for = self.held_for + dt if direction == self.direction else 0.0
        self.direction = direction
        return direction * min(1.0, config.AIM_FINE + (1 - config.AIM_FINE) * self.held_for / config.AIM_RAMP)

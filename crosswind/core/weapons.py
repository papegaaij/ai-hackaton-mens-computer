"""Data-driven weapon definitions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Weapon:
    name: str
    mass: float          # higher mass = less affected by wind
    drag: float          # linear drag coefficient (1/s)
    blast_radius: float  # crater and damage radius (px)
    damage: float        # damage at the centre of the blast
    speed_factor: float = 1.0


WEAPONS: tuple[Weapon, ...] = (
    Weapon("Plasma Orb", mass=1.0, drag=0.05, blast_radius=38, damage=40),
    Weapon("Spore Pod", mass=0.35, drag=0.25, blast_radius=30, damage=30, speed_factor=1.1),
    Weapon("Heavy Slug", mass=3.0, drag=0.01, blast_radius=22, damage=55, speed_factor=0.9),
)

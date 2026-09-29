"""Data-driven weapon definitions. Special behaviours are switched on by their fields (all off by default)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weapon:
    name: str
    mass: float          # higher mass = less affected by wind
    drag: float          # linear drag coefficient (1/s)
    blast_radius: float  # crater and damage radius (px); mound radius for dirt
    damage: float        # damage at the centre of the blast
    speed_factor: float = 1.0
    energy: float = 35.0          # energy a shot costs (the bar holds ENERGY_MAX)
    size: float = 3.0             # draw radius of the projectile (px)
    thrust: float = 0.0           # rocket: acceleration along the flight direction (px/s^2)...
    burn: float = 0.0             # ...for this many seconds after launch
    split: int = 0                # cluster: breaks into this many children at the top of its arc
    split_into: Weapon | None = None
    split_spread: float = 0.0     # horizontal speed difference between neighbouring children (px/s)
    drill: float = 0.0            # driller: bores on through rock for this many seconds after hitting it
    fuse: float = 0.0             # bouncer: bounces around for this many seconds after first touching ground
    bounce: float = 0.5           # fraction of the speed into the ground that is kept on a bounce
    dirt: bool = False            # builds a mound of earth instead of blasting a crater


BOMBLET = Weapon("Bomblet", mass=0.8, drag=0.05, blast_radius=20, damage=14, size=2)

WEAPONS: tuple[Weapon, ...] = (
    Weapon("Spark", mass=0.4, drag=0.15, blast_radius=14, damage=8, speed_factor=1.15, energy=12, size=2),
    Weapon("Plasma Orb", mass=1.0, drag=0.05, blast_radius=36, damage=30, energy=35),
    Weapon("Rocket", mass=3.0, drag=0.02, blast_radius=26, damage=32, speed_factor=0.7, energy=45,
           thrust=500, burn=0.6),
    Weapon("Bouncer", mass=1.2, drag=0.05, blast_radius=32, damage=32, energy=40, size=4, fuse=1.5),
    Weapon("Driller", mass=1.5, drag=0.03, blast_radius=30, damage=30, energy=40, drill=0.35),
    Weapon("Cluster Bomb", mass=1.2, drag=0.05, blast_radius=16, damage=10, energy=60, size=5,
           split=5, split_into=BOMBLET, split_spread=45),
    Weapon("Dirt Bomb", mass=1.0, drag=0.05, blast_radius=40, damage=0, energy=30, size=4, dirt=True),
    Weapon("Megaton", mass=3.5, drag=0.01, blast_radius=60, damage=70, speed_factor=0.85, energy=100, size=6),
)

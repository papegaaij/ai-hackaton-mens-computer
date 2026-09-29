"""Sound design as data: 'Dust & Metal x Cinematic' (see the Crosswind sound pack design doc).

Each recipe is a list of layers. Dust & Metal gives the attack and grit (metal, clanks, debris);
Cinematic gives the body and tail (energy tones, sub-bass booms, rumble).
"""
from __future__ import annotations

from dataclasses import dataclass


def db(value: float) -> float:
    return 10 ** (value / 20)


# Mix levels relative to Megaton (0 dB), from the design doc's mix table.
LEVEL = {
    "big": db(0),          # Megaton, player down, win
    "hit": db(-6),
    "launch": db(-10),
    "detail": db(-14),     # bounces, splits, drill
    "cue": db(-12),        # countdown, energy, weapon switch
    "move": db(-18),       # driving, jump
    "gust": db(-20),
}


@dataclass(frozen=True)
class Layer:
    sample: str
    gain: float = 1.0
    rate: float = 1.0       # playback speed = pitch (0.5 = one octave down)
    delay: float = 0.0      # seconds after the event
    far: bool = False       # muffle (lose the highs) when the event is far from both tanks
    jitter: float = 0.05    # random pitch variation, +/- fraction


def L(sample, gain=1.0, rate=1.0, delay=0.0, far=True, jitter=0.05) -> Layer:
    return Layer(sample, gain, rate, delay, far, jitter)


# ---- launches, per weapon: attack (metal) + tone (cinematic) ----
LAUNCH: dict[str, list[Layer]] = {
    "Spark": [L("impactTin_medium_000", .6, 1.6), L("laserSmall_001", .45, 1.1)],
    "Plasma Orb": [L("impactPlate_heavy_001", .8, 1.3), L("laserLarge_000", .5)],
    "Rocket": [L("impactMetal_heavy_000", .8, .8), L("thrusterFire_000", .9, 1.0, .03), L("phaserUp3", .3, .7)],
    "Bouncer": [L("impactMetal_002", .8, 1.25), L("pluck_001", .3, .6)],
    "Driller": [L("impactMining_000", .8, 1.2), L("computerNoise_001", .35, .8)],
    "Cluster Bomb": [L("impactPunch_heavy_000", .7, .9), L("impactMetal_002", .45, .8)],
    "Dirt Bomb": [L("impactSoft_heavy_000", .9), L("drop_002", .3, .6)],
    "Megaton": [L("impactPunch_heavy_000", 1.0, .6), L("lowFrequency_explosion_000", .6, 1.3, far=False),
                L("impactMetal_heavy_000", .5, .5, .02)],
}

# ---- explosions by size: transient + body/sub + debris ----
EXPLODE_SMALL = [L("explosionCrunch_003", .8, 1.3), L("impactTin_medium_000", .4, .9)]
EXPLODE_MEDIUM = [L("impactMining_000", .8, .8), L("explosionCrunch_000", .8),
                  L("lowFrequency_explosion_000", .35, 1.2, far=False)]
EXPLODE_BIG = ([L("impactPunch_heavy_000", 1.0, .5), L("impactMetal_heavy_000", .8, .45, .03),
                L("lowFrequency_explosion_001", 1.0, far=False), L("lowFrequency_explosion_000", .8, .7, .05, far=False),
                L("explosionCrunch_000", .6, .6)]
               + [L("impactPlank_medium_000", .3, .6 + .08 * i, .5 + i * .18, jitter=.15) for i in range(6)])

DIRT = [L("impactSoft_heavy_000", 1.0, .8)] + [L("impactPlank_medium_000", .35, .7 + .1 * i, .12 + i * .1, jitter=.2)
                                              for i in range(4)]
SPLIT = [L("impactMetal_002", .7, 1.3)] + [L("impactTin_medium_000", .5, 1.0 + .1 * i, .05 + i * .06, jitter=.15)
                                           for i in range(5)]
DRILL = [L("impactMining_000", .9, .6), L("impactSoft_heavy_000", .6, .5)]
HIT = [L("impactPlate_heavy_001", .9, .9), L("glitch_001", .5)]
DOWN = [L("impactMetal_heavy_000", 1.0, .6), L("lowFrequency_explosion_001", .9, .9, .05, far=False)]
WIN = ([L("impactBell_heavy_000", .6, r * .8, i * .18, jitter=0) for i, r in enumerate((1, 1.26, 1.5, 2))]
       + [L("powerUp7", .7, delay=.75, jitter=0), L("forceField_001", .5, .8, .9, jitter=0)])
DRAW = [L("impactBell_heavy_000", .7, .6, jitter=0), L("forceField_001", .5, .6, .3, jitter=0)]
GO = [L("impactPunch_heavy_000", 1.0, .9, jitter=0), L("lowFrequency_explosion_000", .7, 1.1, far=False, jitter=0)]
READY = [L("click_002", .8, 1.3)]
NO_ENERGY = [L("impactTin_medium_000", .7, .7)]
JUMP = [L("impactSoft_heavy_000", .8, 1.3)]
LAND = [L("impactMining_000", .6, 1.2)]
GUST = [L("forceField_001", .5, .5, jitter=.1)]
UI = [L("impactTin_medium_000", .5, 1.8)]


def count(n: int) -> list[Layer]:
    """Countdown 3-2-1: a low metal bell, rising in pitch per number."""
    return [L("impactBell_heavy_000", .8, .7 + .12 * (3 - n), jitter=0)]


def switch(slot: int) -> list[Layer]:
    """Weapon switch: a chamber click whose pitch tells the weapon slot."""
    return [L("click_002", .8, .8 + .07 * slot, jitter=0)]


def bounce(n: int) -> list[Layer]:
    """Bouncer: each bounce a metal clang, lower and softer than the last."""
    return [L("impactMetal_heavy_000", max(.25, .8 - .12 * n), max(.6, 1.4 - .15 * n), jitter=.03)]


# Loops: (sample or generated name, base gain)
DRONE = "spaceEngineLow_000"
DRIVE = "thrusterFire_000"

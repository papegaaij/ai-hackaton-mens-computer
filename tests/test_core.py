import math
import sys

import numpy as np

from crosswind import config
from crosswind.core import physics
from crosswind.core.actions import FireAction
from crosswind.core.game import Game, Phase
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import Weapon


NO_DRAG = Weapon("test", mass=1.0, drag=0.0, blast_radius=10, damage=10)


def fly(p, wind, height=10_000, t=None, ground_y=None):
    steps = 0
    while True:
        physics.step(p, wind, height, config.PHYSICS_DT)
        steps += 1
        if ground_y is not None and p.y >= ground_y and p.vy > 0:
            return p
        if t is not None and steps * config.PHYSICS_DT >= t:
            return p


def test_no_wind_range_matches_analytic():
    p = physics.launch(0, 0, 45, 50, NO_DRAG, 0)
    v = 0.5 * config.MAX_LAUNCH_SPEED
    expected = v * v / config.GRAVITY
    fly(p, wind=0.0, ground_y=0)
    assert abs(p.x - expected) / expected < 0.02


def test_wind_pushes_shot_downwind():
    calm = fly(physics.launch(0, 0, 60, 60, NO_DRAG, 0), 0.0, ground_y=0)
    windy = fly(physics.launch(0, 0, 60, 60, NO_DRAG, 0), 50.0, ground_y=0)
    assert windy.x > calm.x


def test_light_weapon_drifts_more():
    light = Weapon("l", mass=0.3, drag=0.0, blast_radius=1, damage=1)
    a = fly(physics.launch(0, 0, 70, 60, NO_DRAG, 0), 40.0, ground_y=0)
    b = fly(physics.launch(0, 0, 70, 60, light, 0), 40.0, ground_y=0)
    assert b.x > a.x


def test_carve_removes_pixels_but_not_bedrock():
    t = Terrain(200, 200, np.random.default_rng(1))
    before = t.mask.sum()
    t.carve_circle(100, 199, 50)
    assert t.mask.sum() < before
    assert t.mask[-config.BEDROCK:, :].all()


def test_turn_switches_and_wind_changes():
    g = Game(seed=3)
    assert g.current == 0 and g.phase is Phase.AIMING
    g.fire(FireAction(90, 40, 0))  # straight up, lands near self
    for _ in range(2000):
        g.update(1 / 60)
        if g.phase is Phase.AIMING:
            break
    assert g.current == 1 and g.turn == 2


def test_game_over_when_hp_zero():
    g = Game(seed=3)
    g.players[1].hp = 1
    g.players[1].x, g.players[1].y = g.players[0].x + 5, g.players[0].y
    g._explode(g.players[1].x, g.players[1].y, 40, 100)
    g.update(config.RESOLVE_DELAY + 0.1)
    assert g.phase is Phase.GAME_OVER and g.winner == 0


def test_headless_random_play_without_pygame():
    rng = np.random.default_rng(0)
    g = Game(seed=0)
    for _ in range(100):
        if g.phase is Phase.GAME_OVER:
            g = Game(seed=int(rng.integers(1000)))
        g.fire(FireAction(rng.uniform(10, 170), rng.uniform(20, 100), int(rng.integers(3))))
        for _ in range(3000):
            g.update(1 / 30)
            if g.phase in (Phase.AIMING, Phase.GAME_OVER):
                break
    obs = g.observe(0)
    assert obs.dtype == np.float32 and np.isfinite(obs).all()


def test_core_does_not_import_pygame():
    import subprocess
    code = "import sys, crosswind.core.game; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

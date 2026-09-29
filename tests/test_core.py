import math
import sys

import numpy as np

from crosswind import config
from crosswind.core import physics
from crosswind.core.actions import FireAction
from crosswind.core.game import Game, Phase
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import Weapon
from crosswind.core.wind import Wind


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


def test_no_firing_during_countdown():
    g = Game(seed=3)
    assert not g.fire(0, FireAction(90, 40, 0))
    g.update(config.START_COUNTDOWN + 0.1)
    assert g.fire(0, FireAction(90, 40, 0))


def test_fire_starts_recharge_and_blocks_refire():
    g = Game(seed=3, countdown=0)
    assert g.fire(0, FireAction(90, 40, 0))
    assert not g.fire(0, FireAction(90, 40, 0))
    assert len(g.projectiles) == 1
    g.update(config.RELOAD_TIME - 0.1)
    assert not g.fire(0, FireAction(90, 40, 0))
    g.update(0.2)
    assert g.fire(0, FireAction(90, 40, 0))


def test_both_players_shoot_at_the_same_time():
    g = Game(seed=3, countdown=0)
    assert g.fire(0, FireAction(60, 70, 0))
    assert g.fire(1, FireAction(120, 70, 2))
    assert {p.owner for p in g.projectiles} == {0, 1}
    for _ in range(2000):
        g.update(1 / 60)
        if not g.projectiles:
            break
    assert not g.projectiles  # both landed or left the field


def test_fuel_refills_slowly():
    g = Game(seed=3, countdown=0)
    p = g.players[0]
    p.fuel = 0.0
    g.update(1.0)
    assert abs(p.fuel - config.FUEL_REGEN) < 1.0
    g.update(60.0)
    assert p.fuel == config.PLAYER_FUEL


def test_wind_shifts_constantly_and_smoothly():
    wind = Wind(np.random.default_rng(5))
    values = []
    for _ in range(int(60 / config.PHYSICS_DT)):
        wind.step(config.PHYSICS_DT)
        values.append(wind.value)
    values = np.array(values)
    assert np.all(np.abs(values) <= config.WIND_MAX)
    # never calm for long: over every 5 s window the wind moves noticeably
    per_window = values.reshape(12, -1)
    assert np.all(per_window.max(axis=1) - per_window.min(axis=1) > config.WIND_MAX * 0.05)
    # but smoothly: tiny change per physics tick
    assert np.abs(np.diff(values)).max() < config.WIND_MAX * 2 * config.PHYSICS_DT / config.WIND_RESPONSE
    # and swings over the whole range during a match
    assert values.max() - values.min() > config.WIND_MAX


def test_wind_is_deterministic_per_seed():
    a, b = Game(seed=11), Game(seed=11)
    series = []
    for g in (a, b):
        vals = []
        for _ in range(300):
            g.update(1 / 30)
            vals.append(g.wind)
        series.append(vals)
    assert series[0] == series[1]


def test_game_over_when_hp_zero():
    g = Game(seed=3)
    g.players[1].hp = 1
    g.players[1].x, g.players[1].y = g.players[0].x + 5, g.players[0].y
    g._explode(g.players[1].x, g.players[1].y, 40, 100)
    g.update(config.END_DELAY + 0.1)
    assert g.phase is Phase.GAME_OVER and g.winner == 0


def test_mutual_kill_is_a_draw():
    g = Game(seed=3)
    for p in g.players:
        p.hp = 1
    g.players[1].x, g.players[1].y = g.players[0].x + 5, g.players[0].y
    g._explode(g.players[0].x, g.players[0].y, 40, 100)
    g.update(config.END_DELAY + 0.1)
    assert g.phase is Phase.GAME_OVER and g.winner is None


def test_dead_player_cannot_fire():
    g = Game(seed=3, countdown=0)
    g.players[1].hp = 0
    assert not g.fire(1, FireAction(120, 60, 0))


def test_headless_random_play_without_pygame():
    rng = np.random.default_rng(0)
    g = Game(seed=0, countdown=0)
    games = 1
    for _ in range(int(300 / (1 / 30))):  # five simulated minutes at 30 fps
        if g.phase is Phase.GAME_OVER:
            g = Game(seed=int(rng.integers(1000)), countdown=0)
            games += 1
        for i in (0, 1):
            g.move(i, int(rng.integers(-1, 2)), 1 / 30)
            if g.players[i].ready:
                g.fire(i, FireAction(rng.uniform(10, 170), rng.uniform(20, 100), int(rng.integers(3))))
        g.update(1 / 30)
    obs = g.observe(0)
    assert obs.dtype == np.float32 and np.isfinite(obs).all()
    assert games > 1  # matches actually end


def test_core_does_not_import_pygame():
    import subprocess
    code = "import sys, crosswind.core.game; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

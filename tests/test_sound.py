import os

import pytest

from crosswind import config
from crosswind.core.actions import FireAction
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS


def names(game):
    return [s.name for s in game.sounds]


def run(game, seconds):
    for _ in range(int(seconds * 60)):
        game.update(1 / 60)


def test_countdown_emits_count_and_go():
    g = Game(seed=1)
    run(g, config.START_COUNTDOWN + 0.1)
    counts = [s.size for s in g.sounds if s.name == "count"]
    assert counts == [3, 2, 1] and "go" in names(g)


def test_launch_explode_and_hit_events():
    g = Game(seed=1, countdown=0)
    assert g.fire(0, FireAction(90, 30, 1))  # Plasma Orb straight up: lands next to the shooter
    run(g, 4)
    launch = next(s for s in g.sounds if s.name == "launch")
    assert launch.player == 0 and launch.weapon == WEAPONS[1].name
    assert "explode" in names(g)
    g.sounds.clear()
    p1 = g.players[1]
    g._explode(p1.x, p1.y, 36, 200, owner=0, weapon="Plasma Orb")
    assert names(g) == ["explode", "hit", "down"]


def test_no_energy_and_switch_events():
    g = Game(seed=1, countdown=0)
    g.players[0].energy = 0
    assert not g.fire(0, FireAction(45, 50, 1))
    g.cycle_weapon(0, 1)
    assert names(g) == ["no_energy", "switch"]


def test_sound_queue_is_bounded_when_nobody_listens():
    g = Game(seed=1, countdown=0)
    for _ in range(1000):
        g.cycle_weapon(0, 1)
    assert len(g.sounds) <= 256


# ---- engine (dummy audio driver: no speakers needed) ----------------------

@pytest.fixture(scope="module")
def sound_engine():
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    from crosswind.audio.engine import SoundEngine
    eng = SoundEngine()
    if not eng.enabled:
        pytest.skip("no audio driver available")
    yield eng
    pygame.quit()


def test_every_recipe_sample_exists():
    from crosswind.audio import recipes as R
    from crosswind.audio.engine import SOUND_DIR
    layers = [l for v in R.LAUNCH.values() for l in v]
    for name in dir(R):
        value = getattr(R, name)
        if isinstance(value, list) and value and isinstance(value[0], R.Layer):
            layers += value
    layers += R.count(3) + R.switch(7) + R.bounce(4)
    missing = {l.sample for l in layers if not (SOUND_DIR / f"{l.sample}.ogg").exists()}
    assert not missing
    assert set(R.LAUNCH) == {w.name for w in WEAPONS}


def test_engine_plays_a_whole_match(sound_engine):
    import numpy as np
    rng = np.random.default_rng(0)
    g = Game(seed=2)
    for step in range(60 * 25):
        if step % 20 == 0:
            i = step // 20 % 2
            g.cycle_weapon(i, 1)
            g.fire(i, FireAction(rng.uniform(20, 160), rng.uniform(30, 90), g.players[i].weapon))
        g.move(0, 1, 1 / 60)
        g.update(1 / 60)
        sound_engine.update(g, 1 / 60)
        assert not g.sounds  # consumed every frame
    sound_engine.toggle_mute()
    sound_engine.update(g, 1 / 60)
    sound_engine.toggle_mute()

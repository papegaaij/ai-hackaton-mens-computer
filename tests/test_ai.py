import ast
import os
from pathlib import Path

import numpy as np
import pytest

from crosswind import config
from crosswind.ai.aimer import Aimer
from crosswind.ai.controller import ACTION_NVEC, FIRE, AIController, action_mask, make_ai, models_available
from crosswind.ai.obs import ANGLE_BINS, OBS_SIZE, POWER_STEPS, canon_angle
from crosswind.core.game import Game, Phase
from crosswind.core.weapons import WEAPONS

RUNTIME = ["__init__", "nets", "obs", "aimer", "controller", "rules"]  # what runs while playing
AI_DIR = Path(__file__).resolve().parents[1] / "crosswind" / "ai"

pytestmark = pytest.mark.skipif(not models_available(), reason="no trained models in models/")


def test_runtime_ai_cannot_predict_shots():
    """The playing AI may not simulate: no physics, no stepping or copying the game, no torch."""
    for name in RUNTIME:
        tree = ast.parse((AI_DIR / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                assert not any("physics" in m or "torch" in m or m == "copy" for m in mods), (name, mods)
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"_tick", "_tick_projectile", "projectiles_after", "deepcopy"}, name
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                assert not (isinstance(node.func.value, ast.Name) and node.func.value.id in {"game", "g"}), name


class Fixed:
    """A policy that always makes the same decision."""

    def __init__(self, **choice):
        a = np.zeros(len(ACTION_NVEC), dtype=int)
        a[1] = int(np.flatnonzero(POWER_STEPS == 0)[0])
        a[4] = 1  # don't drive
        for k, v in choice.items():
            a[["angle", "power", "weapon", "target", "move", "jump", "fire"].index(k)] = v
        self.action = a

    def __call__(self, obs, mask):
        return self.action


def bin_of(angle):
    return int(np.flatnonzero(ANGLE_BINS == angle)[0])


@pytest.mark.parametrize("index", [0, 1])
def test_ai_turns_the_barrel_at_human_speed_and_shows_the_keys(index):
    g = Game(seed=4, countdown=0)
    ai = AIController(index, Fixed(angle=bin_of(80)), Aimer.load())
    p = g.players[index]
    start = canon_angle(p.angle, index == 1)
    prev = p.angle
    for _ in range(30):
        ai.update(g, 1 / 60)
        g.update(1 / 60)
        assert abs(p.angle - prev) <= config.ANGLE_SPEED / 60 + 1e-9
        prev = p.angle
    assert canon_angle(p.angle, index == 1) > start  # raised toward 80 (mirrored for player 2)
    # the key that turns the barrel that way: angle_left raises the angle number, as for a human
    key, other = ("angle_left", "angle_right") if index == 0 else ("angle_right", "angle_left")
    assert key in ai.held() and other not in ai.held()


def test_ai_fire_and_weapon_keys_light_up_briefly():
    g = Game(seed=4, countdown=0)
    ai = AIController(0, Fixed(weapon=2, fire=1), Aimer.load())
    seen = []
    for _ in range(60):
        ai.update(g, 1 / 60)
        g.update(1 / 60)
        seen.append(ai.held())
    assert any("weapon" in h for h in seen) and any("fire" in h for h in seen)
    assert g.players[0].weapon == 2
    assert not all("fire" in h for h in seen)  # a tap, not held forever


def test_ai_cannot_fire_when_the_rules_say_no():
    g = Game(seed=4)  # countdown running
    assert not action_mask(g, 0)[sum(ACTION_NVEC[:FIRE]) + 1]
    ai = AIController(0, Fixed(fire=1), Aimer.load())
    for _ in range(60):
        ai.update(g, 1 / 60)
        g.update(1 / 60)
    assert not g.projectiles and sum(ai.shots) == 0


def mirror(g: Game) -> Game:
    m = Game(seed=4, countdown=0)
    m.terrain.mask = g.terrain.mask[:, ::-1].copy()
    for src, dst in zip(g.players, reversed(m.players)):
        dst.x, dst.y = g.terrain.width - src.x, src.y
        dst.angle, dst.power, dst.weapon = 180 - src.angle, src.power, src.weapon
    m._wind.value = -g.wind
    return m


def test_observation_is_mirrored_for_player_two():
    g = Game(seed=4, countdown=0)
    g.update(2.0)
    a = AIController(0, None, Aimer.load()).observe(g)
    b = AIController(1, None, Aimer.load()).observe(mirror(g))
    assert a.shape == (OBS_SIZE,)
    np.testing.assert_allclose(a, b, atol=0.02)  # terrain columns can differ by a pixel


@pytest.mark.parametrize("difficulty", ["Easy", "Hard"])
def test_ai_plays_a_whole_match_by_the_rules(difficulty):
    g = Game(seed=7, countdown=0)
    ais = [make_ai(0, difficulty), make_ai(1, "Medium")]
    prev = [(p.angle, p.power, p.weapon) for p in g.players]
    while g.phase is Phase.PLAYING and g.time < 60:
        for ai in ais:
            ai.update(g, 1 / 60)
        g.update(1 / 60)
        for p, (angle, power, weapon) in zip(g.players, prev):
            assert abs(p.angle - angle) <= config.ANGLE_SPEED / 60 + 1e-9
            assert abs(p.power - power) <= config.POWER_SPEED / 60 + 1e-9
            assert (p.weapon - weapon) % len(WEAPONS) in (0, 1)
        prev = [(p.angle, p.power, p.weapon) for p in g.players]
    assert sum(ais[0].shots) > 0


def test_hud_lights_the_ai_keys():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from crosswind.control.human import P1_KEYS, P2_KEYS, HumanController
    from crosswind.render.hud import Hud
    pygame.init()
    screen = pygame.Surface((config.WIDTH, config.HEIGHT))
    g = Game(seed=4, countdown=0)
    ai = make_ai(1, "Hard", P2_KEYS)
    hud = Hud([HumanController(0, P1_KEYS), ai])
    for _ in range(10):
        ai.update(g, 1 / 60)
        g.update(1 / 60)
        hud.draw(screen, g, 1 / 60)
    assert ai.label == "AI (Hard)"


def test_training_env_follows_the_gym_api():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from crosswind.ai.train.env import CrosswindEnv
    check_env(CrosswindEnv(seed=1), skip_render_check=True)


def test_view_turns_around_only_once_the_enemy_is_clearly_past():
    from crosswind.ai.obs import FLIP_MARGIN, facing_flip
    g = Game(seed=4, countdown=0)
    me, them = g.players
    assert facing_flip(g, 0) is False and facing_flip(g, 1) is True  # at the start: enemy ahead
    them.x = me.x - FLIP_MARGIN / 2  # just past, still overlapping: keep facing the same way
    assert facing_flip(g, 0, False) is False
    them.x = me.x - FLIP_MARGIN * 2  # clearly behind us now: turn around
    assert facing_flip(g, 0, False) is True


@pytest.mark.parametrize("difficulty", ["Hard", "Rules"])
def test_ai_shoots_an_enemy_behind_it(difficulty):
    """Tanks that drove past each other: the AI must turn around, not keep firing the old way."""
    from crosswind.ai.rules import RuleBasedController
    from crosswind.ai.train.evaluate import cross
    hits = 0
    for seed in range(3):
        g = Game(seed=seed, countdown=0)
        cross(g)  # player 1 now on the right, player 2 on the left
        ai = RuleBasedController(0, seed=seed) if difficulty == "Rules" else make_ai(0, difficulty)
        while g.phase is Phase.PLAYING and g.time < 30:
            ai.update(g, 1 / 60)
            g.update(1 / 60)
        hits += g.players[1].hp < config.PLAYER_HP
    assert hits >= 2


def test_ai_view_shows_what_both_kinds_of_ai_decided():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from crosswind.ai.rules import RuleBasedController
    from crosswind.render.ai_view import AiView
    pygame.init()
    screen = pygame.Surface((config.WIDTH, config.HEIGHT))
    g = Game(seed=5, countdown=0)
    ais = [make_ai(0, "Hard"), RuleBasedController(1)]
    view = AiView()
    for _ in range(60 * 8):
        for ai in ais:
            ai.update(g, 1 / 60)
        g.update(1 / 60)
        view.draw(screen, g, ais)
    trained, rules = (ai.insight for ai in ais)
    assert trained["kind"] == "trained" and rules["kind"] == "rules"
    assert len(trained["probs"]) == len(ACTION_NVEC)
    assert all(abs(p.sum() - 1) < 1e-5 for p in trained["probs"])
    assert len(trained["suggestions"]) == len(ANGLE_BINS) and rules["reason"]

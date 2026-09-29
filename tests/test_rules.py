"""The rule-based computer player: needs no trained models, so these tests always run."""
import numpy as np

from crosswind import config
from crosswind.ai.rules import RuleBasedController
from crosswind.core.game import Game, Phase
from crosswind.core.weapons import WEAPONS


def test_rules_ai_plays_a_whole_match_by_the_rules():
    g = Game(seed=3, countdown=0)
    ais = [RuleBasedController(i, seed=i) for i in range(2)]
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
    assert sum(ais[0].shots) > 0 and sum(ais[1].shots) > 0
    assert ais[0].label == "AI (Rules)"


def test_rules_ai_brackets_its_shots_closer():
    """On a calm day, against a standing enemy, it brackets its way in from a first guess that is well off."""
    first, later = [], []
    for seed in range(6):
        g = Game(seed=seed, countdown=0, wind_strength=0)
        ai = RuleBasedController(0, seed=seed)  # player 2 just stands there
        ai._bracket(g).power += 20 if seed % 2 else -20  # a bad feel for the first shot: long, or short
        while len(ai.misses) < 8 and g.phase is Phase.PLAYING and g.time < 40:
            ai.update(g, 1 / 60)
            g.update(1 / 60)
        assert len(ai.misses) >= 5, seed
        first.append(ai.misses[0])
        later.append(np.mean(ai.misses[3:]))
    assert np.mean(later) < 0.6 * np.mean(first), (first, later)

import math
import sys

import numpy as np

from crosswind import config
from crosswind.core import physics
from crosswind.core.game import Game, Phase
from crosswind.core.terrain import Terrain
from crosswind.core.weapons import BOMBLET, WEAPONS, Weapon
from crosswind.core.wind import Wind


NO_DRAG = Weapon("test", mass=1.0, drag=0.0, blast_radius=10, damage=10)


def shoot(g, index, angle, power, weapon_index=0):
    """Set a player's aim and weapon directly (a test shortcut past the rate limits), then fire."""
    p = g.players[index]
    p.angle, p.power, p.weapon = angle, power, weapon_index
    return g.fire(index)


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
    assert not shoot(g, 0, 90, 40, 0)
    g.update(config.START_COUNTDOWN + 0.1)
    assert shoot(g, 0, 90, 40, 0)


def weapon(name):
    return next(i for i, w in enumerate(WEAPONS) if w.name == name)


def test_cooldown_blocks_instant_refire():
    g = Game(seed=3, countdown=0)
    assert shoot(g, 0, 90, 40, weapon("Spark"))
    assert not shoot(g, 0, 90, 40, weapon("Spark"))
    g.update(config.FIRE_COOLDOWN + 0.01)
    assert shoot(g, 0, 90, 40, weapon("Spark"))


def test_cheap_weapon_fires_in_bursts():
    g = Game(seed=3, countdown=0)
    shots = 0
    for _ in range(4):
        shots += shoot(g, 0, 90, 40, weapon("Spark"))
        g.update(config.FIRE_COOLDOWN + 0.01)
    assert shots == 4


def test_megaton_needs_a_full_bar():
    g = Game(seed=3, countdown=0)
    p = g.players[0]
    assert shoot(g, 0, 90, 40, weapon("Spark"))
    g.update(config.FIRE_COOLDOWN + 0.01)
    assert not shoot(g, 0, 90, 40, weapon("Megaton"))
    g.update(config.ENERGY_MAX / config.ENERGY_REGEN)
    assert p.energy == config.ENERGY_MAX
    assert shoot(g, 0, 90, 40, weapon("Megaton"))
    assert p.energy == 0


def test_every_weapon_costs_energy():
    costs = [w.energy for w in WEAPONS]
    assert len(WEAPONS) == 8 and min(costs) > 0 and max(costs) == config.ENERGY_MAX


def test_both_players_shoot_at_the_same_time():
    g = Game(seed=3, countdown=0)
    assert shoot(g, 0, 60, 70, 0)
    assert shoot(g, 1, 120, 70, 2)
    assert {p.owner for p in g.projectiles} == {0, 1}
    for _ in range(2000):
        g.update(1 / 60)
        if not g.projectiles:
            break
    assert not g.projectiles  # both landed or left the field


def flat_game(ground_y=400):
    g = Game(seed=3, countdown=0)
    g.terrain.mask[:] = False
    g.terrain.mask[ground_y:, :] = True
    for p in g.players:
        g._settle(p)
    return g


def test_tank_drives_on_flat_ground_and_uses_fuel():
    g = flat_game()
    p = g.players[0]
    x0 = p.x
    for _ in range(60):
        g.move(0, 1, 1 / 60)
    assert abs(p.x - x0 - config.PLAYER_SPEED) < 1.0
    assert abs(p.fuel - (config.PLAYER_FUEL - config.PLAYER_SPEED)) < 1.0


def test_steep_wall_blocks_driving():
    g = flat_game()
    p = g.players[0]
    wall_x = int(p.x) + 20
    g.terrain.mask[300:, wall_x:wall_x + 40] = True
    for _ in range(60):
        g.move(0, 1, 1 / 60)
    assert p.x < wall_x


def test_jump_goes_up_and_lands():
    g = flat_game()
    p = g.players[0]
    y0 = p.y
    assert g.jump(0)
    assert p.fuel == config.PLAYER_FUEL - config.JUMP_FUEL
    assert not g.jump(0)  # no double jump
    g.update(0.3)
    assert p.airborne and y0 - p.y > 30
    g.update(1.5)
    assert not p.airborne and p.y == y0


def test_jump_needs_fuel():
    g = flat_game()
    g.players[0].fuel = config.JUMP_FUEL - 1
    assert not g.jump(0)


def test_jump_clears_wall_that_blocks_driving():
    g = flat_game(ground_y=400)
    p = g.players[0]
    wall_x = int(p.x) + 20
    g.terrain.mask[370:, wall_x:wall_x + 300] = True  # 30 px step up, too steep to drive
    assert g.jump(0)
    for _ in range(120):
        g.move(0, 1, 1 / 60)
        g.update(1 / 60)
    assert p.x > wall_x and not p.airborne
    assert g._feet(p) == 370  # landed on top of the wall


def test_tank_falls_off_steep_drop():
    g = flat_game(ground_y=400)
    p = g.players[0]
    edge = int(p.x) + 5
    g.terrain.mask[:, edge:] = False
    g.terrain.mask[500:, edge:] = True  # 100 px cliff down
    for _ in range(20):
        g.move(0, 1, 1 / 60)
    assert p.airborne and g._feet(p) < 500  # falling, not teleported
    g.update(2.0)
    assert not p.airborne and g._feet(p) == 500


def fire_and_wait(g, angle, power, name, max_t=15.0):
    """Fire for player 0 and run until the first explosion (or build); returns it."""
    assert shoot(g, 0, angle, power, weapon(name))
    t = 0.0
    while not g.events and t < max_t:
        g.update(config.PHYSICS_DT)
        t += config.PHYSICS_DT
    assert g.events
    return g.events[0]


def test_rocket_drifts_less_than_plasma():
    def drift(w):  # relative to the distance flown, since the rocket flies much further
        calm = fly(physics.launch(0, 0, 60, 60, w, 0), 0.0, ground_y=0)
        windy = fly(physics.launch(0, 0, 60, 60, w, 0), 80.0, ground_y=0)
        return (windy.x - calm.x) / calm.x
    assert drift(WEAPONS[weapon("Rocket")]) < drift(WEAPONS[weapon("Plasma Orb")]) * 0.5


def test_cluster_splits_into_bomblets_at_apex():
    g = flat_game()
    assert shoot(g, 0, 60, 50, weapon("Cluster Bomb"))
    while len(g.projectiles) == 1:
        vy = g.projectiles[0].vy
        g.update(config.PHYSICS_DT)
    assert vy < 0  # still rising on the tick before the split
    assert len(g.projectiles) == 5 and all(p.weapon is BOMBLET for p in g.projectiles)
    assert len({round(p.vx) for p in g.projectiles}) == 5  # fanned out


def test_driller_bores_through_a_hill():
    g = flat_game()
    x0 = g.players[0].x
    hill_x = int(x0) + 300
    g.terrain.mask[250:400, hill_x:hill_x + 40] = True  # 40 px thick wall in the way
    ev = fire_and_wait(g, 30, 45, "Driller")
    assert ev.x > hill_x + 40  # came out the other side before exploding


def test_bouncer_bounces_before_exploding():
    g = flat_game()
    assert shoot(g, 0, 60, 45, weapon("Bouncer"))
    p = g.projectiles[0]
    while p.timer is None:
        g.update(config.PHYSICS_DT)
    landed_x = p.x
    assert not g.events  # touching the ground did not set it off
    g.update(WEAPONS[weapon("Bouncer")].fuse - 0.1)
    assert g.projectiles and not g.events
    g.update(0.2)
    assert len(g.events) == 1 and g.events[0].x > landed_x  # rolled on, then blew up


def test_dirt_bomb_builds_a_mound_without_damage():
    g = flat_game()
    before = g.terrain.mask.sum()
    ev = fire_and_wait(g, 60, 45, "Dirt Bomb")
    assert ev.dirt
    assert g.terrain.mask.sum() > before
    assert all(p.hp == config.PLAYER_HP for p in g.players)


def test_megaton_carves_far_more_than_spark():
    def carved(name, angle):
        g = flat_game()
        before = g.terrain.mask.sum()
        fire_and_wait(g, angle, 45, name)
        return before - g.terrain.mask.sum()
    assert carved("Megaton", 60) > 10 * carved("Spark", 60)


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
    assert not shoot(g, 1, 120, 60, 0)


def test_headless_random_play_without_pygame():
    rng = np.random.default_rng(0)
    g = Game(seed=0, countdown=0)
    games = 1
    wanted = [0, 0]  # each player saves up for a randomly picked weapon
    for _ in range(int(300 / (1 / 30))):  # five simulated minutes at 30 fps
        if g.phase is Phase.GAME_OVER:
            g = Game(seed=int(rng.integers(1000)), countdown=0)
            games += 1
        for i in (0, 1):
            g.move(i, int(rng.integers(-1, 2)), 1 / 30)
            if shoot(g, i, rng.uniform(10, 170), rng.uniform(20, 100), wanted[i]):
                wanted[i] = int(rng.integers(len(WEAPONS)))
        g.update(1 / 30)
    assert games > 1  # matches actually end


def test_aim_cannot_turn_faster_than_the_rate_limit():
    g = Game(seed=3, countdown=0)
    p = g.players[0]
    a0, p0 = p.angle, p.power
    g.adjust_aim(0, 90.0, -90.0, 0.1)
    assert abs(p.angle - (a0 + config.ANGLE_SPEED * 0.1)) < 1e-9
    assert abs(p.power - (p0 - config.POWER_SPEED * 0.1)) < 1e-9


def test_weapon_cycles_one_step_per_press():
    g = Game(seed=3, countdown=0)
    g.cycle_weapon(0, 5)
    assert g.players[0].weapon == 1
    g.cycle_weapon(0, -3)
    g.cycle_weapon(0, -1)
    assert g.players[0].weapon == len(WEAPONS) - 1


def test_fire_uses_the_current_aim_and_weapon():
    g = Game(seed=3, countdown=0)
    p = g.players[0]
    p.angle, p.power, p.weapon = 70.0, 55.0, weapon("Rocket")
    assert g.fire(0)
    shot = g.projectiles[0]
    assert shot.weapon is WEAPONS[weapon("Rocket")]
    assert abs(math.degrees(math.atan2(-shot.vy, shot.vx)) - 70.0) < 1e-6


def test_last_impact_records_where_a_shot_came_down():
    g = flat_game()
    assert g.last_impact == [None, None]
    ev = fire_and_wait(g, 60, 45, "Plasma Orb")
    hit = g.last_impact[0]
    assert hit is not None and g.last_impact[1] is None
    assert (hit.x, hit.y) == (ev.x, ev.y) and hit.weapon == weapon("Plasma Orb")


def test_core_does_not_import_pygame():
    import subprocess
    code = "import sys, crosswind.core.game; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

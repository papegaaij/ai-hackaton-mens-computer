# Crosswind

Live 2D artillery duel on a red dust planet. Two players share one keyboard and aim, move and fire at the same time.
Every shot costs energy, which refills over time: cheap weapons can be fired in bursts, heavy ones empty the bar.
Shots are affected by gravity, drag and wind. The wind
gets stronger higher up and shifts constantly, even while a shot is in the air. The terrain can be destroyed.

## Run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m crosswind
.venv/bin/python -m pytest -q   # tests
```

## Controls
| | Player 1 (left) | Player 2 (right) |
|---|---|---|
| Power (down / up) | 1 / 2 | [ / ] |
| Angle | A / S | Up / Down |
| Move left / right (uses fuel, refills slowly) | W / D | Left / Right |
| Jump (over steep sections; uses fuel) | Tab | Enter |
| Next weapon | ` | \\ |
| Fire | Left Shift | Right Shift |

Tap an aim key for a fine nudge; hold it and it speeds up. Esc: menu · R/Enter: rematch.
A 3-2-1 countdown starts each match; you can aim during it but not fire.

## Weapons
The white tick on your energy bar shows what the selected weapon costs (a full bar is 100).

| Weapon | Energy | Use it for |
|---|---|---|
| Spark | 12 | Cheap pokes in bursts. Small and light, so the wind pushes it around a lot. |
| Plasma Orb | 35 | The all-rounder. |
| Rocket | 45 | Strong wind: its engine burns for 0.6 s and it's heavy, so it flies fast and straight. |
| Bouncer | 40 | Rolling into hollows: bounces along the ground and explodes 1.5 s after it first lands. |
| Driller | 40 | Tanks behind a hill: bores on through rock for a moment before it explodes. |
| Cluster Bomb | 60 | A moving target: splits into 5 bomblets at the top of its arc. |
| Dirt Bomb | 30 | Defence: builds a mound of earth (cover, or a ramp to jump from). No damage. |
| Megaton | 100 | The whole bar: a huge crater and heavy damage. Heavy and slow. |

## Architecture
- `crosswind/core/`: the simulation itself, written in pure Python and numpy with no pygame. It is deterministic (seeded), steps in fixed physics ticks and can run headless.
- `crosswind/control/`: the `Controller` interface. `HumanController` reads one player's keys (`P1_KEYS` / `P2_KEYS`). Later, an RL agent can take over one player by returning a `FireAction`.
- `crosswind/render/`: the renderer and HUD. `theme.py` holds the full colour palette.
- `crosswind/scenes/`: menu, battle and game over screens.
- `Game.observe(i)` already returns a numeric observation vector, ready for a future Gymnasium env.

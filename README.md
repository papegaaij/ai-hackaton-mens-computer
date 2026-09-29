# Crosswind

Live 2D artillery duel on a red dust planet. Two players share one keyboard and aim, move and fire at the same time.
After each shot a player's cannon needs 2 seconds to recharge. Shots are affected by gravity, drag and wind. The wind
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
| Power | W / S | I / K |
| Angle | A / D | J / L |
| Move (uses fuel, refills slowly) | Q / E | U / O |
| Next weapon | R | P |
| Fire | Space | Enter |

Tap an aim key for a fine nudge; hold it and it speeds up. Esc: menu · R/Enter: rematch.
A 3-2-1 countdown starts each match; you can aim during it but not fire.

Weapons: **Plasma Orb** (standard), **Spore Pod** (light, drifts a lot in the wind), **Heavy Slug** (heavy, drifts very little).

## Architecture
- `crosswind/core/`: the simulation itself, written in pure Python and numpy with no pygame. It is deterministic (seeded), steps in fixed physics ticks and can run headless.
- `crosswind/control/`: the `Controller` interface. `HumanController` reads one player's keys (`P1_KEYS` / `P2_KEYS`). Later, an RL agent can take over one player by returning a `FireAction`.
- `crosswind/render/`: the renderer and HUD. `theme.py` holds the full colour palette.
- `crosswind/scenes/`: menu, battle and game over screens.
- `Game.observe(i)` already returns a numeric observation vector, ready for a future Gymnasium env.

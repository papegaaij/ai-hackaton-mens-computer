# Crosswind

Turn-based 2D artillery duel on a red dust planet. Two players take turns on one keyboard (hot-seat).
Shots are affected by gravity, wind (which gets stronger higher up) and drag, and the terrain can be destroyed.

## Run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m crosswind
.venv/bin/python -m pytest -q   # tests
```

## Controls (the player whose turn it is)
Left/Right: angle · Up/Down: power (hold Shift for fine control) · A/D: move (uses fuel) · Q/E: change weapon · Space: fire · Esc: menu · R: rematch

Weapons: **Plasma Orb** (standard), **Spore Pod** (light, drifts a lot in the wind), **Heavy Slug** (heavy, drifts very little).

## Architecture
- `crosswind/core/`: the simulation itself, written in pure Python and numpy with no pygame. It is deterministic (seeded), steps in fixed physics ticks and can run headless.
- `crosswind/control/`: the `Controller` interface. `HumanController` reads the keyboard. Later, an RL agent can take over one player by returning a `FireAction`.
- `crosswind/render/`: the renderer and HUD. `theme.py` holds the full colour palette.
- `crosswind/scenes/`: menu, battle and game over screens.
- `Game.observe(i)` already returns a numeric observation vector, ready for a future Gymnasium env.

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
| Angle (up / down) | W / S | Down / Up |
| Move left / right (uses fuel, refills slowly) | A / D | Left / Right |
| Jump (over steep sections; uses fuel) | Tab | Enter |
| Next weapon | ` | ? |
| Fire | Left Shift | Right Shift |

Tap an aim key for a fine nudge; hold it and it speeds up. Esc: menu · M: mute sound · - / =: music quieter / louder · F9: AI view · F10: full screen · R/Enter: rematch.
A 3-2-1 countdown starts each match; you can aim during it but not fire.

## Computer players
In the menu, **F1** and **F2** switch player 1 and player 2 between Human, AI Rules and AI Easy / Medium / Hard.
**Rules** is a rule-based opponent that needs no training or models (`crosswind/ai/rules.py`): it aims at a
fixed angle, guesses a first power from the distance and wind, then brackets its shots like a human, turning the
power up after a short shot and down after a long one, halving the step each time the miss flips side.
With two AIs you can sit back and watch: a new match starts by itself after each one.

The AI plays by the same rules as you:
- It has only your buttons. It turns the angle and power dials at the same speed (the same tap-fine,
  hold-fast ramp), steps through weapons one press at a time, and pays the same fuel and energy.
  Its keys light up in the key help while it "presses" them, so you can see what it is doing.
- It never works out where a shot will fly. It sees what you see: the terrain, both tanks, the HUD
  (including how the wind bar is moving), shots in the air, and where its own last shot came down.
- It is trained, in two parts:
  - The **aimer** learned from over a million practice shots which power lands a shot where it wants,
    much like getting a feel for it.
  - The **tactics** network learned with reinforcement learning (PPO), against bots and then against
    earlier versions of itself. It decides the angle, a correction on the aimer's power, the weapon,
    when to drive, jump and fire.
- Easy and Medium are earlier stages of the same training, with slower reactions and a shakier hand.
- Press **F9** in a match to see what each AI looks at and decides: which way it counts as "ahead", the
  ground it samples, where it aims, its last miss, the enemy shots it tracks, and a panel with the aimer's
  power per angle and each choice with how sure the network was (for Rules: its bracketing and why it
  picked its weapon).

### Retraining
The trained models are in `models/`, so playing needs no extra packages. To train them yourself:
```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python -m crosswind.ai.train.practice          # practice shots -> runs/practice.npz
.venv/bin/python -m crosswind.ai.train.train_aimer       # -> models/aimer.npz
.venv/bin/python -m crosswind.ai.train.train_tactics     # snapshots in runs/tactics/pool/
.venv/bin/python -m crosswind.ai.train.export runs/tactics/final.zip models/tactics_hard.npz
.venv/bin/python -m crosswind.ai.train.evaluate --difficulty Hard --vs random aimer Easy
```

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

## Sound
Style "Dust & Metal × Cinematic": metal and grit up close, sub-bass booms and rumble tails far away.
Every weapon has its own launch and impact sound, the wind bed follows the in-game wind live (louder,
brighter and leaning to the side it blows to), Player 1 sits left in the stereo image and Player 2 right.
Samples: Kenney (kenney.nl), CC0; see `crosswind/assets/sounds/CREDITS.md`.

Music: an eerie alien theme in the menu, a driving pulse in battle and a sting when the match is decided,
crossfaded and mixed just under the sound effects (it dips under big blasts); - and = change its volume. Tracks: *Dark Sci-Fi Audio Pack*
by SRG774, CC0; see `crosswind/assets/music/CREDITS.md`.

## Architecture
- `crosswind/core/`: the simulation itself, written in pure Python and numpy with no pygame. It is deterministic (seeded), steps in fixed physics ticks and can run headless.
- `crosswind/control/`: the `Controller` interface. `HumanController` reads one player's keys (`P1_KEYS` / `P2_KEYS`). Controllers act only through the game's inputs (`adjust_aim`, `move`, `jump`, `cycle_weapon`, `fire`) and report the keys they hold with `held()`.
- `crosswind/ai/`: the computer player (numpy only while playing); `crosswind/ai/train/` trains it.
- `crosswind/render/`: the renderer and HUD. `theme.py` holds the full colour palette.
- `crosswind/audio/`: the sound engine. The core emits `SoundEvent`s (pure data), `recipes.py` says what each one sounds like, `engine.py` plays them with the pygame mixer, `music.py` the music.
- `crosswind/scenes/`: menu, battle and game over screens.

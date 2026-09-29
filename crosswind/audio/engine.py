"""Sound engine: turns the simulation's SoundEvents into layered, panned, prioritised sound (pygame mixer).

- One-shot sounds are layered recipes (see recipes.py); pitch variants and a muffled 'far' variant of each
  sample are rendered once with numpy and cached.
- Loops: wind bed (generated noise that follows the game's wind live), sand grit, planet drone, one drive
  loop per tank.
- Mix: priority decides who gets a channel when all are busy, at most 2 copies of one sample play at once,
  big events duck the rest by ~6 dB, Player 1 sits left and Player 2 right.
If no audio device is available the engine silently does nothing.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pygame

from crosswind import config
from crosswind.audio import recipes as R
from crosswind.audio.recipes import LEVEL, Layer

SOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "sounds"

FAR_DISTANCE = 450      # px from the nearest tank beyond which impacts lose their highs
FAR_CUTOFF = 1400.0     # Hz of the muffled variant
MAX_COPIES = 2          # of one sample playing at the same time
DUCK_DEPTH = 0.5        # ~ -6 dB
DUCK_TIME = 1.0         # s

# priority 1 = never dropped (steals a channel), 5 = first to be dropped
PRIORITY = {"count": 1, "go": 1, "hit": 1, "down": 1, "win": 1, "draw": 1,
            "launch": 2, "explode": 2, "dirt": 2,
            "split": 3, "bounce": 3, "drill": 3, "ready": 3, "no_energy": 3, "jump": 3, "land": 3,
            "switch": 4, "gust": 5, "ui": 5}

_N_LOOPS = 6  # reserved channels: wind low, wind bright, grit, drone, drive P1, drive P2


class SoundEngine:
    def __init__(self) -> None:
        self.enabled = False
        self.muted = False
        self.rng = random.Random()
        self._samples: dict[str, pygame.mixer.Sound] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._variants: dict[tuple, pygame.mixer.Sound] = {}
        self._pending: list[tuple[float, pygame.mixer.Sound, float, float, int, str]] = []
        self._playing: dict[str, list[pygame.mixer.Channel]] = {}
        self._time = 0.0
        self._duck = 0.0
        self._game_id: int | None = None
        self._prev_x: list[float] = []
        self._drive_vol = [0.0, 0.0]
        self._wind_vol = 0.0
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 2, 512)
            self.rate, _, self.channels = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(40)
            pygame.mixer.set_reserved(_N_LOOPS)
            self._load()
            self._loops = [pygame.mixer.Channel(i) for i in range(_N_LOOPS)]
            self._start_loops()
            self.enabled = True
        except (pygame.error, OSError, ValueError) as e:  # no audio device, missing files, ...
            print(f"[crosswind] sound disabled: {e}")

    # ---- loading and rendering variants ---------------------------------
    def _load(self) -> None:
        for path in sorted(SOUND_DIR.glob("*.ogg")):
            snd = pygame.mixer.Sound(str(path))
            self._samples[path.stem] = snd
            arr = pygame.sndarray.array(snd).astype(np.float32)
            self._arrays[path.stem] = arr if arr.ndim == 2 else arr[:, None]

    def _make(self, arr: np.ndarray) -> pygame.mixer.Sound:
        out = np.clip(arr, -32768, 32767).astype(np.int16)
        if self.channels == 1:
            out = out[:, :1]
        elif out.shape[1] == 1:
            out = np.repeat(out, self.channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(out))

    def _lowpass(self, arr: np.ndarray, cutoff: float) -> np.ndarray:
        n = arr.shape[0]
        spec = np.fft.rfft(arr, axis=0)
        freqs = np.fft.rfftfreq(n, 1 / self.rate)
        spec *= (1 / (1 + (freqs / cutoff) ** 4))[:, None]  # smooth 4th-order-ish roll-off
        return np.fft.irfft(spec, n=n, axis=0)

    def _variant(self, name: str, rate: float, far: bool) -> pygame.mixer.Sound:
        rate = round(rate, 2)
        key = (name, rate, far)
        snd = self._variants.get(key)
        if snd is None:
            arr = self._arrays[name]
            if rate != 1.0:
                n = arr.shape[0]
                idx = np.linspace(0, n - 1, max(1, int(n / rate)))
                arr = np.stack([np.interp(idx, np.arange(n), arr[:, c]) for c in range(arr.shape[1])], axis=1)
            if far:
                arr = self._lowpass(arr, FAR_CUTOFF)
            snd = self._make(arr)
            self._variants[key] = snd
        return snd

    # ---- generated loops ---------------------------------------------------
    def _noise_loop(self, seconds: float, low: float, high: float, seed: int, clicks: float = 0.0) -> pygame.mixer.Sound:
        """Seamless stereo noise band (low..high Hz), built in the frequency domain so it loops perfectly."""
        rng = np.random.default_rng(seed)
        n = int(self.rate * seconds)
        if clicks:  # sparse sand ticks instead of continuous noise
            src = (rng.random((n, 2)) < clicks / self.rate) * rng.normal(0, 1, (n, 2))
        else:
            src = rng.normal(0, 1, (n, 2))
        spec = np.fft.rfft(src, axis=0)
        f = np.fft.rfftfreq(n, 1 / self.rate)
        band = (1 / (1 + (f / high) ** 4)) * (1 - 1 / (1 + (f / max(low, 1)) ** 4))
        arr = np.fft.irfft(spec * band[:, None], n=n, axis=0)
        t = np.arange(n) / n
        arr *= (0.75 + 0.25 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6)))[:, None]  # slow built-in swell
        arr *= 20000 / (np.abs(arr).max() + 1e-9)
        return self._make(arr)

    def _start_loops(self) -> None:
        wind_low = self._noise_loop(6.0, 40, 450, seed=1)
        wind_high = self._noise_loop(6.0, 500, 3000, seed=2)
        grit = self._noise_loop(4.0, 2500, 7000, seed=3, clicks=90)
        drone = self._samples[R.DRONE]
        drive = self._variant(R.DRIVE, 0.7, True)
        for ch, snd in zip(self._loops, (wind_low, wind_high, grit, drone, drive, drive)):
            ch.play(snd, loops=-1)
            ch.set_volume(0.0)

    # ---- public API ----------------------------------------------------------
    def toggle_mute(self) -> None:
        if not self.enabled:
            return
        self.muted = not self.muted
        if self.muted:
            pygame.mixer.pause()
            self._pending.clear()
        else:
            pygame.mixer.unpause()

    def ui(self) -> None:
        if self.enabled and not self.muted:
            self._play(R.UI, 0.0, LEVEL["cue"], PRIORITY["ui"])

    def update(self, game, dt: float, events: bool = True) -> None:
        """Call once per frame after game.update(). events=False: ambience only (menu background)."""
        if not self.enabled:
            return
        self._time += dt
        self._duck = max(0.0, self._duck - dt / DUCK_TIME)
        if id(game) != self._game_id:
            self._game_id = id(game)
            self._prev_x = [p.x for p in game.players]
            self._pending.clear()
        if events and not self.muted:
            for ev in game.sounds:
                self._handle(game, ev)
        game.sounds.clear()
        self._flush()
        self._update_loops(game, dt, drive=events)

    # ---- events -> recipes -----------------------------------------------------
    def _pan_x(self, x: float) -> float:
        return max(-1.0, min(1.0, x / config.WIDTH * 2 - 1))

    def _is_far(self, game, x: float, y: float) -> bool:
        tanks = [p for p in game.players if p.alive] or game.players
        return min(math.dist((p.x, p.y), (x, y)) for p in tanks) > FAR_DISTANCE

    def _handle(self, game, ev) -> None:
        name, prio = ev.name, PRIORITY.get(ev.name, 3)
        side = -0.5 if ev.player == 0 else 0.5 if ev.player == 1 else 0.0
        pan, far = self._pan_x(ev.x), False
        if name == "launch":
            layers = R.LAUNCH.get(ev.weapon, R.LAUNCH["Plasma Orb"])
            big = ev.weapon == "Megaton"
            level, pan = (LEVEL["big"] * 0.63 if big else LEVEL["launch"]), (0.0 if big else side)
            if big:
                self._duck = 1.0
        elif name == "explode":
            r = ev.size
            if r >= 55:
                layers, level, pan = R.EXPLODE_BIG, LEVEL["big"], 0.0
                self._duck = 1.0
            else:
                layers = R.EXPLODE_SMALL if r < 20 else R.EXPLODE_MEDIUM
                level = R.db(-10 + 6 * min(1.0, max(0.0, (r - 14) / 26)))  # -10 dB small .. -4 dB large
            far = self._is_far(game, ev.x, ev.y)
        elif name == "dirt":
            layers, level, far = R.DIRT, R.db(-8), self._is_far(game, ev.x, ev.y)
        elif name == "split":
            layers, level = R.SPLIT, LEVEL["detail"]
        elif name == "bounce":
            layers, level = R.bounce(int(ev.size)), LEVEL["detail"]
        elif name == "drill":
            layers, level = R.DRILL, LEVEL["detail"]
        elif name == "hit":
            layers, level = R.HIT, LEVEL["hit"]
        elif name == "down":
            layers, level = R.DOWN, LEVEL["big"]
            self._duck = 1.0
        elif name in ("win", "draw"):
            layers, level, pan = (R.WIN if name == "win" else R.DRAW), LEVEL["big"], 0.0
            self._duck = 1.0
        elif name == "count":
            layers, level, pan = R.count(int(ev.size)), LEVEL["cue"] * 1.6, 0.0
        elif name == "go":
            layers, level, pan = R.GO, LEVEL["hit"], 0.0
        elif name == "ready":
            layers, level, pan = R.READY, LEVEL["cue"], side * 1.6
        elif name == "no_energy":
            layers, level, pan = R.NO_ENERGY, LEVEL["cue"], side * 1.6
        elif name == "switch":
            layers, level, pan = R.switch(int(ev.size)), LEVEL["cue"], side * 1.6
        elif name == "jump":
            layers, level = R.JUMP, LEVEL["move"]
        elif name == "land":
            layers, level = R.LAND, LEVEL["move"]
        elif name == "gust":
            s = abs(ev.size) / config.WIND_MAX
            layers, level, pan = R.GUST, LEVEL["gust"] * (0.4 + s), (0.6 if ev.size > 0 else -0.6) * s
        else:
            return
        self._play(layers, pan, level, prio, far)

    def _play(self, layers: list[Layer], pan: float, level: float, prio: int, far: bool = False) -> None:
        duck = 1.0 if prio == 1 else 1.0 - DUCK_DEPTH * self._duck
        for layer in layers:
            rate = layer.rate * (1 + self.rng.uniform(-layer.jitter, layer.jitter))
            snd = self._variant(layer.sample, rate, far and layer.far)
            vol = min(1.0, layer.gain * level * duck)
            left, right = vol * min(1.0, 1 - pan), vol * min(1.0, 1 + pan)
            self._pending.append((self._time + layer.delay, snd, left, right, prio, layer.sample))
        self._flush()

    def _flush(self) -> None:
        due = [p for p in self._pending if p[0] <= self._time]
        if not due:
            return
        self._pending = [p for p in self._pending if p[0] > self._time]
        for _, snd, left, right, prio, key in sorted(due, key=lambda p: p[4]):
            playing = [c for c in self._playing.get(key, []) if c.get_busy()]
            if len(playing) >= MAX_COPIES:
                playing.pop(0).fadeout(80)
            ch = pygame.mixer.find_channel(force=prio <= 2)
            if ch is None:
                continue  # all channels busy with more important sounds: drop this one
            ch.play(snd)
            ch.set_volume(left, right)
            playing.append(ch)
            self._playing[key] = playing

    # ---- loops -----------------------------------------------------------------
    def _update_loops(self, game, dt: float, drive: bool) -> None:
        s = min(1.0, abs(game.wind) / config.WIND_MAX)
        target = 0.08 + 0.12 * s  # -22 dB calm .. -14 dB storm
        self._wind_vol += (target - self._wind_vol) * min(1.0, dt * 3)
        duck = 1.0 - 0.3 * self._duck  # ~ -3 dB under explosions
        lean = (0.6 if game.wind > 0 else -0.6) * s  # the bed leans to where the wind blows
        wind_low, wind_high, grit, drone, *drives = self._loops

        def set_pan(ch, vol, pan):
            ch.set_volume(vol * min(1.0, 1 - pan), vol * min(1.0, 1 + pan))

        set_pan(wind_low, self._wind_vol * duck, lean * 0.5)
        set_pan(wind_high, 0.22 * s ** 1.5 * duck, lean)
        set_pan(grit, 0.12 * s ** 2 * duck, lean)
        drone.set_volume(0.05 * duck)

        for i, p in enumerate(game.players[:2]):
            moved = abs(p.x - self._prev_x[i]) > 0.05 if i < len(self._prev_x) else False
            want = LEVEL["move"] * 1.5 if (drive and moved and not p.airborne and p.alive) else 0.0
            self._drive_vol[i] += (want - self._drive_vol[i]) * min(1.0, dt * 12)
            set_pan(drives[i], self._drive_vol[i], self._pan_x(p.x))
        self._prev_x = [p.x for p in game.players]


_engine: SoundEngine | None = None


def engine() -> SoundEngine:
    """The one shared sound engine (created on first use, after pygame.init())."""
    global _engine
    if _engine is None:
        _engine = SoundEngine()
    return _engine

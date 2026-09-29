"""Music: a menu theme, a battle loop and game-over stings, crossfaded on two reserved mixer channels.

pygame.mixer.music can't crossfade, so tracks are Sounds on two channels: a loop restarts on the other
channel a few seconds before it ends (the tracks' author recommends a crossfade), and switching tracks
fades one out while the next fades in. The music sits under the game sounds and dips under big blasts.
Tracks: Dark Sci-Fi Audio Pack by SRG774, CC0 (see assets/music/CREDITS.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

MUSIC_DIR = Path(__file__).resolve().parent.parent / "assets" / "music"

MENU, BATTLE, WIN, DRAW = "sector", "urgent", "victory", "transmission"
LOOPING = {MENU, BATTLE}
VOLUME = {MENU: 0.45, BATTLE: 0.32, WIN: 0.5, DRAW: 0.4}  # under the sound effects
SWITCH_FADE = 1.5  # s to crossfade from one track to another
LOOP_FADE = 3.0    # s of overlap when a loop starts over
DUCK_DEPTH = 0.4   # how far the music dips under big blasts


@dataclass
class _Voice:
    channel: pygame.mixer.Channel
    track: str | None = None
    level: float = 0.0   # 0..1 fade position
    target: float = 0.0
    fade: float = 1.0    # s for a full fade
    elapsed: float = 0.0  # s since this pass of the track started


class Music:
    def __init__(self, channels: tuple[pygame.mixer.Channel, pygame.mixer.Channel]):
        self._voices = [_Voice(ch) for ch in channels]
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self.track: str | None = None
        self._active = 0  # voice playing the current track

    def _sound(self, name: str) -> pygame.mixer.Sound:
        if name not in self._sounds:
            self._sounds[name] = pygame.mixer.Sound(str(MUSIC_DIR / f"{name}.ogg"))
        return self._sounds[name]

    def play(self, name: str | None) -> None:
        """Switch to track `name` (None: fade to silence). Asking for the track that is playing does nothing."""
        if name == self.track:
            return
        self.track = name
        self._fade_out(self._voices[self._active], SWITCH_FADE)
        if name is not None:
            self._start(1 - self._active, SWITCH_FADE)

    def _fade_out(self, voice: _Voice, fade: float) -> None:
        voice.target, voice.fade = 0.0, fade

    def _start(self, index: int, fade: float) -> None:
        """Start the current track from the top on voice `index`, fading in over `fade` s."""
        v = self._voices[index]
        v.channel.set_volume(0.0)
        v.channel.play(self._sound(self.track))
        v.track, v.level, v.target, v.fade, v.elapsed = self.track, 0.0, 1.0, fade, 0.0
        self._active = index

    def update(self, dt: float, duck: float, muted: bool) -> None:
        """Per frame: loop crossfades, fades and volume. `duck` 0..1 is how loud the game just got."""
        if muted:  # the whole mixer is paused: keep our clock paused too
            return
        cur = self._voices[self._active]
        if (self.track in LOOPING and cur.track == self.track
                and cur.elapsed >= self._sound(self.track).get_length() - LOOP_FADE):
            self._fade_out(cur, LOOP_FADE)
            self._start(1 - self._active, LOOP_FADE)
        for v in self._voices:
            if v.track is None:
                continue
            v.elapsed += dt
            step = dt / v.fade
            v.level = min(v.target, v.level + step) if v.level < v.target else max(v.target, v.level - step)
            if v.level <= 0.0 and v.target == 0.0:
                v.channel.stop()
                v.track = None
                continue
            v.channel.set_volume(VOLUME[v.track] * v.level * (1.0 - DUCK_DEPTH * duck))

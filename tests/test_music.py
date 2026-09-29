import os

import pytest

from crosswind.audio import music as M


@pytest.fixture(scope="module")
def channels():
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    try:
        pygame.mixer.init(44100, -16, 2, 512)
    except pygame.error:
        pytest.skip("no audio driver available")
    pygame.mixer.set_num_channels(8)
    yield pygame.mixer.Channel(6), pygame.mixer.Channel(7)
    pygame.mixer.quit()


def run(m, seconds, duck=0.0, muted=False):
    for _ in range(int(seconds * 60)):
        m.update(1 / 60, duck, muted)


def test_every_track_exists_and_is_credited():
    credits = (M.MUSIC_DIR / "CREDITS.md").read_text()
    for track in (M.MENU, M.BATTLE, M.WIN, M.DRAW):
        assert (M.MUSIC_DIR / f"{track}.ogg").exists()
        assert f"{track}.ogg" in credits


def test_switching_tracks_crossfades(channels):
    m = M.Music(channels)
    m.play(M.MENU)
    run(m, M.SWITCH_FADE + 0.1)
    menu = m._voices[m._active]
    assert menu.track == M.MENU and menu.level == 1.0
    m.play(M.BATTLE)
    run(m, M.SWITCH_FADE / 2)
    old, new = m._voices[1 - m._active], m._voices[m._active]
    assert old.track == M.MENU and new.track == M.BATTLE and 0 < old.level < 1 and 0 < new.level < 1
    run(m, M.SWITCH_FADE)
    assert old.track is None and not old.channel.get_busy()  # faded out and stopped
    assert new.level == 1.0
    assert abs(new.channel.get_volume() - M.VOLUME[M.BATTLE]) < 0.01


def test_a_loop_starts_over_with_a_crossfade(channels):
    m = M.Music(channels)
    m.play(M.BATTLE)
    first = m._active
    m._voices[first].elapsed = m._sound(M.BATTLE).get_length() - M.LOOP_FADE - 0.05
    run(m, 0.1)
    assert m._active != first and m._voices[m._active].track == M.BATTLE
    assert m._voices[first].target == 0.0  # the old pass fades out underneath


def test_stings_do_not_loop(channels):
    m = M.Music(channels)
    m.play(M.WIN)
    m._voices[m._active].elapsed = m._sound(M.WIN).get_length()
    active = m._active
    run(m, 0.1)
    assert m._active == active


def test_music_ducks_and_pauses_with_mute(channels):
    m = M.Music(channels)
    m.play(M.MENU)
    run(m, M.SWITCH_FADE + 0.1)
    v = m._voices[m._active]
    run(m, 0.02, duck=1.0)
    assert v.channel.get_volume() < M.VOLUME[M.MENU] * 0.7
    elapsed = v.elapsed
    run(m, 1.0, muted=True)
    assert v.elapsed == elapsed  # the clock stops while the mixer is paused

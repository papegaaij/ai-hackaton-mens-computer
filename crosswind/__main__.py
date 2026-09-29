"""Entry point: python -m crosswind"""
import pygame

from crosswind import config
from crosswind.audio.engine import engine
from crosswind.scenes.scenes import MenuScene

FULLSCREEN_KEY = pygame.K_F10
MUSIC_KEYS = {pygame.K_MINUS: -1, pygame.K_KP_MINUS: -1, pygame.K_EQUALS: 1, pygame.K_KP_PLUS: 1}  # music down / up
WINDOW_FLAGS = pygame.SCALED | pygame.RESIZABLE  # always drawn at WIDTH x HEIGHT, scaled up by pygame


def toggle_fullscreen() -> None:
    """Switch between a window and the whole screen (letterboxed, same picture, just bigger)."""
    try:
        pygame.display.toggle_fullscreen()
    except pygame.error:  # not every video driver can toggle in place: open the display again instead
        full = pygame.display.get_surface().get_flags() & pygame.FULLSCREEN
        flags = WINDOW_FLAGS if full else pygame.SCALED | pygame.FULLSCREEN
        try:
            pygame.display.set_mode((config.WIDTH, config.HEIGHT), flags)
        except pygame.error:
            pass  # no full screen on this display: stay as we are


def main() -> None:
    pygame.mixer.pre_init(44100, -16, 2, 512)  # small buffer: low latency for shots
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), WINDOW_FLAGS)
    pygame.display.set_caption("Crosswind")
    clock = pygame.time.Clock()
    scene = MenuScene(screen)
    running = True
    while running:
        dt = min(clock.tick(config.FPS) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == FULLSCREEN_KEY:
                toggle_fullscreen()
            elif event.type == pygame.KEYDOWN and event.key in MUSIC_KEYS:
                engine().change_music_volume(MUSIC_KEYS[event.key])
            else:
                scene = scene.handle_event(event)
        scene = scene.update(dt)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()

"""Entry point: python -m crosswind"""
import pygame

from crosswind import config
from crosswind.scenes.scenes import MenuScene


def main() -> None:
    pygame.mixer.pre_init(44100, -16, 2, 512)  # small buffer: low latency for shots
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Crosswind")
    clock = pygame.time.Clock()
    scene = MenuScene(screen)
    running = True
    while running:
        dt = min(clock.tick(config.FPS) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                scene = scene.handle_event(event)
        scene = scene.update(dt)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()

"""Keyboard controller. Hot-seat: both humans share the same keys, only the active one is polled."""
from __future__ import annotations

import pygame

from crosswind import config
from crosswind.core.actions import FireAction
from crosswind.core.game import Game


class HumanController:
    def __init__(self) -> None:
        self._fire_requested = False
        self._weapon_step = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_SPACE:
            self._fire_requested = True
        elif event.key == pygame.K_q:
            self._weapon_step -= 1
        elif event.key == pygame.K_e:
            self._weapon_step += 1

    def update(self, game: Game, dt: float) -> FireAction | None:
        keys = pygame.key.get_pressed()
        fine = 0.25 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
        d_angle = (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) * config.ANGLE_SPEED * dt * fine
        d_power = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * config.POWER_SPEED * dt * fine
        game.adjust_aim(d_angle, d_power)
        game.move(keys[pygame.K_d] - keys[pygame.K_a], dt)
        if self._weapon_step:
            game.cycle_weapon(self._weapon_step)
            self._weapon_step = 0
        if self._fire_requested:
            self._fire_requested = False
            p = game.active
            return FireAction(p.angle, p.power, p.weapon)
        return None

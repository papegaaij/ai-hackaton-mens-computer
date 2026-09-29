"""Keyboard controllers. Both players share one keyboard, each with their own cluster of keys."""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from crosswind import config
from crosswind.core.actions import FireAction
from crosswind.core.game import Game

_KEY_LABELS = {pygame.K_SPACE: "Space", pygame.K_RETURN: "Enter", pygame.K_TAB: "Tab",
               pygame.K_LSHIFT: "LShift", pygame.K_RSHIFT: "RShift", pygame.K_UP: "Up", pygame.K_DOWN: "Down",
               pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right"}


@dataclass(frozen=True)
class KeyMap:
    power_up: int
    power_down: int
    angle_left: int
    angle_right: int
    move_left: int
    move_right: int
    jump: int
    weapon: int
    fire: int

    def describe(self) -> list[str]:
        """One line per control, for the on-screen help."""
        def n(key: int) -> str:
            return _KEY_LABELS.get(key) or pygame.key.name(key).upper()
        return [
            f"Move: {n(self.move_left)} / {n(self.move_right)}",
            f"Angle: {n(self.angle_left)} / {n(self.angle_right)}",
            f"Power: {n(self.power_down)} / {n(self.power_up)}",
            f"Jump: {n(self.jump)}",
            f"Weapon: {n(self.weapon)}",
            f"Fire: {n(self.fire)}",
        ]


P1_KEYS = KeyMap(power_up=pygame.K_2, power_down=pygame.K_1, angle_left=pygame.K_a, angle_right=pygame.K_s,
                 move_left=pygame.K_w, move_right=pygame.K_d, jump=pygame.K_TAB, weapon=pygame.K_BACKQUOTE,
                 fire=pygame.K_LSHIFT)
P2_KEYS = KeyMap(power_up=pygame.K_RIGHTBRACKET, power_down=pygame.K_LEFTBRACKET, angle_left=pygame.K_UP,
                 angle_right=pygame.K_DOWN, move_left=pygame.K_LEFT, move_right=pygame.K_RIGHT,
                 jump=pygame.K_RETURN, weapon=pygame.K_BACKSLASH, fire=pygame.K_RSHIFT)


class HumanController:
    def __init__(self, index: int, keys: KeyMap) -> None:
        self.index = index
        self.keys = keys
        self._fire_requested = False
        self._jump_requested = False
        self._weapon_step = 0
        # per aim axis: (direction held, seconds held) for tap = fine, hold = fast
        self._held = [(0, 0.0), (0, 0.0)]

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == self.keys.fire:
            self._fire_requested = True
        elif event.key == self.keys.jump:
            self._jump_requested = True
        elif event.key == self.keys.weapon:
            self._weapon_step += 1

    def _ramp(self, axis: int, direction: int, dt: float) -> float:
        """Aim speed as a fraction of full speed: starts slow on a tap, ramps up while the key is held."""
        held_dir, held_for = self._held[axis]
        held_for = held_for + dt if direction == held_dir else 0.0
        self._held[axis] = (direction, held_for)
        return direction * min(1.0, config.AIM_FINE + (1 - config.AIM_FINE) * held_for / config.AIM_RAMP)

    def update(self, game: Game, dt: float) -> FireAction | None:
        pressed = pygame.key.get_pressed()
        k = self.keys
        d_angle = self._ramp(0, pressed[k.angle_left] - pressed[k.angle_right], dt) * config.ANGLE_SPEED * dt
        d_power = self._ramp(1, pressed[k.power_up] - pressed[k.power_down], dt) * config.POWER_SPEED * dt
        game.adjust_aim(self.index, d_angle, d_power)
        game.move(self.index, pressed[k.move_right] - pressed[k.move_left], dt)
        if self._jump_requested:
            self._jump_requested = False
            game.jump(self.index)
        if self._weapon_step:
            game.cycle_weapon(self.index, self._weapon_step)
            self._weapon_step = 0
        if self._fire_requested:
            self._fire_requested = False
            p = game.players[self.index]
            return FireAction(p.angle, p.power, p.weapon)
        return None

"""HUD: per-player HP, fuel, energy and aim; wind; start countdown; key help."""
from __future__ import annotations

import math

import pygame

from crosswind import config
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS
from crosswind.render import theme


class Hud:
    FIRE_BANNER = 0.8  # s the "FIRE!" banner stays up after the countdown

    def __init__(self, controllers) -> None:
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 20)
        self.big = pygame.font.Font(None, 96)
        self.controllers = controllers  # per player: shows its keys, lit while it presses them
        self.help_lines = [c.keys.describe() for c in controllers]  # per player: (label, [(control, name)])
        self._fire_banner = 0.0

    def _text(self, screen, text, pos, font=None, color=theme.HUD_TEXT, center=False, right=False):
        surf = (font or self.font).render(text, True, color)
        if center:
            rect = surf.get_rect(center=pos)
        elif right:
            rect = surf.get_rect(topright=pos)
        else:
            rect = surf.get_rect(topleft=pos)
        screen.blit(surf, rect)
        return rect

    def _key_chip(self, screen, name: str, x: float, y: float, down: bool, color) -> int:
        """Draw a key cap; lit up in the player's colour while held. Returns its width."""
        text = self.small.render(name, True, theme.HUD_PANEL[:3] if down else theme.HUD_TEXT)
        rect = pygame.Rect(x, y, max(22, text.get_width() + 10), 18)
        if down:
            pygame.draw.rect(screen, color, rect, border_radius=4)
        else:
            pygame.draw.rect(screen, theme.HP_BACK, rect, border_radius=4)
            pygame.draw.rect(screen, theme.HUD_DIM, rect, width=1, border_radius=4)
        screen.blit(text, text.get_rect(center=rect.center))
        return rect.width

    def draw(self, screen: pygame.Surface, game: Game, dt: float) -> None:
        w, h = screen.get_size()
        panel = pygame.Surface((w, 84), pygame.SRCALPHA)
        panel.fill(theme.HUD_PANEL)
        screen.blit(panel, (0, 0))

        for pl in game.players:
            col = theme.PLAYER_COLORS[pl.index] if pl.alive else theme.HUD_DIM
            left = pl.index == 0
            x = 20 if left else w - 280
            side_x = x + 266 if left else x - 6  # numbers next to the bars, on the outside of the panel

            self._text(screen, f"PLAYER {pl.index + 1}", (x, 8), color=col)
            pygame.draw.rect(screen, theme.HP_BACK, (x, 30, 260, 12), border_radius=6)
            pygame.draw.rect(screen, col, (x, 30, 260 * pl.hp / config.PLAYER_HP, 12), border_radius=6)
            self._text(screen, f"{pl.hp:.0f}", (side_x, 29), self.small, right=not left)
            pygame.draw.rect(screen, theme.HP_BACK, (x, 46, 260, 4))
            pygame.draw.rect(screen, theme.HUD_DIM, (x, 46, 260 * pl.fuel / config.PLAYER_FUEL, 4))

            # energy bar, with a tick at what the selected weapon costs
            weapon = WEAPONS[pl.weapon]
            armed = game.can_fire(pl.index)
            pygame.draw.rect(screen, theme.HP_BACK, (x, 54, 260, 6), border_radius=3)
            pygame.draw.rect(screen, col if armed else theme.HUD_DIM,
                             (x, 54, 260 * pl.energy / config.ENERGY_MAX, 6), border_radius=3)
            cost_x = x + 260 * weapon.energy / config.ENERGY_MAX
            pygame.draw.line(screen, theme.HUD_TEXT, (cost_x, 52), (cost_x, 61), 2)
            if not pl.alive:
                status = "DOWN"
            elif armed:
                status = "READY"
            elif pl.energy < weapon.energy:
                status = f"{pl.energy:.0f}/{weapon.energy:.0f}"
            else:
                status = ""  # countdown, cooldown or game over
            self._text(screen, status, (side_x, 51), self.small, col if armed else theme.HUD_DIM, right=not left)

            aim = (f"Angle {pl.angle:5.1f}   Power {pl.power:5.1f}   "
                   f"[{pl.weapon + 1}/{len(WEAPONS)} {weapon.name} - {weapon.energy:.0f}]")
            self._text(screen, aim, (x + 260 if not left else x, 65), self.small, right=not left)

        # wind indicator (centre)
        cx = w // 2
        strength = game.wind / config.WIND_MAX
        self._text(screen, f"WIND {abs(strength) * 10:.1f} {'>>' if strength > 0 else '<<' if strength < 0 else ''}",
                   (cx, 14), center=True)
        pygame.draw.rect(screen, theme.HP_BACK, (cx - 100, 30, 200, 8), border_radius=4)
        bw = int(100 * abs(strength))
        bx = cx if strength > 0 else cx - bw
        pygame.draw.rect(screen, theme.SPORE, (bx, 30, bw, 8), border_radius=4)
        pygame.draw.line(screen, theme.HUD_TEXT, (cx, 26), (cx, 42), 2)

        # key help, each player's keys on their own side
        line_h = 20
        for i, (ctrl, lines) in enumerate(zip(self.controllers, self.help_lines)):
            held = ctrl.held()
            right = i == 1
            edge = w - 12 if right else 12
            top = h - 10 - line_h * (len(lines) + 1)
            title = f"PLAYER {i + 1}" + (f" · {ctrl.label}" if ctrl.label else "")
            self._text(screen, title, (edge, top), self.small, theme.PLAYER_COLORS[i], right=right)
            for j, (label, keys) in enumerate(lines):
                y = top + (j + 1) * line_h
                label_x = edge - 150 if right else edge  # label column, then key chips
                self._text(screen, label, (label_x, y + 2), self.small, theme.HUD_DIM)
                x = edge - 150 + 62 if right else edge + 62
                for control, name in keys:
                    x += self._key_chip(screen, name, x, y, control in held, theme.PLAYER_COLORS[i]) + 4
        self._text(screen, "Esc menu", (cx, h - 14), self.small, theme.HUD_DIM, center=True)

        # start countdown: 3, 2, 1, FIRE!
        if game.countdown > 0:
            self._fire_banner = self.FIRE_BANNER
            self._text(screen, str(math.ceil(game.countdown)), (cx, h // 3), self.big, theme.HUD_TEXT, center=True)
        elif self._fire_banner > 0:
            self._fire_banner -= dt
            self._text(screen, "FIRE!", (cx, h // 3), self.big, theme.ROCK_EDGE, center=True)

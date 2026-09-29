"""HUD: per-player HP, fuel, recharge and aim; wind; start countdown; key help."""
from __future__ import annotations

import math

import pygame

from crosswind import config
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS
from crosswind.render import theme


class Hud:
    FIRE_BANNER = 0.8  # s the "FIRE!" banner stays up after the countdown

    def __init__(self, help_lines: list[list[str]]) -> None:
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 20)
        self.big = pygame.font.Font(None, 96)
        self.help_lines = help_lines  # per player: one line per control
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

            # recharge: fills up until the next shot is allowed
            armed = game.can_fire(pl.index)
            charge = 1.0 - pl.reload / config.RELOAD_TIME
            pygame.draw.rect(screen, theme.HP_BACK, (x, 54, 260, 6), border_radius=3)
            pygame.draw.rect(screen, col if armed else theme.HUD_DIM, (x, 54, 260 * charge, 6), border_radius=3)
            if not pl.alive:
                status = "DOWN"
            elif pl.reload > 0:
                status = f"{pl.reload:.1f}s"
            else:
                status = "READY" if armed else ""  # blank during the countdown and after the game
            self._text(screen, status, (side_x, 51), self.small, col if armed else theme.HUD_DIM, right=not left)

            aim = f"Angle {pl.angle:5.1f}   Power {pl.power:5.1f}   [{WEAPONS[pl.weapon].name}]"
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
        line_h = 18
        for i, lines in enumerate(self.help_lines):
            top = h - 12 - line_h * (len(lines) + 1)
            for j, line in enumerate([f"PLAYER {i + 1}"] + lines):
                color = theme.PLAYER_COLORS[i] if j == 0 else theme.HUD_DIM
                pos = (12, top + j * line_h) if i == 0 else (w - 12, top + j * line_h)
                self._text(screen, line, pos, self.small, color, right=(i == 1))
        self._text(screen, "Esc menu", (cx, h - 14), self.small, theme.HUD_DIM, center=True)

        # start countdown: 3, 2, 1, FIRE!
        if game.countdown > 0:
            self._fire_banner = self.FIRE_BANNER
            self._text(screen, str(math.ceil(game.countdown)), (cx, h // 3), self.big, theme.HUD_TEXT, center=True)
        elif self._fire_banner > 0:
            self._fire_banner -= dt
            self._text(screen, "FIRE!", (cx, h // 3), self.big, theme.ROCK_EDGE, center=True)

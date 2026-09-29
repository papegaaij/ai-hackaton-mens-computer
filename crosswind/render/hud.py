"""HUD: HP bars, wind, aim and weapon, turn banner."""
from __future__ import annotations

import pygame

from crosswind import config
from crosswind.core.game import Game, Phase
from crosswind.core.weapons import WEAPONS
from crosswind.render import theme


class Hud:
    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 20)
        self.big = pygame.font.Font(None, 64)
        self.banner_timer = 0.0
        self._last_turn = -1

    def _text(self, screen, text, pos, font=None, color=theme.HUD_TEXT, center=False):
        surf = (font or self.font).render(text, True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        screen.blit(surf, rect)
        return rect

    def draw(self, screen: pygame.Surface, game: Game, dt: float) -> None:
        w = screen.get_width()
        panel = pygame.Surface((w, 64), pygame.SRCALPHA)
        panel.fill(theme.HUD_PANEL)
        screen.blit(panel, (0, 0))

        for pl in game.players:
            col = theme.PLAYER_COLORS[pl.index]
            left = pl.index == 0
            x = 20 if left else w - 280
            label = f"PLAYER {pl.index + 1}" + ("  <" if game.current == pl.index and not left else "")
            if left and game.current == 0:
                label = "> " + label
            self._text(screen, label, (x, 8), color=col if game.current == pl.index else theme.HUD_DIM)
            pygame.draw.rect(screen, theme.HP_BACK, (x, 32, 260, 12), border_radius=6)
            pygame.draw.rect(screen, col, (x, 32, 260 * pl.hp / config.PLAYER_HP, 12), border_radius=6)
            self._text(screen, f"{pl.hp:.0f}", (x + 264 if left else x - 34, 30), self.small)
            pygame.draw.rect(screen, theme.HP_BACK, (x, 48, 260, 4))
            pygame.draw.rect(screen, theme.HUD_DIM, (x, 48, 260 * pl.fuel / config.PLAYER_FUEL, 4))

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

        p = game.active
        self._text(screen, f"Angle {p.angle:5.1f}   Power {p.power:5.1f}   [{WEAPONS[p.weapon].name}]   Turn {game.turn}",
                   (cx, 52), self.small, center=True)

        help_text = "Left/Right angle  Up/Down power (Shift fine)  A/D move  Q/E weapon  Space fire  Esc menu"
        self._text(screen, help_text, (w // 2, screen.get_height() - 14), self.small, theme.HUD_DIM, center=True)

        # hot-seat banner
        if game.turn != self._last_turn:
            self._last_turn = game.turn
            self.banner_timer = 1.6
        if self.banner_timer > 0 and game.phase is Phase.AIMING:
            self.banner_timer -= dt
            col = theme.PLAYER_COLORS[game.current]
            self._text(screen, f"PLAYER {game.current + 1} - GET READY", (w // 2, screen.get_height() // 3),
                       self.big, col, center=True)

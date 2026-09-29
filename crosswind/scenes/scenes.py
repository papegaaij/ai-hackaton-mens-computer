"""Menu, battle and game-over scenes. Each scene returns the next scene (or itself) from update()."""
from __future__ import annotations

import math

import pygame

from crosswind.control.human import P1_KEYS, P2_KEYS, HumanController
from crosswind.core.game import Game, Phase
from crosswind.render import theme
from crosswind.render.hud import Hud
from crosswind.render.renderer import Renderer


class MenuScene:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.preview = Game(seed=None)
        self.renderer = Renderer(*screen.get_size())
        self.title = pygame.font.Font(None, 120)
        self.font = pygame.font.Font(None, 34)
        self.t = 0.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            return BattleScene(self.screen)
        return self

    def update(self, dt):
        self.t += dt
        self.preview.update(dt)  # keeps the wind drifting behind the title
        self.renderer.draw(self.screen, self.preview, dt)
        w, h = self.screen.get_size()
        for text, font, y, col in (
            ("CROSSWIND", self.title, h * 0.3, theme.ROCK_EDGE),
            ("Artillery duel on a red dust planet", self.font, h * 0.3 + 70, theme.HUD_TEXT),
            ("2 players, one keyboard. Fire at will.", self.font, h * 0.3 + 110, theme.HUD_DIM),
        ):
            s = font.render(text, True, col)
            self.screen.blit(s, s.get_rect(center=(w / 2, y)))
        if math.sin(self.t * 4) > -0.3:
            s = self.font.render("Press ENTER to start", True, theme.HUD_TEXT)
            self.screen.blit(s, s.get_rect(center=(w / 2, h * 0.3 + 180)))
        return self


class BattleScene:
    REMATCH_GRACE = 1.0  # s on the game-over screen before rematch keys count, so mashing fire can't skip it

    def __init__(self, screen: pygame.Surface, seed: int | None = None):
        self.screen = screen
        self.game = Game(seed=seed)
        self.renderer = Renderer(*screen.get_size())
        # Two human controllers sharing the keyboard, each on its own keys. Swap one for an AI later.
        self.controllers = [HumanController(0, P1_KEYS), HumanController(1, P2_KEYS)]
        self.hud = Hud([c.keys.describe() for c in self.controllers])
        self.big = pygame.font.Font(None, 90)
        self.font = pygame.font.Font(None, 34)
        self._over_time = 0.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return MenuScene(self.screen)
            if (self.game.phase is Phase.GAME_OVER and self._over_time >= self.REMATCH_GRACE
                    and event.key in (pygame.K_r, pygame.K_RETURN)):
                return BattleScene(self.screen)
        if self.game.phase is Phase.PLAYING:
            for ctrl in self.controllers:
                if hasattr(ctrl, "handle_event"):
                    ctrl.handle_event(event)
        return self

    def update(self, dt):
        g = self.game
        if g.phase is Phase.PLAYING:
            for i, ctrl in enumerate(self.controllers):
                action = ctrl.update(g, dt)
                if action is not None:
                    g.fire(i, action)
        else:
            self._over_time += dt
        g.update(dt)
        self.renderer.draw(self.screen, g, dt)
        self.hud.draw(self.screen, g, dt)
        if g.phase is Phase.GAME_OVER:
            self._draw_game_over()
        return self

    def _draw_game_over(self):
        w, h = self.screen.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 140))
        self.screen.blit(shade, (0, 0))
        win = self.game.winner
        text = "DRAW" if win is None else f"PLAYER {win + 1} WINS"
        col = theme.HUD_TEXT if win is None else theme.PLAYER_COLORS[win]
        s = self.big.render(text, True, col)
        self.screen.blit(s, s.get_rect(center=(w / 2, h * 0.4)))
        s = self.font.render("R: rematch    Esc: menu", True, theme.HUD_TEXT)
        self.screen.blit(s, s.get_rect(center=(w / 2, h * 0.4 + 70)))

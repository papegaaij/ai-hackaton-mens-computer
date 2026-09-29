"""Menu, battle and game-over scenes. Each scene returns the next scene (or itself) from update()."""
from __future__ import annotations

import math

import pygame

from crosswind.ai.controller import DIFFICULTIES, make_ai, models_available
from crosswind.ai.rules import RuleBasedController
from crosswind.audio import music
from crosswind.audio.engine import engine
from crosswind.control.human import P1_KEYS, P2_KEYS, HumanController
from crosswind.core.game import Game, Phase
from crosswind.render import theme
from crosswind.render.hud import Hud
from crosswind.render.renderer import Renderer


HUMAN = "Human"
RULES = "Rules"  # the rule-based AI: needs no trained models
KEYS = (P1_KEYS, P2_KEYS)
MODE_KEYS = (pygame.K_F1, pygame.K_F2)  # cycle who plays player 1 / player 2


def make_controller(index: int, mode: str):
    if mode == HUMAN:
        return HumanController(index, KEYS[index])
    if mode == RULES:
        return RuleBasedController(index, keys=KEYS[index], label="AI (Rules)")
    return make_ai(index, mode, KEYS[index])


class MenuScene:
    def __init__(self, screen: pygame.Surface, modes: tuple[str, str] = (HUMAN, HUMAN)):
        self.screen = screen
        self.modes = list(modes)
        self.choices = [HUMAN, RULES, *(DIFFICULTIES if models_available() else ())]
        self.preview = Game(seed=None)
        self.renderer = Renderer(*screen.get_size())
        self.title = pygame.font.Font(None, 120)
        self.font = pygame.font.Font(None, 34)
        self.t = 0.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            engine().toggle_mute()
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            engine().ui()
            return BattleScene(self.screen, modes=tuple(self.modes))
        if event.type == pygame.KEYDOWN and event.key in MODE_KEYS:
            i = MODE_KEYS.index(event.key)
            self.modes[i] = self.choices[(self.choices.index(self.modes[i]) + 1) % len(self.choices)]
        return self

    def update(self, dt):
        self.t += dt
        self.preview.update(dt)  # keeps the wind drifting behind the title
        self.renderer.draw(self.screen, self.preview, dt)
        engine().update(self.preview, dt, events=False)  # wind ambience only
        engine().play_music(music.MENU)
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
        for i, mode in enumerate(self.modes):
            who = mode if mode == HUMAN else f"AI ({mode})"
            text = f"Player {i + 1}: {who}" + (f"   [F{i + 1}]" if len(self.choices) > 1 else "")
            s = self.font.render(text, True, theme.PLAYER_COLORS[i])
            self.screen.blit(s, s.get_rect(center=(w / 2, h * 0.3 + 240 + i * 36)))
        return self


class BattleScene:
    REMATCH_GRACE = 1.0  # s on the game-over screen before rematch keys count, so mashing fire can't skip it
    AUTO_REMATCH = 4.0   # s on the game-over screen before an AI-vs-AI match starts over by itself

    def __init__(self, screen: pygame.Surface, seed: int | None = None, modes: tuple[str, str] = (HUMAN, HUMAN)):
        self.screen = screen
        self.modes = modes
        self.game = Game(seed=seed)
        self.renderer = Renderer(*screen.get_size())
        # Each player is a human on their own keys, or an AI that shows the same keys as it "presses" them.
        self.controllers = [make_controller(i, mode) for i, mode in enumerate(modes)]
        self.hud = Hud(self.controllers)
        self.big = pygame.font.Font(None, 90)
        self.font = pygame.font.Font(None, 34)
        self._over_time = 0.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return MenuScene(self.screen, self.modes)
            if event.key == pygame.K_m:
                engine().toggle_mute()
            if (self.game.phase is Phase.GAME_OVER and self._over_time >= self.REMATCH_GRACE
                    and event.key in (pygame.K_r, pygame.K_RETURN)):
                return BattleScene(self.screen, modes=self.modes)
        if self.game.phase is Phase.PLAYING:
            for ctrl in self.controllers:
                if hasattr(ctrl, "handle_event"):
                    ctrl.handle_event(event)
        return self

    def update(self, dt):
        g = self.game
        if g.phase is Phase.PLAYING:
            for ctrl in self.controllers:
                ctrl.update(g, dt)
        else:
            self._over_time += dt
            if HUMAN not in self.modes and self._over_time >= self.AUTO_REMATCH:
                return BattleScene(self.screen, modes=self.modes)
        g.update(dt)
        self.renderer.draw(self.screen, g, dt)
        self.hud.draw(self.screen, g, dt)
        engine().update(g, dt)
        if g.phase is Phase.PLAYING:
            engine().play_music(music.BATTLE)
        else:
            engine().play_music(music.WIN if g.winner is not None else music.DRAW)
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

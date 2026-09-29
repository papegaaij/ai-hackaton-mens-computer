"""Draws a Game state. Reads the simulation, never changes it."""
from __future__ import annotations

import math
import random

import numpy as np
import pygame

from crosswind import config
from crosswind.core.game import Game
from crosswind.core.wind import wind_at
from crosswind.render import theme


def _vertical_gradient(w: int, h: int, top, bottom) -> np.ndarray:
    t = np.linspace(0, 1, h)[:, None]
    col = np.array(top) * (1 - t) + np.array(bottom) * t  # (h, 3)
    return np.repeat(col[None, :, :], w, axis=0).astype(np.uint8)  # (w, h, 3) for surfarray


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color, self.size = color, size


class Renderer:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.rng = random.Random(7)
        self.sky = self._make_sky()
        self._rock_rgb = _vertical_gradient(width, height, theme.ROCK_TOP, theme.ROCK_BOTTOM)
        self._terrain_surf: pygame.Surface | None = None
        self._terrain_version = -1
        self.spores = [self._new_spore(anywhere=True) for _ in range(70)]
        self.dust = [Particle(self.rng.uniform(0, width), self.rng.uniform(height * 0.1, height * 0.85),
                              0, 0, 1, theme.DUST, 1) for _ in range(28)]
        self._glow = {s: self._make_glow(theme.SPORE, s * 5) for s in (1, 2, 3)}
        self._shot_glow = self._make_glow(theme.TRAIL, 26)
        self._edge_glow = pygame.Surface((width, height))
        self.particles: list[Particle] = []
        self.shake = 0.0
        self.time = 0.0

    @staticmethod
    def _make_glow(color, radius: int) -> pygame.Surface:
        surf = pygame.Surface((radius * 2, radius * 2))
        for i in range(radius, 0, -1):
            f = (1 - i / radius) ** 2 * 0.5
            pygame.draw.circle(surf, tuple(int(c * f) for c in color), (radius, radius), i)
        return surf

    # ---- static layers ---------------------------------------------------
    def _make_sky(self) -> pygame.Surface:
        surf = pygame.surfarray.make_surface(_vertical_gradient(self.w, self.h, theme.SKY_TOP, theme.SKY_BOTTOM))
        for _ in range(160):
            x, y = self.rng.randrange(self.w), self.rng.randrange(int(self.h * 0.7))
            b = self.rng.randint(90, 255)
            c = tuple(int(v * b / 255) for v in theme.STAR)
            surf.set_at((x, y), c)
        for (mx, my, r, col) in ((self.w * 0.78, 110, 46, theme.MOON_A), (self.w * 0.64, 70, 18, theme.MOON_B)):
            glow = self._make_glow(col, int(r * 2.6))
            surf.blit(glow, (mx - glow.get_width() / 2, my - glow.get_height() / 2), special_flags=pygame.BLEND_ADD)
            pygame.draw.circle(surf, col, (mx, my), r)
            pygame.draw.circle(surf, tuple(int(v * 0.8) for v in col), (mx + r * 0.3, my - r * 0.2), r * 0.25)
        return surf

    def _terrain_surface(self, game: Game) -> pygame.Surface:
        t = game.terrain
        if self._terrain_version != t.version or self._terrain_surf is None:
            mask = t.mask.T  # (w, h)
            rgb = self._rock_rgb.copy()
            above = np.zeros_like(mask)
            above[:, 1:] = mask[:, :-1]
            edge = mask & ~above
            # thicken the glowing crust a bit
            edge2 = edge | np.roll(edge, 1, axis=1) | np.roll(edge, 2, axis=1)
            rgb[edge2 & mask] = theme.ROCK_EDGE
            # soft additive glow rising above the crust
            glow = np.zeros(mask.shape, dtype=np.float32)
            for k in range(1, 9):
                glow = np.maximum(glow, np.roll(edge, -k, axis=1) * (1 - k / 9) ** 1.5)
            glow[mask] = 0
            glow_rgb = (glow[:, :, None] * np.array(theme.ROCK_EDGE) * 0.55).astype(np.uint8)
            self._edge_glow = pygame.surfarray.make_surface(glow_rgb)
            rgb[~mask] = (0, 0, 0)
            surf = pygame.surfarray.make_surface(rgb)
            surf.set_colorkey((0, 0, 0))
            self._terrain_surf = surf.convert() if pygame.display.get_surface() else surf
            self._terrain_surf.set_colorkey((0, 0, 0))
            self._terrain_version = t.version
        return self._terrain_surf

    # ---- particles -------------------------------------------------------
    def _new_spore(self, anywhere=False) -> Particle:
        x = self.rng.uniform(0, self.w) if anywhere else self.rng.choice((-10, self.w + 10))
        return Particle(x, self.rng.uniform(0, self.h * 0.8), 0, 0, 1, theme.SPORE, self.rng.choice((1, 2, 2, 3)))

    def _update_particles(self, game: Game, dt: float) -> None:
        for s in self.spores:
            target = wind_at(game.wind, s.y, self.h) * 1.6
            s.vx += (target - s.vx) * min(1.0, dt * 1.5)
            s.x += s.vx * dt
            s.y += math.sin(self.time * 1.3 + s.x * 0.02) * 8 * dt
            if s.x < -20 or s.x > self.w + 20:
                s.x = -10 if game.wind > 0 else self.w + 10
                s.y = self.rng.uniform(0, self.h * 0.8)
        for p in self.particles:
            p.life -= dt
            p.vy += config.GRAVITY * 0.5 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
        self.particles = [p for p in self.particles if p.life > 0]

    def _spawn_explosion(self, x: float, y: float, r: float, dirt: bool = False) -> None:
        self.shake = min(12.0, r * (0.1 if dirt else 0.3))
        for _ in range(int(r * 2)):
            a = self.rng.uniform(0, 2 * math.pi)
            sp = self.rng.uniform(40, r * (3 if dirt else 8))
            col = self.rng.choice(theme.DIRT if dirt else theme.EXPLOSION)
            self.particles.append(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp - 60,
                                           self.rng.uniform(0.4, 1.1), col, self.rng.choice((2, 3, 4))))

    # ---- main draw -------------------------------------------------------
    def draw(self, screen: pygame.Surface, game: Game, dt: float) -> None:
        self.time += dt
        for ev in game.events:
            self._spawn_explosion(ev.x, ev.y, ev.radius, ev.dirt)
        game.events.clear()
        self._update_particles(game, dt)

        ox = oy = 0
        if self.shake > 0.1:
            ox, oy = self.rng.uniform(-self.shake, self.shake), self.rng.uniform(-self.shake, self.shake)
            self.shake *= 0.85

        screen.blit(self.sky, (0, 0))
        for d in self.dust:
            d.x += wind_at(game.wind, d.y, self.h) * 3 * dt
            if d.x < -80 or d.x > self.w + 80:
                d.x = -60 if game.wind > 0 else self.w + 60
            length = min(70, abs(game.wind) * 0.8) + 4
            dirx = 1 if game.wind >= 0 else -1
            pygame.draw.line(screen, theme.DUST, (d.x, d.y), (d.x - dirx * length, d.y), 1)
        for s in self.spores:
            a = 0.85 + 0.15 * math.sin(self.time * 3 + s.x)
            g = self._glow[s.size]
            screen.blit(g, (s.x - g.get_width() / 2, s.y - g.get_height() / 2), special_flags=pygame.BLEND_ADD)
            pygame.draw.circle(screen, tuple(int(c * a) for c in s.color), (s.x, s.y), s.size)
        screen.blit(self._terrain_surface(game), (ox, oy))
        screen.blit(self._edge_glow, (ox, oy), special_flags=pygame.BLEND_ADD)

        for pl in game.players:
            self._draw_tank(screen, game, pl, ox, oy)

        for pr in game.projectiles:
            pts = pr.trail[-60:]
            for i, (tx, ty) in enumerate(pts):
                f = (i + 1) / len(pts)
                pygame.draw.circle(screen, tuple(int(c * f) for c in theme.TRAIL), (tx + ox, ty + oy), 1 + f * 2)
            w = pr.weapon
            if pr.age < w.burn:  # rocket exhaust
                speed = math.hypot(pr.vx, pr.vy) or 1.0
                for _ in range(2):
                    self.particles.append(Particle(
                        pr.x - pr.vx / speed * 6, pr.y - pr.vy / speed * 6,
                        -pr.vx * 0.2 + self.rng.uniform(-30, 30), -pr.vy * 0.2 + self.rng.uniform(-30, 30),
                        self.rng.uniform(0.15, 0.35), self.rng.choice(theme.EXPLOSION[1:]), 2))
            g = self._shot_glow
            screen.blit(g, (pr.x + ox - g.get_width() / 2, pr.y + oy - g.get_height() / 2), special_flags=pygame.BLEND_ADD)
            core = theme.PLAYER_COLORS[pr.owner]
            if w.fuse and pr.timer is not None and math.sin(self.time * 25) > 0:
                core = theme.FUSE
            pygame.draw.circle(screen, (255, 255, 255), (pr.x + ox, pr.y + oy), w.size + 2)
            pygame.draw.circle(screen, core, (pr.x + ox, pr.y + oy), w.size)
            if pr.y < 0:  # off-screen marker
                pygame.draw.polygon(screen, theme.TRAIL, [(pr.x, 4), (pr.x - 6, 14), (pr.x + 6, 14)])

        for p in self.particles:
            f = p.life / p.max_life
            pygame.draw.circle(screen, tuple(int(c * f) for c in p.color), (p.x + ox, p.y + oy), p.size * f + 1)

    def _draw_tank(self, screen, game: Game, pl, ox, oy) -> None:
        col = theme.PLAYER_COLORS[pl.index]
        if not pl.alive:
            col = (70, 60, 90)
        r = config.PLAYER_RADIUS
        x, y = pl.x + ox, pl.y + oy
        if game.can_fire(pl.index):
            pulse = 0.5 + 0.5 * math.sin(self.time * 5)
            glow = pygame.Surface((r * 5, r * 5), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*col, int(40 + 40 * pulse)), (r * 2.5, r * 2.5), r * 2)
            screen.blit(glow, (x - r * 2.5, y - r * 2.5))
        # barrel
        tx, ty = game.barrel_tip(pl)
        pygame.draw.line(screen, col, (x, y - r * 0.5), (tx + ox, ty + oy), round(r * 0.28))
        # dome body + legs (alien crawler)
        for lx in (-r * 0.8, -r * 0.3, r * 0.3, r * 0.8):
            pygame.draw.line(screen, tuple(int(c * 0.6) for c in col), (x + lx, y), (x + lx * 1.3, y + r * 0.6),
                             round(r * 0.14))
        pygame.draw.ellipse(screen, tuple(int(c * 0.5) for c in col), (x - r, y - r * 0.4, r * 2, r * 0.9))
        pygame.draw.circle(screen, col, (x, y - r * 0.4), r * 0.65, draw_top_left=True, draw_top_right=True)
        eye = (x + (r * 0.3 if pl.angle < 90 else -r * 0.3), y - r * 0.6)
        pygame.draw.circle(screen, (255, 255, 255), eye, r * 0.21)
        pygame.draw.circle(screen, (20, 10, 40), eye, r * 0.11)

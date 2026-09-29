"""The AI view (F9): what each computer player looks at, and what it decided and why.

On the battlefield: which way it counts as "ahead", the ground points it samples, where it aims, its last
impact and the miss, and the enemy shots it is tracking. In a panel on its side: the aimer's power for
each angle, and the choices it made with how sure it was (or, for the Rules AI, its bracketing).
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from crosswind import config
from crosswind.ai.obs import AIM_TARGETS, ANGLE_BINS, INCOMING, POWER_STEPS, TERRAIN_SAMPLES
from crosswind.core.game import Game
from crosswind.core.weapons import WEAPONS
from crosswind.render import theme

PANEL_W = 320
PANEL_TOP = 92
MOVE_NAMES = ("back", "stay", "toward enemy")  # the move choice, as the AI sees it (enemy ahead)
TARGET_NAMES = {"enemy": "the enemy", "halfway": "halfway", "near": "just in front"}


def _arrow(surf, color, start, end, width=2, head=7):
    pygame.draw.line(surf, color, start, end, width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    for side in (-0.5, 0.5):
        tip = (end[0] - head * math.cos(ang + side), end[1] - head * math.sin(ang + side))
        pygame.draw.line(surf, color, end, tip, width)


def _dashed(surf, color, start, end, dash=8):
    length = math.dist(start, end)
    for i in range(0, int(length), dash * 2):
        a, b = i / length, min(1.0, (i + dash) / length)
        pygame.draw.line(surf, color, (start[0] + (end[0] - start[0]) * a, start[1] + (end[1] - start[1]) * a),
                         (start[0] + (end[0] - start[0]) * b, start[1] + (end[1] - start[1]) * b), 2)


class AiView:
    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 16)

    def draw(self, screen: pygame.Surface, game: Game, controllers) -> None:
        layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for ctrl in controllers:
            insight = getattr(ctrl, "insight", None)
            if insight is None:
                continue
            color = theme.PLAYER_COLORS[ctrl.index]
            self._battlefield(layer, game, ctrl.index, insight, color)
            self._panel(layer, game, ctrl, insight, color)
        screen.blit(layer, (0, 0))

    # ---- on the battlefield ------------------------------------------------
    def _battlefield(self, surf, game: Game, index: int, ins: dict, color) -> None:
        me, them = game.players[index], game.players[1 - index]
        faint = (*color, 110)
        # which way is "ahead" (toward the enemy) in its mirrored view
        ahead = -1 if ins["flip"] else 1
        y = me.y - config.PLAYER_RADIUS * 1.9
        _arrow(surf, color, (me.x - 16 * ahead, y), (me.x + 20 * ahead, y))
        self._label(surf, "ahead", (me.x + 24 * ahead, y - 7), color, right=ahead < 0)
        if ins["kind"] == "trained":
            for x in np.linspace(0, game.terrain.width - 1, TERRAIN_SAMPLES):  # the ground heights it reads
                pos = (float(x), game.terrain.surface_y(x))
                pygame.draw.circle(surf, (0, 0, 0, 150), pos, 5)
                pygame.draw.circle(surf, color, pos, 3)
            tx, ty = ins["target"]  # where the aimer was asked to aim
            pygame.draw.circle(surf, color, (tx, ty), 12, 2)
            pygame.draw.line(surf, color, (tx - 18, ty), (tx + 18, ty), 1)
            pygame.draw.line(surf, color, (tx, ty - 18), (tx, ty + 18), 1)
        # its last impact, and how far off the enemy it was
        hit = game.last_impact[index]
        if hit is not None and game.time - hit.time < 8.0:
            pygame.draw.line(surf, color, (hit.x - 6, hit.y - 6), (hit.x + 6, hit.y + 6), 3)
            pygame.draw.line(surf, color, (hit.x - 6, hit.y + 6), (hit.x + 6, hit.y - 6), 3)
            _dashed(surf, faint, (hit.x, hit.y), (them.x, them.y))
            miss = math.dist((hit.x, hit.y), (them.x, them.y))
            self._label(surf, f"miss {miss:.0f} px", ((hit.x + them.x) / 2, (hit.y + them.y) / 2 - 12), color)
        # the enemy shots it keeps an eye on, with where they are heading
        shots = sorted((p for p in game.projectiles if p.owner != index),
                       key=lambda p: (p.x - me.x) ** 2 + (p.y - me.y) ** 2)[:INCOMING]
        for p in shots:
            pygame.draw.circle(surf, color, (p.x, p.y), 10, 2)
            _arrow(surf, faint, (p.x, p.y), (p.x + p.vx * 0.2, p.y + p.vy * 0.2), 1, 5)

    # ---- the panel ---------------------------------------------------------
    def _panel(self, surf, game: Game, ctrl, ins: dict, color) -> None:
        w = surf.get_width()
        x = 12 if ctrl.index == 0 else w - 12 - PANEL_W
        lines = self._trained_lines(ins) if ins["kind"] == "trained" else self._rules_lines(ins)
        chart = 78 if ins["kind"] == "trained" and ins["suggestions"] is not None else 0
        height = 30 + chart + 18 * len(lines)
        pygame.draw.rect(surf, theme.HUD_PANEL, (x, PANEL_TOP, PANEL_W, height), border_radius=6)
        pygame.draw.rect(surf, (*color, 140), (x, PANEL_TOP, PANEL_W, height), 1, border_radius=6)
        self._label(surf, f"PLAYER {ctrl.index + 1} · {ctrl.label}: what it sees", (x + 10, PANEL_TOP + 8), color)
        y = PANEL_TOP + 28
        if chart:
            self._chart(surf, ins, x + 10, y, color)
            y += chart
        for text, strong in lines:
            small = self.font.size(text)[0] > PANEL_W - 20  # too long: squeeze it in
            self._label(surf, text, (x + 10, y + 2 * small), theme.HUD_TEXT if strong else theme.HUD_DIM, small=small)
            y += 18

    def _chart(self, surf, ins: dict, x: float, y: float, color) -> None:
        """The aimer's power for each angle it can pick (for the selected weapon, aiming at the enemy)."""
        powers = ins["suggestions"]
        chosen = ins["action"][0]
        probs = ins["probs"][0] if ins["probs"] else None
        bar_w, top, h = 16, y + 12, 44
        self._label(surf, "aimer: power to hit, per angle", (x, y - 2), theme.HUD_DIM, small=True)
        for i, power in enumerate(powers):
            bx = x + i * bar_w
            reach = 5 <= power <= 100
            bh = h * min(max(float(power), 0.0), 110.0) / 110.0
            col = color if i == chosen else (theme.HUD_TEXT if reach else theme.HP_BACK)
            pygame.draw.rect(surf, (*col, 230 if reach or i == chosen else 160),
                             (bx + 2, top + h - bh, bar_w - 4, bh))
            if probs is not None:  # how much the network liked this angle
                pygame.draw.rect(surf, (*color, 200), (bx + 2, top + h + 3, (bar_w - 4) * float(probs[i]), 3))
        right = x + len(powers) * bar_w
        self._label(surf, "ahead", (x, top + h + 8), theme.HUD_DIM, small=True)
        self._label(surf, "up", (x + 8.5 * bar_w, top + h + 8), theme.HUD_DIM, small=True, center=True)
        self._label(surf, "behind", (right, top + h + 8), theme.HUD_DIM, small=True, right=True)

    def _trained_lines(self, ins: dict) -> list[tuple[str, bool]]:
        a_bin, p_step, weapon, target, move, jump, fire = ins["action"]
        probs = ins["probs"]

        def sure(k: int, choice: int) -> str:
            return f"  ({probs[k][choice]:.0%} sure)" if probs else ""

        power = min(max(ins["aimer"] + ins["correction"] + ins["noise"], 5.0), 100.0)
        noise = f", {ins['noise']:+.0f} shaky hand" if round(ins["noise"]) else ""
        lines = [
            (f"Angle {ANGLE_BINS[a_bin]:.0f}° from ahead{sure(0, a_bin)}", True),
            (f"Power {power:.0f} = aimer {ins['aimer']:.0f}, {POWER_STEPS[p_step]:+.0f} own{noise}", True),
            (f"Weapon {WEAPONS[weapon].name}{sure(2, weapon)}", True),
            (f"Aim at {TARGET_NAMES[AIM_TARGETS[target]]}{sure(3, target)}", False),
            (f"Drive: {MOVE_NAMES[move]}{sure(4, move)}", False),
        ]
        if probs:
            lines.append((f"Jump {probs[5][1]:.0%}   Fire {probs[6][1]:.0%}   (chance it wanted to)", False))
        trend = ins["trend"] / config.WIND_MAX
        lines.append((f"Wind bar moving {'right' if trend > 0 else 'left'} ({abs(trend):.2f}/s)", False))
        return lines

    def _rules_lines(self, ins: dict) -> list[tuple[str, bool]]:
        lines = [(f"Weapon {WEAPONS[ins['weapon']].name}: {ins['reason']}", True)]
        lines.append((f"Angle {ins['angle']:.0f}° from ahead" + (" (big hill in between)" if ins["high"] else ""), True))
        if ins["estimate"] is None:
            lines.append(("No shot seen land yet: first guess from the distance", True))
        else:
            lines.append((f"Power estimate {ins['estimate']:.1f}, step {ins['step']:.1f}", True))
            if ins["miss"] is not None:
                side = {-1: "short", 1: "long"}.get(ins["side"], "")
                lines.append((f"Last miss {ins['miss']:.0f} px {side}".rstrip(), False))
        lines.append((f"Dial power {ins['power']:.0f}" if ins["power"] is not None else "Dial power -", False))
        state = "dodging" if ins["dodging"] else ("waiting for its shot to land" if ins["in_flight"] and not ins["ready"]
                                                  else ("ready to fire" if ins["ready"] else "turning the dials"))
        lines.append((state.capitalize(), False))
        return lines

    def _label(self, surf, text, pos, color, right=False, small=False, center=False) -> None:
        img = (self.small if small else self.font).render(text, True, color)
        anchor = "midtop" if center else ("topright" if right else "topleft")
        surf.blit(img, img.get_rect(**{anchor: pos}))

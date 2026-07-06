"""Transient visual effects and stateless animation helpers (pure pygame).

Two kinds of animation coexist in the renderer:

* **Transient effects** (the ``Effect`` subclasses): spawned once when an
  event happens (new comm contact, target discovery, newly explored cells),
  live for ``ttl_ms`` milliseconds, then are pruned by ``EffectManager``.

* **Persistent overlays** (comm links, distress beacons, smoke, beacon
  rings): stateless functions of the current env state and the clock,
  redrawn every frame from the module-level helpers below.

All timing is millisecond-based (``renderer._now()``) so visuals run at the
same wall-clock speed in 10 FPS stepped mode and 60 FPS smooth mode.
All comments are in English.
"""

import math

import pygame

from gui import theme


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def ease_smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_pos(a, b, t: float):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class GridView:
    """Grid (r, c) -> pixel conversion shared by every effect."""

    __slots__ = ("cs", "y_off")

    def __init__(self, cs: int, y_off: int):
        self.cs = cs
        self.y_off = y_off

    def center(self, cell):
        r, c = cell
        return (int(c * self.cs + self.cs / 2),
                int(r * self.cs + self.y_off + self.cs / 2))

    def topleft(self, cell):
        r, c = cell
        return int(c * self.cs), int(r * self.cs + self.y_off)


# ---------------------------------------------------------------------------
# Transient effects
# ---------------------------------------------------------------------------

class Effect:
    ttl_ms = 500
    layer = "over"           # "under" draws below entities, "over" above

    def __init__(self, now: int):
        self.t0 = now

    def progress(self, now: int) -> float:
        return (now - self.t0) / self.ttl_ms

    def done(self, now: int) -> bool:
        return self.progress(now) >= 1.0

    def draw(self, overlay, view: GridView, now: int):
        raise NotImplementedError


class CommRipple(Effect):
    """Expanding radio rings at a drone that just made a new comm contact."""

    ttl_ms = 700

    def __init__(self, now, cell, color):
        super().__init__(now)
        self.cell = cell
        self.color = color

    def draw(self, overlay, view, now):
        t = self.progress(now)
        center = view.center(self.cell)
        cs = view.cs
        for lag in (0.0, 0.30):          # two staggered rings
            if t < lag:
                continue
            tt = ease_smoothstep((t - lag) / (1.0 - lag))
            rad = (0.25 + 1.9 * tt) * cs
            alpha = int(210 * (1.0 - tt))
            if alpha > 4:
                pygame.draw.circle(overlay, theme.with_alpha(self.color, alpha),
                                   center, max(2, int(rad)), max(2, cs // 10))


class DiscoveryFlash(Effect):
    """One-shot celebration ring when the team first spots the target."""

    ttl_ms = 900

    def __init__(self, now, cell):
        super().__init__(now)
        self.cell = cell

    def draw(self, overlay, view, now):
        t = ease_smoothstep(self.progress(now))
        center = view.center(self.cell)
        cs = view.cs
        fill_a = int(130 * (1.0 - t))
        if fill_a > 4:
            pygame.draw.circle(overlay, theme.with_alpha(theme.TARGET, fill_a),
                               center, int((0.4 + 0.8 * t) * cs))
        ring_a = int(235 * (1.0 - t))
        if ring_a > 4:
            pygame.draw.circle(overlay, theme.with_alpha(theme.TARGET, ring_a),
                               center, max(2, int((0.4 + 2.6 * t) * cs)),
                               max(2, cs // 8))


class CellFade(Effect):
    """Newly explored cell brightening out of the fog (drawn under entities)."""

    ttl_ms = 450
    layer = "under"

    def __init__(self, now, cell):
        super().__init__(now)
        self.cell = cell

    def draw(self, overlay, view, now):
        alpha = int(255 * (1.0 - self.progress(now)))
        if alpha <= 4:
            return
        x, y = view.topleft(self.cell)
        pygame.draw.rect(overlay, theme.with_alpha(theme.FOG, alpha),
                         pygame.Rect(x, y, view.cs, view.cs))


class EffectManager:
    """Owns live transient effects, split by draw layer."""

    def __init__(self):
        self._under = []
        self._over = []

    def clear(self):
        self._under.clear()
        self._over.clear()

    def spawn(self, effect: Effect):
        (self._under if effect.layer == "under" else self._over).append(effect)

    def draw_under(self, overlay, view, now):
        self._under = [e for e in self._under if not e.done(now)]
        for e in self._under:
            e.draw(overlay, view, now)

    def draw_over(self, overlay, view, now):
        self._over = [e for e in self._over if not e.done(now)]
        for e in self._over:
            e.draw(overlay, view, now)


# ---------------------------------------------------------------------------
# Persistent (stateless) overlay helpers
# ---------------------------------------------------------------------------

def draw_dashed_line(surf, color, p1, p2, width=2, dash=8.0, gap=6.0, offset=0.0):
    """Dashed line whose dashes flow along the segment as `offset` grows."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    period = dash + gap
    t = -(offset % period)
    while t < length:
        a, b = max(0.0, t), min(length, t + dash)
        if b > a:
            pygame.draw.line(surf, color,
                             (x1 + ux * a, y1 + uy * a),
                             (x1 + ux * b, y1 + uy * b), width)
        t += period


def draw_link(overlay, p1, p2, c1, c2, cs, now):
    """Animated comm link: flowing dashes + a pulse dot traveling i -> j."""
    col = theme.blend(c1, c2, 0.5)
    draw_dashed_line(overlay, theme.with_alpha(col, 190), p1, p2,
                     width=max(2, cs // 9),
                     dash=cs * 0.45, gap=cs * 0.30, offset=now * 0.04)
    t = ease_smoothstep((now % 900) / 900.0)
    px, py = lerp_pos(p1, p2, t)
    pygame.draw.circle(overlay, theme.with_alpha((255, 246, 205), 235),
                       (int(px), int(py)), max(2, cs // 7))


def draw_expanding_rings(overlay, center, base_rad, max_rad, color,
                         period_ms, now, width=2, n_rings=2, alpha_max=170):
    """Continuously expanding/fading rings (radio beacon)."""
    for k in range(n_rings):
        t = ((now / period_ms) + k / n_rings) % 1.0
        rad = base_rad + (max_rad - base_rad) * t
        alpha = int(alpha_max * (1.0 - t))
        if alpha > 4:
            pygame.draw.circle(overlay, theme.with_alpha(color, alpha),
                               center, max(2, int(rad)), width)


def draw_smoke(overlay, center, cs, now, seed=0):
    """Stateless rising smoke puffs above a wreck."""
    cx, cy = center
    for k in range(3):
        ph = ((now / 1400.0) + k / 3.0 + 0.13 * seed) % 1.0
        y = cy - cs * (0.25 + 0.85 * ph)
        x = cx + math.sin((ph * 3.1 + k) * 2.4) * cs * 0.10
        alpha = int(110 * (1.0 - ph))
        if alpha > 4:
            pygame.draw.circle(overlay, theme.with_alpha(theme.SMOKE, alpha),
                               (int(x), int(y)),
                               max(1, int(cs * (0.10 + 0.16 * ph))))

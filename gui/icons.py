"""Procedural icon factory for the renderer (pure pygame, no image assets).

Every icon is drawn with primitives at 2x resolution and smoothscaled down
for cheap anti-aliasing, then cached by ``(kind, size, color, phase)``.
``clear_cache()`` must be called whenever the cell size changes (window
resize) so icons are rebuilt at the new size.

Kinds:
    "drone"        quadcopter in the agent colour; 4 rotor-blade phases
    "wreck"        crashed grey quadcopter, tilted, red X accent, scorch mark
    "target_core"  discovered-target beacon core (glow + ring + bright dot)
    "target_ghost" undiscovered target: translucent dashed diamond

All comments are in English.
"""

import math

import pygame

from gui import theme

_cache = {}

DRONE_PHASES = 4          # pre-rendered rotor-blade positions


def clear_cache():
    _cache.clear()


def get_icon(kind, size, color=None, phase=0):
    """Return a cached SRCALPHA surface of `size` x `size` pixels."""
    size = max(6, int(size))
    key = (kind, size, color, phase)
    surf = _cache.get(key)
    if surf is None:
        surf = _BUILDERS[kind](size, color, phase)
        _cache[key] = surf
    return surf


def get_label(text, size, color=(255, 255, 255)):
    """Small cached text label with a dark outline (readable on any colour)."""
    key = ("label", text, size, color)
    surf = _cache.get(key)
    if surf is None:
        f = theme.font(max(9, int(size * 0.5)), bold=True)
        base = f.render(text, True, color)
        dark = f.render(text, True, (18, 20, 26))
        surf = pygame.Surface((base.get_width() + 2, base.get_height() + 2),
                              pygame.SRCALPHA)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            surf.blit(dark, (1 + dx, 1 + dy))
        surf.blit(base, (1, 1))
        _cache[key] = surf
    return surf


# ---------------------------------------------------------------------------
# Builders (draw at 2x, smoothscale down)
# ---------------------------------------------------------------------------

def _quad_hubs(c, off):
    return [(c - off, c - off), (c + off, c - off),
            (c - off, c + off), (c + off, c + off)]


def _draw_quad_frame(surf, c, off, body_color, arm_w, hub_r):
    """X-frame arms + rotor hubs shared by the live drone and the wreck."""
    arm_col = theme.darken(body_color, 55)
    for hx, hy in _quad_hubs(c, off):
        pygame.draw.line(surf, arm_col, (c, c), (hx, hy), arm_w)
        pygame.draw.circle(surf, theme.darken(body_color, 30), (hx, hy), hub_r)


def _build_drone(size, color, phase):
    color = color or theme.AGENT_COLORS[0]
    S = max(24, size * 2)
    surf = pygame.Surface((S, S), pygame.SRCALPHA)
    c = S / 2
    off = S * 0.30
    blade_len = S * 0.155
    blade_w = max(2, int(S * 0.045))
    disc_col = theme.with_alpha(theme.SMOKE, 55)
    blade_col = theme.with_alpha((230, 233, 240), 215)

    _draw_quad_frame(surf, c, off, color, max(2, int(S * 0.06)), S * 0.105)

    # Rotor discs + blades; each phase advances the blades by 22.5 degrees.
    ang0 = math.radians(phase * (90.0 / DRONE_PHASES))
    for k, (hx, hy) in enumerate(_quad_hubs(c, off)):
        pygame.draw.circle(surf, disc_col, (hx, hy), blade_len)
        a = ang0 + (math.pi / 4 if k % 2 else 0.0)   # desync adjacent rotors
        for da in (0.0, math.pi / 2):
            dx = math.cos(a + da) * blade_len
            dy = math.sin(a + da) * blade_len
            pygame.draw.line(surf, blade_col,
                             (hx - dx, hy - dy), (hx + dx, hy + dy), blade_w)
        pygame.draw.circle(surf, theme.darken(color, 30), (hx, hy), S * 0.055)

    # Round body with outline and a small highlight.
    body_r = S * 0.185
    pygame.draw.circle(surf, theme.darken(color, 45), (c, c), body_r + max(1, S // 36))
    pygame.draw.circle(surf, color, (c, c), body_r)
    pygame.draw.circle(surf, theme.brighten(color, 55),
                       (c - body_r * 0.32, c - body_r * 0.32), body_r * 0.32)
    return pygame.transform.smoothscale(surf, (size, size))


def _build_wreck(size, color, phase):
    S = max(24, size * 2)
    base = pygame.Surface((S, S), pygame.SRCALPHA)
    c = S / 2

    # Scorch mark under the wreck (not rotated with the body).
    scorch = pygame.Rect(0, 0, int(S * 0.86), int(S * 0.40))
    scorch.center = (int(c), int(c + S * 0.20))
    pygame.draw.ellipse(base, (0, 0, 0, 95), scorch)

    # Grey dead quadcopter (no blades), tilted to read as "crashed".
    body = pygame.Surface((S, S), pygame.SRCALPHA)
    grey = theme.BROKEN
    _draw_quad_frame(body, c, S * 0.28, grey, max(2, int(S * 0.06)), S * 0.10)
    body_r = S * 0.17
    pygame.draw.circle(body, theme.darken(grey, 40), (c, c), body_r + max(1, S // 36))
    pygame.draw.circle(body, grey, (c, c), body_r)
    body = pygame.transform.rotate(body, 25)
    base.blit(body, body.get_rect(center=(c, c - S * 0.02)))

    # Red X accent on top (kept upright and crisp).
    xr = S * 0.26
    xw = max(3, int(S * 0.085))
    pygame.draw.line(base, theme.BROKEN_X, (c - xr, c - xr), (c + xr, c + xr), xw)
    pygame.draw.line(base, theme.BROKEN_X, (c - xr, c + xr), (c + xr, c - xr), xw)
    return pygame.transform.smoothscale(base, (size, size))


def _build_target_core(size, color, phase):
    S = max(24, size * 2)
    surf = pygame.Surface((S, S), pygame.SRCALPHA)
    c = S / 2
    # Soft glow, beacon ring, bright core.
    pygame.draw.circle(surf, theme.with_alpha(theme.TARGET, 42), (c, c), S * 0.48)
    pygame.draw.circle(surf, theme.with_alpha(theme.TARGET, 80), (c, c), S * 0.37)
    pygame.draw.circle(surf, theme.TARGET, (c, c), S * 0.27, max(2, int(S * 0.07)))
    pygame.draw.circle(surf, (255, 228, 222), (c, c), S * 0.115)
    return pygame.transform.smoothscale(surf, (size, size))


def _build_target_ghost(size, color, phase):
    S = max(24, size * 2)
    surf = pygame.Surface((S, S), pygame.SRCALPHA)
    c = S / 2
    R = S * 0.38
    col = theme.with_alpha(theme.TARGET, 150)   # alpha modulated again at draw time
    pts = [(c, c - R), (c + R, c), (c, c + R), (c - R, c)]
    w = max(2, int(S * 0.05))
    for a, b in zip(pts, pts[1:] + pts[:1]):
        _dashed_segment(surf, col, a, b, dash=S * 0.09, gap=S * 0.07, width=w)
    pygame.draw.circle(surf, col, (c, c), S * 0.06)
    return pygame.transform.smoothscale(surf, (size, size))


def _dashed_segment(surf, color, p1, p2, dash, gap, width):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    while t < length:
        t2 = min(length, t + dash)
        pygame.draw.line(surf, color,
                         (x1 + ux * t, y1 + uy * t),
                         (x1 + ux * t2, y1 + uy * t2), width)
        t = t2 + gap


_BUILDERS = {
    "drone":        _build_drone,
    "wreck":        _build_wreck,
    "target_core":  _build_target_core,
    "target_ghost": _build_target_ghost,
}

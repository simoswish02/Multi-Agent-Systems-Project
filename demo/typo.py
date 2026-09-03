"""Kinetic typography for the demo film.

Every element is drawn with the GUI's own palette and font stack
(``gui.theme``) so the titles look like they belong to the same product as the
footage. Each function takes a normalised progress ``t`` and is otherwise
stateless, so scenes stay declarative and any frame can be re-rendered on its
own.
"""

import math

import pygame

from gui import theme
from demo import config as C


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def clamp(t, lo=0.0, hi=1.0):
    return max(lo, min(hi, t))


def smooth(t):
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t, p=3.0):
    return 1.0 - (1.0 - clamp(t)) ** p


def ease_in(t, p=3.0):
    return clamp(t) ** p


def span(t, start, end=1.0):
    """Map a beat's progress onto a sub-window of it, as a 0..1 span.

    Elements drawn by `lower_third`, `callout` and `title_card` fade in at the
    start of their span and out at the end, so they must be handed a value
    that actually sweeps to 1 -- feeding them a clamped, saturating progress
    parks them at "fading out" and they never appear. Use this instead of
    dividing by hand.
    """
    if end <= start:
        return 1.0
    return clamp((t - start) / (end - start))


def inout(t, rise=0.12, fall=0.12):
    """Fade in over the first `rise` of the span, out over the last `fall`."""
    t = clamp(t)
    a = smooth(t / rise) if rise > 0 else 1.0
    b = smooth((1.0 - t) / fall) if fall > 0 else 1.0
    return min(a, b)


def lerp(a, b, t):
    return a + (b - a) * clamp(t)


def lerp_pt(a, b, t):
    t = clamp(t)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def tracked(surf, text, font, color, pos, tracking=2, alpha=255):
    """Render text with extra letter spacing (pygame has no tracking)."""
    x, y = pos
    for ch in text:
        g = font.render(ch, True, color)
        if alpha < 255:
            g.set_alpha(alpha)
        surf.blit(g, (x, y))
        x += g.get_width() + tracking
    return x - pos[0]


def tracked_width(text, font, tracking=2):
    return sum(font.size(ch)[0] + tracking for ch in text) - tracking


def wrap(text, font, max_w):
    """Greedy word wrap into a list of lines."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def blit_alpha(dst, src, pos, alpha):
    if alpha >= 255:
        dst.blit(src, pos)
        return
    if alpha <= 0:
        return
    src.set_alpha(int(alpha))
    dst.blit(src, pos)
    src.set_alpha(255)


# ---------------------------------------------------------------------------
# Chapter tag: "02 / LIVE SEARCH", top-left gutter
# ---------------------------------------------------------------------------

def chapter(surf, index, label, t, *, pos=(64, 62)):
    a = inout(t, 0.06, 0.10)
    if a <= 0.01:
        return
    alpha = int(255 * a)
    x, y = pos
    x += int(-18 * (1 - ease_out(clamp(t / 0.06))))

    f_num = theme.font(15, bold=True, mono=True)
    f_lab = theme.font(15, bold=True)

    num = f_num.render("%02d" % index, True, theme.ACCENT)
    blit_alpha(surf, num, (x, y), alpha)
    w = num.get_width() + 10

    sep = f_num.render("/", True, theme.BORDER)
    blit_alpha(surf, sep, (x + w, y), alpha)
    w += sep.get_width() + 10

    tracked(surf, label.upper(), f_lab, theme.TEXT_DIM, (x + w, y),
            tracking=3, alpha=alpha)

    rule_w = int(46 * ease_out(clamp(t / 0.12)))
    if rule_w > 1:
        line = pygame.Surface((rule_w, 2), pygame.SRCALPHA)
        line.fill(theme.with_alpha(theme.ACCENT, alpha))
        surf.blit(line, (x, y + 26))


# ---------------------------------------------------------------------------
# Lower third: headline + body, bottom-left
# ---------------------------------------------------------------------------

def lower_third(surf, head, body, t, *, pos=None, width=760, accent=None):
    """Slides up from the bottom-left over a soft scrim."""
    a = inout(t, 0.10, 0.14)
    if a <= 0.01:
        return
    alpha = int(255 * a)
    accent = accent or theme.ACCENT

    f_head = theme.font(C.LOWER_FONT_PX, bold=True)
    f_body = theme.font(21)

    lines = wrap(body, f_body, width - 34) if body else []
    pad = 18
    h = pad * 2 + (f_head.get_height() if head else 0) \
        + (8 if head and lines else 0) + len(lines) * (f_body.get_height() + 4)

    x, y = pos or (64, C.H - 96 - h)
    y += int(26 * (1 - ease_out(clamp(t / 0.10))))

    panel = pygame.Surface((width, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, theme.with_alpha((12, 13, 18), 236),
                     panel.get_rect(), border_radius=6)
    pygame.draw.rect(panel, theme.with_alpha(accent, 255),
                     pygame.Rect(0, 0, 3, h), border_radius=2)
    blit_alpha(surf, panel, (x, y), alpha)

    cy = y + pad
    if head:
        tracked(surf, head.upper(), f_head, accent, (x + 20, cy),
                tracking=2, alpha=alpha)
        cy += f_head.get_height() + (8 if lines else 0)
    for ln in lines:
        g = f_body.render(ln, True, theme.TEXT)
        blit_alpha(surf, g, (x + 20, cy), alpha)
        cy += f_body.get_height() + 4


# ---------------------------------------------------------------------------
# Full-screen title card
# ---------------------------------------------------------------------------

def title_card(surf, lines, sub, t, *, kicker=None, scrim=0.0):
    """Big centred title. `scrim` darkens whatever is behind it."""
    a = inout(t, 0.14, 0.18)
    if a <= 0.01:
        return
    alpha = int(255 * a)

    if scrim > 0:
        sc = pygame.Surface((C.W, C.H), pygame.SRCALPHA)
        sc.fill(theme.with_alpha((8, 9, 13), int(255 * scrim * a)))
        surf.blit(sc, (0, 0))

    f_title = theme.font(C.TITLE_FONT_PX, bold=True)
    f_sub   = theme.font(C.SUB_FONT_PX)
    f_kick  = theme.font(16, bold=True, mono=True)

    total = len(lines) * (f_title.get_height() + 6)
    if sub:
        total += 30 + f_sub.get_height()
    if kicker:
        total += 40
    y = (C.H - total) // 2

    if kicker:
        w = tracked_width(kicker.upper(), f_kick, 5)
        tracked(surf, kicker.upper(), f_kick, theme.ACCENT,
                ((C.W - w) // 2, y), tracking=5, alpha=alpha)
        y += 40

    # Each line rises a little, staggered.
    for k, ln in enumerate(lines):
        lt = clamp((t - 0.03 * k) / 0.16)
        g = f_title.render(ln, True, theme.TEXT)
        dy = int(20 * (1 - ease_out(lt)))
        blit_alpha(surf, g, ((C.W - g.get_width()) // 2, y + dy), alpha)
        y += f_title.get_height() + 6

    if sub:
        y += 22
        rw = int(120 * ease_out(clamp(t / 0.2)))
        if rw > 1:
            line = pygame.Surface((rw, 2), pygame.SRCALPHA)
            line.fill(theme.with_alpha(theme.ACCENT, alpha))
            surf.blit(line, ((C.W - rw) // 2, y))
        y += 20
        for ln in wrap(sub, f_sub, 1100):
            g = f_sub.render(ln, True, theme.TEXT_DIM)
            blit_alpha(surf, g, ((C.W - g.get_width()) // 2, y), alpha)
            y += f_sub.get_height() + 4


# ---------------------------------------------------------------------------
# Callout: dot + elbow leader + label, pointing at something in the frame
# ---------------------------------------------------------------------------

def callout(surf, anchor, text, t, *, side="right", color=None, dist=150,
            title=None):
    """`anchor` is a canvas-space pixel the callout points at."""
    a = inout(t, 0.12, 0.14)
    if a <= 0.01:
        return
    alpha = int(255 * a)
    color = color or theme.INFO
    ax, ay = anchor
    sgn = 1 if side == "right" else -1

    grow = ease_out(clamp(t / 0.16))
    elbow = (ax + sgn * int(34 * grow), ay - int(34 * grow))
    end   = (elbow[0] + sgn * int(dist * grow), elbow[1])

    layer = pygame.Surface((C.W, C.H), pygame.SRCALPHA)
    pygame.draw.line(layer, theme.with_alpha(color, 210), (ax, ay), elbow, 2)
    pygame.draw.line(layer, theme.with_alpha(color, 210), elbow, end, 2)
    pygame.draw.circle(layer, theme.with_alpha(color, 235), (ax, ay), 5, 2)
    pygame.draw.circle(layer, theme.with_alpha(color, 90), (ax, ay),
                       int(5 + 7 * (1 - grow)) if grow < 1 else 5, 1)
    blit_alpha(surf, layer, (0, 0), alpha)

    if grow < 0.55:
        return

    f_t = theme.font(15, bold=True)
    f_b = theme.font(C.CALLOUT_FONT_PX)
    lines = wrap(text, f_b, 320)
    tx = end[0] + sgn * 10
    ty = end[1] - (len(lines) * (f_b.get_height() + 3)) // 2
    if title:
        ty -= f_t.get_height() + 4

    ta = int(alpha * smooth((grow - 0.55) / 0.45))
    if title:
        g = f_t.render(title.upper(), True, color)
        blit_alpha(surf, g, (tx if sgn > 0 else tx - g.get_width(), ty), ta)
        ty += f_t.get_height() + 4
    for ln in lines:
        g = f_b.render(ln, True, theme.TEXT)
        blit_alpha(surf, g, (tx if sgn > 0 else tx - g.get_width(), ty), ta)
        ty += f_b.get_height() + 3


# ---------------------------------------------------------------------------
# Stat cards (outro)
# ---------------------------------------------------------------------------

def stat_cards(surf, stats, t, *, y=None, card=(400, 210), gap=34, alpha=1.0):
    """`stats` is a list of (big_value, headline, footnote).

    `t` drives the staggered entrance only; the caller owns the exit via
    `alpha`, so cards can stay up across beats without flickering out.
    """
    cw, ch = card
    n = len(stats)
    total = n * cw + (n - 1) * gap
    x0 = (C.W - total) // 2
    y = y if y is not None else (C.H - ch) // 2

    f_val  = theme.font(62, bold=True)
    f_head = theme.font(20, bold=True)
    f_foot = theme.font(17)

    for k, (val, head, foot) in enumerate(stats):
        lt = clamp((t - 0.12 * k) / 0.38)          # this card's entrance
        a  = ease_out(lt) * clamp(alpha)
        if a <= 0.01:
            continue
        av = int(255 * a)
        x  = x0 + k * (cw + gap)
        dy = int(26 * (1 - ease_out(lt)))

        panel = pygame.Surface((cw, ch), pygame.SRCALPHA)
        pygame.draw.rect(panel, theme.with_alpha(theme.PANEL, 240),
                         panel.get_rect(), border_radius=8)
        pygame.draw.rect(panel, theme.with_alpha(theme.BORDER, 255),
                         panel.get_rect(), 1, border_radius=8)
        pygame.draw.rect(panel, theme.with_alpha(theme.ACCENT, 255),
                         pygame.Rect(0, 0, int(cw * ease_out(lt)), 3),
                         border_top_left_radius=8, border_top_right_radius=8)
        blit_alpha(surf, panel, (x, y + dy), av)

        g = f_val.render(val, True, theme.TEXT)
        blit_alpha(surf, g, (x + (cw - g.get_width()) // 2, y + dy + 28), av)

        hy = y + dy + 28 + g.get_height() + 8
        for ln in wrap(head, f_head, cw - 44):
            gg = f_head.render(ln, True, theme.ACCENT)
            blit_alpha(surf, gg, (x + (cw - gg.get_width()) // 2, hy), av)
            hy += f_head.get_height() + 2
        hy += 5
        for ln in wrap(foot, f_foot, cw - 40):
            gg = f_foot.render(ln, True, theme.TEXT_DIM)
            blit_alpha(surf, gg, (x + (cw - gg.get_width()) // 2, hy), av)
            hy += f_foot.get_height() + 2


# ---------------------------------------------------------------------------
# Misc chrome
# ---------------------------------------------------------------------------

def progress_rule(surf, t, *, y=None, width=1400, color=None):
    """A hairline that fills across the frame: the film's own progress."""
    y = y if y is not None else C.H - 46
    color = color or theme.BORDER
    x0 = (C.W - width) // 2
    pygame.draw.rect(surf, color, pygame.Rect(x0, y, width, 2))
    pygame.draw.rect(surf, theme.ACCENT,
                     pygame.Rect(x0, y, int(width * clamp(t)), 2))


def cursor(surf, pos, *, click=0.0):
    """Synthetic mouse pointer; `click` in [0,1] plays a press ripple."""
    x, y = int(pos[0]), int(pos[1])
    if click > 0:
        r = int(6 + 20 * click)
        layer = pygame.Surface((C.W, C.H), pygame.SRCALPHA)
        pygame.draw.circle(layer, theme.with_alpha(theme.ACCENT,
                                                   int(150 * (1 - click))),
                           (x, y), r, 2)
        surf.blit(layer, (0, 0))
    pts = [(x, y), (x, y + 17), (x + 4, y + 13), (x + 7, y + 19),
           (x + 10, y + 18), (x + 7, y + 12), (x + 12, y + 12)]
    pygame.draw.polygon(surf, (14, 15, 20), [(p[0] + 1, p[1] + 1) for p in pts])
    pygame.draw.polygon(surf, theme.TEXT, pts)
    pygame.draw.polygon(surf, (14, 15, 20), pts, 1)


def vignette(size, strength=0.55):
    """Cached radial darkening, composited over the whole canvas."""
    key = (size, round(strength, 3))
    if key in _VIGNETTE:
        return _VIGNETTE[key]
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w / 2.0, h / 2.0
    maxd = math.hypot(cx, cy)
    step = 8
    for r in range(int(maxd), 0, -step):
        f = r / maxd
        a = int(255 * strength * max(0.0, (f - 0.55) / 0.45) ** 2)
        if a <= 0:
            continue
        pygame.draw.circle(surf, (0, 0, 0, a), (int(cx), int(cy)), r, step + 1)
    _VIGNETTE[key] = surf
    return surf


_VIGNETTE = {}

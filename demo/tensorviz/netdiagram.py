"""The network, drawn to scale with what it actually is.

Every block, width and label here is read off ``agents/networks.py``: the
ConvNeXt stage widths and depths, the token count fed to the attention layers,
the feature dimensions that meet at the concatenation, and the dueling split.
``demo/tensorviz/verify.py`` checks the numbers against the live module, so the
diagram cannot quietly go stale.
"""

import pygame

from gui import theme
from demo import typo


# ---------------------------------------------------------------------------
# Block primitives
# ---------------------------------------------------------------------------

def block(surf, rect, title, sub, t, *, color=None, fill=None, alpha=255,
          hot=0.0):
    """A rounded schematic block that scales up as it appears."""
    t = typo.clamp(t)
    if t <= 0.02:
        return
    color = color or theme.BORDER
    e = typo.ease_out(t)
    w = max(2, int(rect.width * (0.72 + 0.28 * e)))
    h = max(2, int(rect.height * (0.72 + 0.28 * e)))
    r = pygame.Rect(0, 0, w, h)
    r.center = rect.center
    a = int(alpha * e)

    card = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(card, theme.with_alpha(fill or theme.PANEL_ALT, 240),
                     card.get_rect(), border_radius=6)
    edge = theme.blend(color, theme.WARNING, hot) if hot else color
    pygame.draw.rect(card, theme.with_alpha(edge, 255), card.get_rect(),
                     2 if hot else 1, border_radius=6)
    typo.blit_alpha(surf, card, r.topleft, a)

    if e < 0.5:
        return
    ta = int(a * (e - 0.5) / 0.5)
    f_t = theme.font(19, bold=True)
    f_s = theme.font(14)
    g = f_t.render(title, True, theme.TEXT)
    typo.blit_alpha(surf, g, (r.centerx - g.get_width() // 2,
                              r.centery - (16 if sub else 9)), ta)
    if sub:
        gs = f_s.render(sub, True, theme.TEXT_DIM)
        typo.blit_alpha(surf, gs, (r.centerx - gs.get_width() // 2,
                                   r.centery + 7), int(ta * 0.9))


def arrow(surf, a, b, t, *, color=None, width=2, head=8):
    """A connector that draws itself in, ending in a small head."""
    t = typo.clamp(t)
    if t <= 0.02:
        return
    color = color or theme.BORDER
    end = typo.lerp_pt(a, b, typo.smooth(t))
    pygame.draw.line(surf, color, a, end, width)
    if t > 0.85:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        tip = b
        pygame.draw.polygon(surf, color, [
            tip,
            (tip[0] - ux * head + px * head * 0.5,
             tip[1] - uy * head + py * head * 0.5),
            (tip[0] - ux * head - px * head * 0.5,
             tip[1] - uy * head - py * head * 0.5),
        ])


def tensor_bar(surf, rect, label, t, *, color=None, alpha=255):
    """A feature vector drawn as a slim filled bar with its dimension."""
    t = typo.clamp(t)
    if t <= 0.02:
        return
    color = color or theme.ACCENT
    w = int(rect.width * typo.ease_out(t))
    a = int(alpha * typo.ease_out(t))
    bar = pygame.Surface((max(1, w), rect.height), pygame.SRCALPHA)
    bar.fill(theme.with_alpha(color, 205))
    typo.blit_alpha(surf, bar, rect.topleft, a)
    if t > 0.6:
        f = theme.font(15, bold=True, mono=True)
        g = f.render(label, True, theme.TEXT)
        typo.blit_alpha(surf, g, (rect.right + 12,
                                  rect.centery - g.get_height() // 2),
                        int(a * (t - 0.6) / 0.4))


# ---------------------------------------------------------------------------
# The trunk
# ---------------------------------------------------------------------------

# (title, sub, relative width) -- straight from agents/networks.py
GLOBAL_TRUNK = [
    ("stem",       "conv 3x3 -> 96",     150),
    ("stage 1",    "96ch  x3  @32x32",   160),
    ("stage 2",    "192ch x4  @16x16",   160),
    ("stage 3",    "384ch x6  @8x8",     160),
    ("attention",  "3 x MHSA, 66 tokens", 200),
]

LOCAL_TRUNK = [
    ("stem",      "conv 3x3 -> 96",       150),
    ("3 blocks",  "dilations 1, 2, 1",    160),
    ("FiLM",      "conditioned on global", 180),
]


def trunk(surf, x0, y, blocks, t, *, height=76, gap=26, color=None,
          stagger=0.10, hot=None):
    """Draw a row of blocks, each appearing a little after the last.

    Returns the x where the row ends, so the caller can attach what follows.
    """
    x = x0
    rects = []
    for k, (title, sub, w) in enumerate(blocks):
        lt = typo.clamp((t - stagger * k) / max(1e-6, 1.0 - stagger * len(blocks)))
        r = pygame.Rect(x, y - height // 2, w, height)
        block(surf, r, title, sub, lt, color=color,
              hot=0.0 if hot is None else hot(k))
        if k:
            arrow(surf, (rects[-1].right, y), (r.left, y),
                  typo.clamp((lt - 0.2) / 0.4), color=theme.BORDER)
        rects.append(r)
        x = r.right + gap
    return rects


# ---------------------------------------------------------------------------
# The dueling head and the Q-values
# ---------------------------------------------------------------------------

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]   # env/grid_env.py::ACTIONS


def dueling(surf, rect, t, *, v_val=None, a_vals=None):
    """The V / A split, with the live scalars once they are known."""
    half = rect.height // 2 - 10
    rv = pygame.Rect(rect.x, rect.y, rect.width, half)
    ra = pygame.Rect(rect.x, rect.bottom - half, rect.width, half)
    block(surf, rv, "V(s)", "one number: is this a good place to be?",
          typo.clamp(t / 0.6), color=theme.INFO)
    block(surf, ra, "A(s, a)", "four numbers: how much better is each move?",
          typo.clamp((t - 0.2) / 0.6), color=theme.WARNING)
    return rv, ra


def q_bars(surf, origin, q, t, *, mask=None, chosen=None, width=460,
           row_h=54, gap=12, now=0.0):
    """The four Q-values as horizontal bars, with the argmax called out.

    `q` is the raw network output. Masked-out actions are shown greyed, which
    is what ``select_action`` does to them before the argmax.
    """
    x, y = origin
    f_a = theme.font(21, bold=True, mono=True)
    f_v = theme.font(20, bold=True, mono=True)
    f_n = theme.font(15)

    finite = [v for k, v in enumerate(q)
              if mask is None or mask[k]]
    lo = min(finite) if finite else 0.0
    hi = max(finite) if finite else 1.0
    span = max(1e-6, hi - lo)

    for k in range(len(q)):
        lt = typo.clamp((t - 0.08 * k) / 0.42)
        if lt <= 0.01:
            continue
        a = int(255 * typo.ease_out(lt))
        yy = y + k * (row_h + gap)
        valid = mask is None or bool(mask[k])
        is_best = (chosen == k)

        col = theme.ACCENT if is_best else (theme.SURFACE if valid
                                            else (46, 48, 58))
        frac = 0.10 + 0.90 * ((q[k] - lo) / span) if valid else 0.06
        bw = int(width * frac * typo.ease_out(lt))

        track = pygame.Surface((width, row_h), pygame.SRCALPHA)
        pygame.draw.rect(track, theme.with_alpha((26, 28, 36), 220),
                         track.get_rect(), border_radius=5)
        typo.blit_alpha(surf, track, (x + 96, yy), a)

        bar = pygame.Surface((max(2, bw), row_h), pygame.SRCALPHA)
        pygame.draw.rect(bar, theme.with_alpha(col, 240), bar.get_rect(),
                         border_radius=5)
        typo.blit_alpha(surf, bar, (x + 96, yy), a)

        name_col = theme.TEXT if valid else theme.TEXT_DIM
        g = f_a.render(ACTION_NAMES[k], True,
                       theme.ACCENT if is_best else name_col)
        typo.blit_alpha(surf, g, (x, yy + row_h // 2 - g.get_height() // 2), a)

        if valid:
            g = f_v.render("%+.2f" % q[k], True,
                           (16, 22, 18) if is_best else theme.TEXT)
            typo.blit_alpha(surf, g, (x + 96 + 14,
                                      yy + row_h // 2 - g.get_height() // 2), a)
        else:
            g = f_n.render("blocked -- wall or edge", True, theme.TEXT_DIM)
            typo.blit_alpha(surf, g, (x + 96 + 14,
                                      yy + row_h // 2 - g.get_height() // 2), a)

        if is_best and t > 0.75:
            pulse = 0.5 + 0.5 * __import__("math").sin(now / 190.0)
            ring = pygame.Rect(x + 90, yy - 4, width + 12, row_h + 8)
            pygame.draw.rect(surf, theme.blend(theme.ACCENT, theme.TEXT,
                                               0.35 * pulse),
                             ring, 2, border_radius=7)
            g = f_n.render("argmax -> this move", True, theme.ACCENT)
            typo.blit_alpha(surf, g, (x + 96 + width + 26,
                                      yy + row_h // 2 - g.get_height() // 2),
                            int(255 * typo.clamp((t - 0.75) / 0.2)))


def equation(surf, center, t, alpha=255):
    """Q = V + A - mean(A), the line that makes the split add up."""
    t = typo.clamp(t)
    if t <= 0.02:
        return
    f = theme.font(25, bold=True, mono=True)
    parts = [("Q", theme.TEXT), (" = ", theme.TEXT_DIM),
             ("V", theme.INFO), (" + ", theme.TEXT_DIM),
             ("A", theme.WARNING), (" - ", theme.TEXT_DIM),
             ("mean(A)", theme.TEXT_DIM)]
    total = sum(f.size(p)[0] for p, _ in parts)
    x = center[0] - total // 2
    a = int(alpha * typo.ease_out(t))
    for text, col in parts:
        g = f.render(text, True, col)
        typo.blit_alpha(surf, g, (x, center[1] - g.get_height() // 2), a)
        x += g.get_width()

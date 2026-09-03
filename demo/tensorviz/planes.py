"""Channel planes: the drone's observation drawn as stacked matrices.

A plane is one 32x32 channel rendered as a heat grid in a single hue, sheared
into an isometric card so a stack of six reads as depth rather than as six
unrelated thumbnails. Every value comes from the observation the network is
about to be handed.
"""



import pygame

from gui import theme
from demo import typo


# The six global channels and the five local ones, exactly as
# env/grid_env.py documents them.
GLOBAL_CHANNELS = [
    ("Visited",      theme.FREE,          "cells this drone has seen as free"),
    ("Obstacle",     (150, 158, 184),     "walls it has discovered"),
    ("Trajectory",   theme.WARNING,       "how often it revisited each cell"),
    ("Target",       theme.TARGET,        "the target, if it ever saw it"),
    ("Own_Position", theme.INFO,          "a single 1.0 at its own cell"),
    ("Broken",       theme.BROKEN_X,      "crash sites it knows about"),
]

LOCAL_CHANNELS = [
    ("Visited",        theme.FREE,        "same belief, fine grained"),
    ("Obstacle",       (150, 158, 184),   ""),
    ("Trajectory",     theme.WARNING,     ""),
    ("Target",         theme.TARGET,      ""),
    ("Other_Position", theme.ACCENT,      "live teammates inside vision"),
]

SHEAR = 0.34          # horizontal skew per unit of height: the isometric look


# ---------------------------------------------------------------------------
# One plane
# ---------------------------------------------------------------------------

_PLANE_CACHE = {}
_SHEAR_CACHE = {}
_CACHE_CAP = 400


def _cache(store, key, make):
    surf = store.get(key)
    if surf is None:
        if len(store) > _CACHE_CAP:
            store.clear()
        surf = store[key] = make()
    return surf


def plane_surface(arr, color, cell=9, *, grid=True, floor=(20, 22, 29)):
    """Render a 2-D array as a heat grid in one hue. Returns a Surface.

    Cached on the array's contents: during the network scene the episode is
    frozen for seconds at a time, so the same six planes would otherwise be
    rebuilt cell by cell sixty times a second.
    """
    return _cache(_PLANE_CACHE,
                  (arr.tobytes(), arr.shape, tuple(color[:3]), cell, grid),
                  lambda: _build_plane(arr, color, cell, grid, floor))


def _build_plane(arr, color, cell, grid, floor):
    h, w = arr.shape
    surf = pygame.Surface((w * cell, h * cell), pygame.SRCALPHA)
    surf.fill((*floor, 235))

    r0, g0, b0 = floor
    r1, g1, b1 = color[:3]
    for y in range(h):
        row = arr[y]
        for x in range(w):
            v = float(row[x])
            if v <= 0.004:
                continue
            v = min(1.0, v)
            col = (int(r0 + (r1 - r0) * v),
                   int(g0 + (g1 - g0) * v),
                   int(b0 + (b1 - b0) * v))
            surf.fill(col, pygame.Rect(x * cell, y * cell, cell, cell))

    if grid and cell >= 6:
        line = (*theme.BORDER, 46)
        for x in range(0, w + 1):
            pygame.draw.line(surf, line, (x * cell, 0), (x * cell, h * cell))
        for y in range(0, h + 1):
            pygame.draw.line(surf, line, (0, y * cell), (w * cell, y * cell))
    pygame.draw.rect(surf, (*color[:3], 190), surf.get_rect(), 1)
    return surf


def plane_card(arr, color, cell=9, amount=SHEAR, *, grid=True):
    """A plane, sheared -- the form actually drawn in a stack.

    Cached on the array's contents rather than on the intermediate surface, so
    a cleared cache can never hand back the wrong picture.
    """
    return _cache(
        _SHEAR_CACHE,
        (arr.tobytes(), arr.shape, tuple(color[:3]), cell, grid,
         round(amount, 3)),
        lambda: shear(plane_surface(arr, color, cell, grid=grid), amount))


def shear(surf, amount=SHEAR):
    """Skew a surface horizontally so a stack of planes reads as depth.

    Rows are shifted progressively: cheap, and it keeps the cell grid crisp
    (a rotation would resample it into mush).
    """
    w, h = surf.get_size()
    out = pygame.Surface((w + int(abs(amount) * h), h), pygame.SRCALPHA)
    for y in range(h):
        dx = int((h - y) * amount)
        out.blit(surf, (dx, y), pygame.Rect(0, y, w, 1))
    return out


# ---------------------------------------------------------------------------
# A stack of planes
# ---------------------------------------------------------------------------

class Stack:
    """A set of channel planes drawn as a receding stack.

    `spread` in [0,1] fans the planes apart -- 0 is a single collapsed map,
    1 is the fully exploded stack with every channel labelled.
    """

    def __init__(self, channels, *, cell=9, gap=74, label_side="left"):
        self.channels = channels
        self.cell = cell
        self.gap = gap
        self.label_side = label_side

    def height(self, spread=1.0):
        return int(self.gap * spread * (len(self.channels) - 1))

    def draw(self, surf, arrays, origin, spread, *, alpha=255, focus=None,
             labels=True, label_alpha=None, order=None):
        """Draw the stack. `focus` dims every plane but that index."""
        n = len(arrays)
        ox, oy = origin
        spread = typo.clamp(spread)
        label_alpha = alpha if label_alpha is None else label_alpha
        f_name = theme.font(19, bold=True)
        f_note = theme.font(15)

        seq = order if order is not None else range(n - 1, -1, -1)
        for k in seq:                       # back to front
            arr = arrays[k]
            name, color, note = self.channels[k]
            card = plane_card(arr, color, self.cell)
            y = oy + int(self.gap * spread * k)
            x = ox - int(self.cell * arr.shape[1] * 0.0)

            a = alpha
            if focus is not None and k != focus:
                a = int(alpha * 0.22)
            if a < 255:
                card = card.copy()          # cached: never mutate in place
                card.set_alpha(max(0, a))
            surf.blit(card, (x, y))

            if not labels or spread < 0.35:
                continue
            la = int(label_alpha * typo.smooth((spread - 0.35) / 0.35))
            if focus is not None and k != focus:
                la = int(la * 0.35)
            if la <= 6:
                continue
            cw = card.get_width()
            ty = y + card.get_height() // 2 - 16
            if self.label_side == "left":
                tx = x - 22
                _right_label(surf, f_name, f_note, name, note, color,
                             (tx, ty), la, k)
            else:
                _left_label(surf, f_name, f_note, name, note, color,
                            (x + cw + 22, ty), la, k)


def _right_label(surf, f_name, f_note, name, note, color, pos, alpha, idx):
    """Label to the LEFT of the plane, right-aligned against it."""
    x, y = pos
    num = f_note.render("%d" % idx, True, theme.TEXT_DIM)
    g = f_name.render(name, True, color)
    typo.blit_alpha(surf, g, (x - g.get_width(), y), alpha)
    typo.blit_alpha(surf, num, (x - g.get_width() - 22, y + 3), int(alpha * 0.8))
    if note:
        gn = f_note.render(note, True, theme.TEXT_DIM)
        typo.blit_alpha(surf, gn, (x - gn.get_width(), y + 24), int(alpha * 0.85))


def _left_label(surf, f_name, f_note, name, note, color, pos, alpha, idx):
    x, y = pos
    num = f_note.render("%d" % idx, True, theme.TEXT_DIM)
    typo.blit_alpha(surf, num, (x, y + 3), int(alpha * 0.8))
    g = f_name.render(name, True, color)
    typo.blit_alpha(surf, g, (x + 22, y), alpha)
    if note:
        gn = f_note.render(note, True, theme.TEXT_DIM)
        typo.blit_alpha(surf, gn, (x + 22, y + 24), int(alpha * 0.85))


# ---------------------------------------------------------------------------
# A grid of planes
# ---------------------------------------------------------------------------

class Grid:
    """Channel planes laid out so every one of them is actually readable.

    A stack conveys "these are channels of one tensor", but six sheared planes
    at a legible size occlude each other almost completely -- you see depth and
    no content. So the planes start collapsed on the first slot and fan out
    into a grid: `spread` 0 is the single map they came from, 1 is all six
    side by side with their labels.
    """

    def __init__(self, channels, *, cols=3, cell=8, gap_x=56, gap_y=92):
        self.channels = channels
        self.cols = cols
        self.cell = cell
        self.gap_x = gap_x
        self.gap_y = gap_y

    def plane_size(self, grid_n=32):
        w = grid_n * self.cell
        return int(w + SHEAR * w), w

    def slot(self, k, origin, grid_n=32):
        pw, ph = self.plane_size(grid_n)
        col, row = k % self.cols, k // self.cols
        return (origin[0] + col * (pw + self.gap_x),
                origin[1] + row * (ph + self.gap_y))

    def size(self, n, grid_n=32):
        pw, ph = self.plane_size(grid_n)
        rows = (n + self.cols - 1) // self.cols
        return (self.cols * pw + (self.cols - 1) * self.gap_x,
                rows * ph + (rows - 1) * self.gap_y)

    def draw(self, surf, arrays, origin, spread, *, alpha=255, focus=None,
             labels=True, stagger=0.55):
        """Fan the planes out to their slots and label them."""
        spread = typo.clamp(spread)
        n = len(arrays)
        f_name = theme.font(19, bold=True)
        f_note = theme.font(14)
        home = self.slot(0, origin, arrays[0].shape[1])

        for k in range(n - 1, -1, -1):        # back to front while collapsed
            arr = arrays[k]
            name, color, note = self.channels[k]
            card = plane_card(arr, color, self.cell)

            # Each plane leaves the pile a little after the one before it.
            lt = typo.clamp((spread - stagger * k / max(1, n - 1))
                            / max(1e-6, 1.0 - stagger))
            e = typo.smooth(lt)
            tgt = self.slot(k, origin, arr.shape[1])
            x = int(home[0] + (tgt[0] - home[0]) * e)
            y = int(home[1] + (tgt[1] - home[1]) * e)

            a = alpha
            if focus is not None and k != focus:
                a = int(alpha * 0.26)
            if a < 255:
                card = card.copy()            # cached: never mutate in place
                card.set_alpha(max(0, a))
            surf.blit(card, (x, y))

            if not labels or e < 0.55:
                continue
            la = int(alpha * typo.smooth((e - 0.55) / 0.45))
            if focus is not None and k != focus:
                la = int(la * 0.4)
            if la <= 6:
                continue
            ly = y + card.get_height() + 10
            num = f_note.render("%d" % k, True, theme.TEXT_DIM)
            typo.blit_alpha(surf, num, (x + 6, ly + 3), int(la * 0.8))
            g = f_name.render(name, True, color)
            typo.blit_alpha(surf, g, (x + 26, ly), la)
            if note:
                gn = f_note.render(note, True, theme.TEXT_DIM)
                typo.blit_alpha(surf, gn, (x + 26, ly + 23), int(la * 0.85))

    def cell_px(self, k, origin, r, c, grid_n=32):
        """Where grid cell (r, c) of plane `k` lands on screen, shear included."""
        x, y = self.slot(k, origin, grid_n)
        w = grid_n * self.cell
        return (x + int((w - r * self.cell) * SHEAR) + c * self.cell
                + self.cell // 2,
                y + r * self.cell + self.cell // 2)


# ---------------------------------------------------------------------------
# The 13x13 crop
# ---------------------------------------------------------------------------

def patch_surface(patch, channels, cell=26, *, alpha=255):
    """The drone-centred patch, channels composited into one readable grid."""
    surf = _cache(_PLANE_CACHE, ("patch", patch.tobytes(), patch.shape, cell),
                  lambda: _build_patch(patch, channels, cell))
    if alpha < 255:
        surf = surf.copy()
        surf.set_alpha(alpha)
    return surf


def _build_patch(patch, channels, cell):
    n, h, w = patch.shape
    surf = pygame.Surface((w * cell, h * cell), pygame.SRCALPHA)
    surf.fill((18, 20, 27, 240))

    for k in range(n):
        color = channels[k][1]
        for y in range(h):
            for x in range(w):
                v = float(patch[k, y, x])
                if v <= 0.004:
                    continue
                c = (*color[:3], int(min(255, 60 + 195 * min(1.0, v))))
                cellsurf = pygame.Surface((cell - 1, cell - 1), pygame.SRCALPHA)
                cellsurf.fill(c)
                surf.blit(cellsurf, (x * cell, y * cell))

    line = (*theme.BORDER, 70)
    for x in range(w + 1):
        pygame.draw.line(surf, line, (x * cell, 0), (x * cell, h * cell))
    for y in range(h + 1):
        pygame.draw.line(surf, line, (0, y * cell), (w * cell, y * cell))

    # The centre cell is the drone: mark it, since that is the whole point.
    cx, cy = w // 2, h // 2
    pygame.draw.rect(surf, theme.INFO,
                     pygame.Rect(cx * cell, cy * cell, cell, cell), 2)
    pygame.draw.rect(surf, (*theme.ACCENT, 220), surf.get_rect(), 2)
    return surf


# ---------------------------------------------------------------------------
# The context vector
# ---------------------------------------------------------------------------

CTX_LABELS = [
    ("vision",   "how far it sees"),
    ("comm",     "how far it talks"),
    ("team",     "how many drones"),
    ("id",       "which drone it is"),
    ("alive",    "how many it believes are left"),
]


def ctx_chips(surf, origin, ctx, raws, t, *, focus=None, width=232, gap=14):
    """Five chips showing raw -> normalised, exactly as ctx_to_numpy computes."""
    x, y = origin
    f_key = theme.font(16, bold=True)
    f_val = theme.font(34, bold=True, mono=True)
    f_raw = theme.font(15, mono=True)
    f_sub = theme.font(14)

    for k, (key, note) in enumerate(CTX_LABELS):
        lt = typo.clamp((t - 0.09 * k) / 0.34)
        if lt <= 0.01:
            continue
        a = int(255 * typo.ease_out(lt))
        hot = (focus == k)
        col = theme.WARNING if hot else theme.ACCENT
        cx = x + k * (width + gap)
        dy = int(20 * (1 - typo.ease_out(lt)))

        card = pygame.Surface((width, 130), pygame.SRCALPHA)
        pygame.draw.rect(card, theme.with_alpha(theme.PANEL, 244),
                         card.get_rect(), border_radius=7)
        pygame.draw.rect(card, theme.with_alpha(col if hot else theme.BORDER,
                                                255),
                         card.get_rect(), 2 if hot else 1, border_radius=7)
        typo.blit_alpha(surf, card, (cx, y + dy), a)

        g = f_key.render(key.upper(), True, col)
        typo.blit_alpha(surf, g, (cx + 16, y + dy + 12), a)

        g = f_val.render("%.2f" % ctx[k], True, theme.TEXT)
        typo.blit_alpha(surf, g, (cx + 16, y + dy + 36), a)

        g = f_raw.render(raws[k], True, theme.TEXT_DIM)
        typo.blit_alpha(surf, g, (cx + 16, y + dy + 78), a)

        g = f_sub.render(note, True, theme.TEXT_DIM)
        if g.get_width() > width - 28:
            g = f_sub.render(note[:26] + "...", True, theme.TEXT_DIM)
        typo.blit_alpha(surf, g, (cx + 16, y + dy + 100), int(a * 0.9))


# ---------------------------------------------------------------------------
# Connective tissue
# ---------------------------------------------------------------------------

def flow_line(surf, a, b, t, color, *, width=2, dots=3, now=0.0):
    """An animated connector: it draws itself in, then pulses travel along it."""
    t = typo.clamp(t)
    if t <= 0.01:
        return
    end = typo.lerp_pt(a, b, typo.smooth(t))
    pygame.draw.line(surf, color, a, end, width)
    if t < 0.999:
        pygame.draw.circle(surf, color, (int(end[0]), int(end[1])), width + 1)
        return
    for k in range(dots):
        p = ((now / 1400.0) + k / float(dots)) % 1.0
        px, py = typo.lerp_pt(a, b, p)
        pygame.draw.circle(surf, (255, 248, 214), (int(px), int(py)), width + 1)


def brace(surf, x, y0, y1, color, label, t, *, side=1, alpha=255):
    """A curly-ish brace grouping a stack, with a label."""
    t = typo.clamp(t)
    if t <= 0.02:
        return
    ym = (y0 + y1) / 2.0
    h = (y1 - y0) * t / 2.0
    pygame.draw.line(surf, color, (x, ym - h), (x, ym + h), 2)
    for yy in (ym - h, ym + h):
        pygame.draw.line(surf, color, (x, yy), (x + 10 * side, yy), 2)
    pygame.draw.line(surf, color, (x, ym), (x - 10 * side, ym), 2)
    if t > 0.7 and label:
        f = theme.font(17, bold=True)
        g = f.render(label, True, color)
        gx = x - 18 - g.get_width() if side > 0 else x + 18
        typo.blit_alpha(surf, g, (gx, ym - g.get_height() // 2),
                        int(alpha * (t - 0.7) / 0.3))

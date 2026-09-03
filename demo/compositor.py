"""The 1920x1080 canvas: places captured GUI frames, moves a virtual camera
over them, and handles scrims and transitions.

The captured GUI is 1396x940 of pixel-perfect UI. It is never blindly upscaled:
at zoom 1 it sits 1:1 inside the canvas, and when a scene wants a panel read it
crops that region and scales *it* up, so the 12 px sidebar text becomes legible
instead of merely bigger.
"""

import pygame

from gui import theme
from demo import config as C
from demo import typo


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Cam:
    """A view of the captured frame, in captured-frame pixel coordinates.

    `zoom` 1.0 shows the whole frame; 2.0 shows half of it, twice as large.
    """

    __slots__ = ("cx", "cy", "zoom")

    def __init__(self, cx=None, cy=None, zoom=1.0):
        self.cx, self.cy, self.zoom = cx, cy, zoom

    def resolved(self, src_size):
        sw, sh = src_size
        cx = sw / 2.0 if self.cx is None else self.cx
        cy = sh / 2.0 if self.cy is None else self.cy
        return cx, cy, max(1.0, self.zoom)

    def lerp(self, other, t, ease=True):
        t = typo.smooth(t) if ease else typo.clamp(t)
        a, b = self, other
        return Cam(_mix(a.cx, b.cx, t), _mix(a.cy, b.cy, t),
                   _mix(a.zoom, b.zoom, t))


def _mix(a, b, t):
    if a is None or b is None:
        return b if t > 0.5 else a
    return a + (b - a) * t


def frame_rect(rect, src_size, pad=1.12, max_zoom=3.0):
    """Camera framing `rect` as tightly as the frame's aspect ratio allows.

    The panel always keeps the captured aspect, so the zoom that fits `rect`
    depends only on how much of the source it occupies.
    """
    sw, sh = src_size
    w = max(1.0, rect.width * pad)
    h = max(1.0, rect.height * pad)
    zoom = min(sw / w, sh / h)          # limited by the tighter axis
    return Cam(rect.centerx, rect.centery, max(1.0, min(zoom, max_zoom)))


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class Canvas:
    """One 1920x1080 output frame under construction."""

    def __init__(self, size=None):
        self.size = size or (C.W, C.H)
        self.surf = pygame.Surface(self.size).convert() \
            if pygame.display.get_surface() else pygame.Surface(self.size)
        self.panel_rect = None      # where the GUI landed, canvas space
        self._src_size = None
        self._cam = None

    # -- background --------------------------------------------------------
    def backdrop(self, tint=None):
        self.surf.fill(tint or (10, 11, 15))

    def vignette(self, strength=0.5):
        self.surf.blit(typo.vignette(self.size, strength), (0, 0))

    # -- the captured GUI --------------------------------------------------
    def place(self, src, cam=None, *, margin=70, shadow=True, border=True):
        """Blit the captured GUI (a pygame Surface) through `cam`.

        Returns the canvas-space rect it occupies, so callouts can be anchored
        with `to_canvas()`.
        """
        cw, chh = self.size
        sw, sh = src.get_size()

        # The panel keeps the captured aspect ratio and fits inside the margins.
        avail_w, avail_h = cw - 2 * margin, chh - 2 * margin
        scale = min(avail_w / sw, avail_h / sh, 1.0)
        pw, ph = int(sw * scale), int(sh * scale)
        px, py = (cw - pw) // 2, (chh - ph) // 2
        rect = pygame.Rect(px, py, pw, ph)

        cam = cam or Cam()
        cx, cy, zoom = cam.resolved((sw, sh))
        vw, vh = sw / zoom, sh / zoom
        vx = min(max(cx - vw / 2.0, 0.0), sw - vw)
        vy = min(max(cy - vh / 2.0, 0.0), sh - vh)
        view = pygame.Rect(int(vx), int(vy), max(1, int(vw)), max(1, int(vh)))

        if shadow:
            sh_surf = pygame.Surface((pw + 36, ph + 36), pygame.SRCALPHA)
            pygame.draw.rect(sh_surf, (0, 0, 0, 120), sh_surf.get_rect(),
                             border_radius=14)
            self.surf.blit(sh_surf, (px - 18, py - 12))

        crop = src.subsurface(view) if zoom > 1.0 else src
        if crop.get_size() != (pw, ph):
            crop = pygame.transform.smoothscale(crop, (pw, ph))
        self.surf.blit(crop, (px, py))

        if border:
            pygame.draw.rect(self.surf, theme.BORDER, rect, 1)

        self.panel_rect = rect
        self._src_size = (sw, sh)
        self._cam = (view, rect)
        return rect

    def place_in(self, rect, src, cam=None, *, border=True, label=None,
                 label_color=None):
        """Blit a captured frame into an arbitrary rect (split-screen panels).

        Unlike `place`, this does not become the canvas's active view, so
        `to_canvas` keeps referring to whatever `place` last put down.
        """
        sw, sh = src.get_size()
        cam = cam or Cam()
        cx, cy, zoom = cam.resolved((sw, sh))
        vw, vh = sw / zoom, sh / zoom
        vx = min(max(cx - vw / 2.0, 0.0), sw - vw)
        vy = min(max(cy - vh / 2.0, 0.0), sh - vh)
        view = pygame.Rect(int(vx), int(vy), max(1, int(vw)), max(1, int(vh)))

        crop = src.subsurface(view)
        if crop.get_size() != rect.size:
            crop = pygame.transform.smoothscale(crop, rect.size)
        self.surf.blit(crop, rect.topleft)
        if border:
            pygame.draw.rect(self.surf, label_color or theme.BORDER, rect, 1)
        if label:
            f = theme.font(19, bold=True)
            g = f.render(label, True, label_color or theme.TEXT)
            self.surf.blit(g, (rect.x, rect.bottom + 12))
        return rect

    def to_canvas(self, pt):
        """Map a captured-frame pixel to canvas space under the current view."""
        if self._cam is None:
            return pt
        view, rect = self._cam
        fx = (pt[0] - view.x) / view.width
        fy = (pt[1] - view.y) / view.height
        return (rect.x + fx * rect.width, rect.y + fy * rect.height)

    def rect_to_canvas(self, r):
        x0, y0 = self.to_canvas((r.left, r.top))
        x1, y1 = self.to_canvas((r.right, r.bottom))
        return pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    # -- emphasis ----------------------------------------------------------
    def scrim(self, alpha):
        """Darken the whole canvas (for title cards and transitions)."""
        if alpha <= 0:
            return
        s = pygame.Surface(self.size, pygame.SRCALPHA)
        s.fill((8, 9, 13, int(min(255, alpha))))
        self.surf.blit(s, (0, 0))

    def spotlight(self, keep, amount=0.62, *, radius=8, ring=True):
        """Dim everything except `keep` (a canvas-space Rect)."""
        if amount <= 0:
            return
        a = int(255 * amount)
        s = pygame.Surface(self.size, pygame.SRCALPHA)
        s.fill((8, 9, 13, a))
        pygame.draw.rect(s, (0, 0, 0, 0), keep, border_radius=radius)
        self.surf.blit(s, (0, 0))
        if ring:
            pygame.draw.rect(self.surf, theme.ACCENT, keep, 2,
                             border_radius=radius)

    # -- output ------------------------------------------------------------
    def bytes(self):
        return pygame.image.tobytes(self.surf, "RGB")


# ---------------------------------------------------------------------------
# Transitions between finished frames
# ---------------------------------------------------------------------------

def crossfade(a, b, t):
    """Blend two finished canvases; returns a new Surface."""
    out = a.copy()
    b = b.copy()
    b.set_alpha(int(255 * typo.clamp(t)))
    out.blit(b, (0, 0))
    return out


def dip(surf, t, color=(8, 9, 13)):
    """Dip to `color` and back: t in [0,1], darkest at 0.5."""
    k = 1.0 - abs(t - 0.5) * 2.0
    if k <= 0:
        return surf
    s = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    s.fill((*color, int(255 * typo.smooth(k))))
    surf.blit(s, (0, 0))
    return surf


def wipe(a, b, t, *, width=140):
    """Hard-edged accent wipe from left to right."""
    out = a.copy()
    w = out.get_width()
    x = int(-width + (w + width) * typo.smooth(typo.clamp(t)))
    if x > 0:
        out.blit(b, (0, 0), pygame.Rect(0, 0, min(x, w), out.get_height()))
    if 0 <= x <= w:
        pygame.draw.rect(out, theme.ACCENT, pygame.Rect(x - 3, 0, 3,
                                                        out.get_height()))
    return out

"""Scene 1 -- the setup screen.

The real widgets from ``gui/config_screen.py`` are instantiated with the real
ranges (which come from ``domain_randomization``, so the screen can only offer
configurations the policy was actually trained to handle) and driven
programmatically: a synthetic cursor travels to each slider and drags it, and
``gui.theme.set_mouse_override`` makes the widgets report the hover states a
user would see.

Nothing is mocked up. What the film shows is the screen you get from
``python main.py --mode simulate``.
"""

import pygame

from gui import theme
from gui.config_screen import Slider, FloatSlider, Toggle, TextField
from demo import config as C
from demo import typo
from demo.compositor import Canvas


PANEL_SCALE = 1.35          # the 560x716 screen, enlarged to fill the frame
PANEL_RIGHT = 110           # its margin from the right edge
TEXT_X      = 96            # the copy lives in its own left-hand column


# ---------------------------------------------------------------------------
# A drivable copy of the setup screen
# ---------------------------------------------------------------------------

class Screen:
    """The setup screen's widgets, laid out exactly as ``ConfigScreen.run``."""

    def __init__(self, cfg):
        from gui.config_screen import ConfigScreen
        self.cs = ConfigScreen(cfg)
        cs = self.cs
        self.size = (cs.W, cs.H)

        pad, gap = 28, 56
        x, w, y = pad, cs.W - 2 * pad, 84
        self.pad = pad

        def nxt(maker):
            nonlocal y
            wid = maker(y)
            y += gap
            return wid

        self.agents = nxt(lambda yy: Slider(x, yy, w, "Number of drones",
                                            1, cs.a_max, 1))
        self.vision = nxt(lambda yy: Slider(x, yy, w, "Vision radius",
                                            1, cs.v_max, 3))
        self.comm   = nxt(lambda yy: Slider(x, yy, w, "Communication radius",
                                            1, cs.c_max, 5))
        self.dens   = nxt(lambda yy: FloatSlider(x, yy, w, "Obstacle density",
                                                 cs.d_lo, cs.d_hi, 0.20,
                                                 step=0.01, fmt="{:.2f}"))
        self.steps  = nxt(lambda yy: Slider(x, yy, w, "Max steps", 50, 600, 200))
        self.eps    = nxt(lambda yy: Slider(x, yy, w, "Episodes to simulate",
                                            1, 50, 10))
        self.fault  = nxt(lambda yy: FloatSlider(x, yy, w,
                                                 "Fault probability / step",
                                                 0.0, 0.01, 0.0, step=0.0005))
        self.sliders = [self.agents, self.vision, self.comm, self.dens,
                        self.steps, self.eps, self.fault]

        self.nav = Toggle(x, y, "Auto-nav: BFS to target once known (eval-only)")
        y += 36
        self.label_y = y + 2
        self.field = TextField(x, self.label_y + 18, w - 110, 30,
                               "checkpoints/mas_50k_dr_faults_ep50000.pt")
        self.browse = pygame.Rect(self.field.rect.right + 8,
                                  self.field.rect.y, 102, 30)
        self.start = pygame.Rect(x, cs.H - 56, w, 38)

        self.surf = pygame.Surface(self.size)
        self.f    = theme.font(14)
        self.f_b  = theme.font(19, bold=True)
        self.f_s  = theme.font(12)

    def handle_px(self, slider):
        """Centre of a slider's handle, in screen coordinates."""
        r = slider._handle_rect()
        return (r.centerx, r.centery)

    def draw(self, mouse=None, start_lit=0.0):
        """Redraw exactly as ConfigScreen.run does, with a faked pointer."""
        theme.set_mouse_override(mouse)
        try:
            s = self.surf
            s.fill(theme.BG)
            s.blit(self.f_b.render("Simulation setup", True, theme.TEXT),
                   (self.pad, 26))
            for sl in self.sliders:
                sl.draw(s, self.f)
            self.nav.draw(s, self.f)
            s.blit(self.f.render("Network weights (.pt)", True,
                                 theme.COLOR_LABEL), (self.pad, self.label_y))
            self.field.draw(s, self.f)
            theme.draw_button(s, self.browse, "Browse...", self.f)
            s.blit(self.f_s.render(
                "Spawn is forced to the top-left corner (0,0).",
                True, theme.COLOR_LABEL), (self.pad, self.start.y - 26))
            theme.draw_button(s, self.start, "Start simulation", self.f_b,
                              active=True, radius=8)
            if start_lit > 0:
                glow = pygame.Surface(self.start.size, pygame.SRCALPHA)
                glow.fill(theme.with_alpha((255, 255, 255),
                                           int(90 * start_lit)))
                s.blit(glow, self.start.topleft)
            return s
        finally:
            theme.set_mouse_override(None)


# ---------------------------------------------------------------------------
# Scripted cursor moves
# ---------------------------------------------------------------------------

class Move:
    """Drag one slider to a value, then dwell on it."""

    def __init__(self, slider, target, label, travel=0.30, drag=0.42):
        self.slider, self.target, self.label = slider, target, label
        self.travel, self.drag = travel, drag


def _drive(screen, moves, t):
    """Where the cursor is, and what the sliders read, at scene progress `t`."""
    n = len(moves)
    span = 1.0 / n
    k = min(n - 1, int(t / span))
    lt = (t - k * span) / span
    mv = moves[k]

    # Everything before this move is already at its target.
    for done in moves[:k]:
        _set(done.slider, done.target)

    start = _handle_of(screen, moves[k - 1].slider) if k else (screen.cs.W - 90, 44)
    end0  = _handle_of(screen, mv.slider)

    if lt < mv.travel:                       # fly to the handle
        p = typo.smooth(lt / mv.travel)
        return typo.lerp_pt(start, end0, p), mv, 0.0
    if lt < mv.travel + mv.drag:             # drag it
        p = typo.smooth((lt - mv.travel) / mv.drag)
        _set(mv.slider, _blend(mv.slider, mv.target, p))
        return _handle_of(screen, mv.slider), mv, p
    _set(mv.slider, mv.target)               # dwell
    return _handle_of(screen, mv.slider), mv, 1.0


def _handle_of(screen, slider):
    return screen.handle_px(slider)


def _set(slider, value):
    if isinstance(slider, FloatSlider):
        slider.value = int(round((value - slider.flo) / slider.step))
    elif isinstance(slider, Toggle):
        slider.value = bool(value)
    else:
        slider.value = int(round(value))


def _blend(slider, target, p):
    if isinstance(slider, FloatSlider):
        cur = slider.fvalue
    else:
        cur = slider.value
    return cur + (target - cur) * p


# ---------------------------------------------------------------------------

def render(ctx):
    import yaml
    with open(C.CONFIG) as f:
        cfg = yaml.safe_load(f)

    screen = Screen(cfg)
    canvas = Canvas()

    b10, b11 = ctx.beat("b10"), ctx.beat("b11")
    n10, n11 = ctx.frames(b10), ctx.frames(b11)

    moves = [
        Move(screen.agents, 4,      "four drones"),
        Move(screen.vision, 3,      "vision radius 3"),
        Move(screen.comm,   5,      "comm range 5"),
        Move(screen.dens,   0.20,   "20% obstacles"),
        Move(screen.fault,  0.003,  "0.3% failure per step"),
    ]
    _set(screen.agents, 1)
    _set(screen.vision, 1)
    _set(screen.comm, 1)
    _set(screen.dens, screen.cs.d_lo)
    _set(screen.fault, 0.0)

    # -- b10: the sliders ---------------------------------------------------
    for k in range(n10):
        t = k / float(n10)
        mouse, mv, drag = _drive(screen, moves, typo.clamp(t / 0.94))
        src = screen.draw(mouse=mouse)
        _compose(canvas, src, k, mouse, drag and mv.label)
        typo.lower_third(
            canvas.surf, "The whole envelope",
            "Slider ranges come straight from the training distribution, so "
            "the screen can only ask for a world the policy actually learned "
            "in.",
            t, pos=(TEXT_X, 520), width=740)
        yield canvas.surf

    # -- b11: the point of it all ------------------------------------------
    for k in range(n11):
        t = k / float(n11)
        lit = typo.smooth(typo.clamp((t - 0.55) / 0.25))
        src = screen.draw(mouse=(screen.start.centerx, screen.start.centery - 4)
                          if t > 0.5 else None, start_lit=lit)
        _compose(canvas, src, n10 + k,
                 (screen.start.centerx, screen.start.centery - 4)
                 if t > 0.5 else None, None, click=lit)
        typo.lower_third(
            canvas.surf, "One policy, every configuration",
            "Team size, sensing, communication, clutter and failure rate all "
            "change. The 22-million-parameter network behind them does not.",
            t, pos=(TEXT_X, 520), width=740)
        yield canvas.surf


def _compose(canvas, src, frame_idx, mouse, label, click=0.0):
    """Place the setup screen, enlarged, on the film canvas."""
    w, h = src.get_size()
    pw, ph = int(w * PANEL_SCALE), int(h * PANEL_SCALE)
    big = pygame.transform.smoothscale(src, (pw, ph))
    x, y = C.W - pw - PANEL_RIGHT, (C.H - ph) // 2

    canvas.backdrop()
    shadow = pygame.Surface((pw + 40, ph + 40), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 130), shadow.get_rect(), border_radius=16)
    canvas.surf.blit(shadow, (x - 20, y - 12))
    canvas.surf.blit(big, (x, y))
    pygame.draw.rect(canvas.surf, theme.BORDER, pygame.Rect(x, y, pw, ph), 1)
    canvas.vignette(0.42)

    typo.chapter(canvas.surf, 1, "Setup", (frame_idx / float(C.FPS)) / 5.0)

    if mouse is not None:
        mx = x + mouse[0] * PANEL_SCALE
        my = y + mouse[1] * PANEL_SCALE
        if label:
            f = theme.font(19, bold=True)
            g = f.render(label, True, theme.ACCENT)
            bg = pygame.Surface((g.get_width() + 20, g.get_height() + 12),
                                pygame.SRCALPHA)
            pygame.draw.rect(bg, theme.with_alpha((12, 13, 18), 225),
                             bg.get_rect(), border_radius=5)
            pygame.draw.rect(bg, theme.with_alpha(theme.ACCENT, 160),
                             bg.get_rect(), 1, border_radius=5)
            canvas.surf.blit(bg, (mx + 22, my + 14))
            canvas.surf.blit(g, (mx + 32, my + 20))
        typo.cursor(canvas.surf, (mx, my), click=click)

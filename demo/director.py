"""Shot sequencing for the scenes shot on live GUI footage.

A scene declares a list of ``Shot``s, one per narration beat, and the director
turns them into frames: it eases the camera between shots, keeps the chapter
tag and lower thirds on their own timing, and steps the episode at whatever
pace the shot asks for.

Scenes stay declarative; nothing here knows what any particular scene is about.
"""

from dataclasses import dataclass, field

import pygame

from gui import theme
from demo import config as C
from demo import typo
from demo.compositor import Canvas, Cam, frame_rect


TRANSITION_S = 1.15         # seconds for the camera to settle on a new shot
CHAPTER_S    = 4.2          # how long the chapter tag stays up


@dataclass
class Shot:
    """One beat's worth of picture."""
    beat:    str                       # narration beat id
    look:    object = "all"            # region name, Cam, or callable(sim)->Cam
    rps:     float = None              # env rounds per second (0 freezes)
    lower:   tuple = None              # (headline, body) lower third
    accent:  tuple = None              # lower-third accent colour
    pad:     float = 1.12              # framing looseness
    on_round: object = None            # callback(RoundInfo)
    overlay: object = None             # callback(canvas, t, sim, k, n)
    hold_cam: bool = False             # keep the previous shot's camera


def resolve_cam(look, sim, pad=1.12):
    """A region name, an explicit Cam, or a callable that returns one."""
    if look is None or look == "all":
        return Cam()
    if isinstance(look, Cam):
        return look
    if callable(look):
        return look(sim)
    rect = sim.regions()[look]
    return frame_rect(rect, sim.size, pad=pad)


def play(ctx, sim, shots, *, chapter=None, on_frame=None):
    """Render every shot in order, yielding finished canvas surfaces.

    `chapter` is (index, label) for the tag in the top-left gutter.
    `on_frame(canvas, scene_t, sim)` runs last on every frame, for scene-wide
    chrome.
    """
    canvas   = Canvas()
    cam_prev = Cam()
    total    = sum(ctx.frames(ctx.beat(s.beat)) for s in shots)
    elapsed  = 0

    for shot in shots:
        beat = ctx.beat(shot.beat)
        n    = ctx.frames(beat)
        cam_to = cam_prev if shot.hold_cam else resolve_cam(shot.look, sim,
                                                            shot.pad)

        for k in range(n):
            t   = k / float(n)
            src = sim.tick(shot.rps, on_round=shot.on_round)

            # Ease onto the new framing over the first TRANSITION_S.
            ct  = min(1.0, (k / float(C.FPS)) / TRANSITION_S)
            cam = cam_prev.lerp(cam_to, ct)

            canvas.backdrop()
            canvas.place(src, cam)
            canvas.vignette(0.45)

            scene_t = (elapsed + k) / float(C.FPS)
            if chapter is not None:
                typo.chapter(canvas.surf, chapter[0], chapter[1],
                             scene_t / CHAPTER_S)
            if shot.lower:
                typo.lower_third(canvas.surf, shot.lower[0], shot.lower[1], t,
                                 accent=shot.accent)
            if shot.overlay:
                shot.overlay(canvas, t, sim, k, n)
            if on_frame:
                on_frame(canvas, scene_t, sim)

            yield canvas.surf

        cam_prev = cam_to
        elapsed += n


# ---------------------------------------------------------------------------
# Overlay helpers scenes can hand to Shot.overlay
# ---------------------------------------------------------------------------

def track_agent(idx, text, *, title=None, side="right", color=None,
                window=(0.18, 0.92), dist=140):
    """A callout that follows drone `idx` around the map."""
    def _draw(canvas, t, sim, k, n):
        lo, hi = window
        if not (lo <= t <= hi):
            return
        col = color or theme.AGENT_COLORS[idx % len(theme.AGENT_COLORS)]
        anchor = canvas.to_canvas(sim.agent_px(idx))
        typo.callout(canvas.surf, (int(anchor[0]), int(anchor[1])), text,
                     (t - lo) / max(1e-6, hi - lo), side=side, color=col,
                     title=title, dist=dist)
    return _draw


def highlight_region(name, *, amount=0.55, window=(0.2, 0.95), label=None):
    """Dim everything except one panel of the GUI."""
    def _draw(canvas, t, sim, k, n):
        lo, hi = window
        if not (lo <= t <= hi):
            return
        a = typo.inout((t - lo) / max(1e-6, hi - lo), 0.14, 0.14) * amount
        keep = canvas.rect_to_canvas(sim.regions()[name])
        keep = keep.inflate(18, 18).clip(pygame.Rect(0, 0, C.W, C.H))
        canvas.spotlight(keep, a, ring=a > amount * 0.6)
        if label and a > amount * 0.5:
            f = theme.font(17, bold=True)
            g = f.render(label.upper(), True, theme.ACCENT)
            g.set_alpha(int(255 * a / max(amount, 1e-6)))
            canvas.surf.blit(g, (keep.x, max(8, keep.y - 26)))
    return _draw


def compose(*overlays):
    """Run several overlays in order on the same frame."""
    def _draw(canvas, t, sim, k, n):
        for o in overlays:
            if o:
                o(canvas, t, sim, k, n)
    return _draw


def when(window, overlay):
    """Restrict an overlay to a slice of its shot, re-normalising its own t."""
    lo, hi = window

    def _draw(canvas, t, sim, k, n):
        if lo <= t <= hi:
            overlay(canvas, (t - lo) / max(1e-6, hi - lo), sim, k, n)
    return _draw

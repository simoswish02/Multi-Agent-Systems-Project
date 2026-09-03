"""Scene 5 -- one policy, four worlds.

Four missions run side by side, stepped in lockstep, driven by the same
checkpoint: a lone drone, a team of six the policy never trained on, heavy
clutter, and a failure rate more than three times the worst it ever saw.

The labels state where each world sits relative to the training distribution,
and those bounds are read from ``domain_randomization`` rather than typed in,
so a configuration cannot be called out-of-distribution unless it is.
"""

import os

import pygame

from gui import theme
from demo import config as C
from demo import typo
from demo.capture import SimSpec, Sim
from demo.compositor import Canvas, Cam


PANEL = 380
PANEL_Y = 286
PANEL_X0 = 130
PANEL_GAP = 50
CELL = 12                    # small maps: four of them share the frame


def _worlds(dr):
    """The four configurations, each tagged in- or out-of-distribution."""
    a_max = int(dr.get("n_agents", [1, 4])[1])
    d_max = float(dr.get("obstacle_density", [0.10, 0.30])[1])
    f_max = float(dr.get("fault_prob", [0.0, 0.003])[1])
    return [
        (SimSpec(n_agents=1, vision=3, comm=5, density=0.18,
                 fault_prob=0.0, max_steps=200, seed=11),
         "One drone", "alone, no one to talk to", False),
        (SimSpec(n_agents=a_max + 2, vision=3, comm=5, density=0.20,
                 fault_prob=0.0, max_steps=200, seed=12),
         "%d drones" % (a_max + 2),
         "a team size it never trained on", True),
        (SimSpec(n_agents=4, vision=2, comm=4, density=d_max,
                 fault_prob=0.0, max_steps=200, seed=13),
         "Heavy clutter", "%.0f%% obstacles, short sight" % (d_max * 100),
         False),
        (SimSpec(n_agents=4, vision=3, comm=5, density=0.20,
                 fault_prob=0.01, max_steps=200, seed=14),
         "Failing fast", "%.1fx the worst rate it trained on" % (0.01 / f_max),
         True),
    ]


def render(ctx):
    dr = ctx.policy.config.get("domain_randomization", {})
    worlds = _worlds(dr)

    sims = []
    for spec, _, _, _ in worlds:
        s = Sim(spec, ctx.policy, cell=CELL).reset()
        s.frame(1.0)
        sims.append(s)

    canvas = Canvas()
    b50, b51 = ctx.beat("b50"), ctx.beat("b51")
    n50, n51 = ctx.frames(b50), ctx.frames(b51)
    total = n50 + n51

    for k in range(total):
        t = k / float(total)
        canvas.backdrop()
        canvas.vignette(0.5)

        for j, (sim, (_, title, note, ood)) in enumerate(zip(sims, worlds)):
            # Panels arrive one after another across the first beat.
            appear = typo.clamp((k - j * (n50 * 0.16)) / max(1.0, n50 * 0.22))
            if appear <= 0.01:
                continue
            src = sim.tick(3.0)
            _panel(canvas, j, src, sim, title, note, ood, appear)

        typo.chapter(canvas.surf, 5, "Generalization", (k / float(C.FPS)) / 5.0)

        if k < n50:
            typo.lower_third(
                canvas.surf, "Four worlds at once",
                "Team size, sensing, clutter and failure rate all differ. Two "
                "of these are outside the ranges the policy was trained on.",
                k / float(n50), pos=(64, C.H - 148), width=760)
        else:
            _same_weights(canvas, (k - n50) / float(n51))
        yield canvas.surf

    for s in sims:
        s.close()


# ---------------------------------------------------------------------------

def _panel(canvas, j, src, sim, title, note, ood, appear):
    x = PANEL_X0 + j * (PANEL + PANEL_GAP)
    e = typo.ease_out(appear)
    size = max(4, int(PANEL * (0.86 + 0.14 * e)))
    rect = pygame.Rect(x + (PANEL - size) // 2,
                       PANEL_Y + (PANEL - size) // 2, size, size)

    # Frame only the map: the sidebar has no place in a four-up.
    grid = sim.grid_px
    cam = Cam(grid / 2.0, sim.renderer.TOOLBAR_H + grid / 2.0,
              sim.size[0] / float(grid))
    col = theme.WARNING if ood else theme.BORDER

    layer = pygame.Surface(canvas.size, pygame.SRCALPHA)
    tmp = Canvas(canvas.size)
    tmp.surf = layer
    tmp.place_in(rect, src, cam, border=True, label_color=col)
    typo.blit_alpha(canvas.surf, layer, (0, 0), int(255 * e))

    if e < 0.5:
        return
    a = int(255 * (e - 0.5) / 0.5)
    f_t = theme.font(23, bold=True)
    f_n = theme.font(16)
    f_b = theme.font(13, bold=True)
    ty = rect.bottom + 16
    typo.blit_alpha(canvas.surf, f_t.render(title, True, theme.TEXT),
                    (rect.x, ty), a)
    typo.blit_alpha(canvas.surf, f_n.render(note, True, theme.TEXT_DIM),
                    (rect.x, ty + 30), a)

    tag = "OUTSIDE TRAINING" if ood else "IN DISTRIBUTION"
    tcol = theme.WARNING if ood else theme.TEXT_DIM
    g = f_b.render(tag, True, tcol)
    box = pygame.Rect(rect.x, ty + 56, g.get_width() + 16, 22)
    pygame.draw.rect(canvas.surf, tcol, box, 1, border_radius=4)
    typo.blit_alpha(canvas.surf, g, (box.x + 8, box.y + 4), a)


def _same_weights(canvas, t):
    """The punchline: one file drives all four."""
    a = typo.inout(t, 0.12, 0.10)
    if a <= 0.02:
        return
    alpha = int(255 * a)
    f_b = theme.font(46, bold=True)
    f_m = theme.font(20, bold=True, mono=True)

    text = "Same weights. Every time."
    g = f_b.render(text, True, theme.TEXT)
    x = (C.W - g.get_width()) // 2
    y = C.H - 190
    typo.blit_alpha(canvas.surf, g, (x, y), alpha)

    name = os.path.basename(C.WEIGHTS)
    gm = f_m.render(name, True, theme.ACCENT)
    typo.blit_alpha(canvas.surf, gm, ((C.W - gm.get_width()) // 2, y + 62),
                    int(alpha * 0.95))

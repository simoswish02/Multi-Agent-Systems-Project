"""Scene 6 -- results and sign-off.

Every number on screen is read out of ``testing_results/csv/`` at build time,
so the film can never quote a figure the experiments no longer support. If a
CSV is missing the build stops rather than invent one.
"""

import csv
import os

import pygame

from gui import theme
from demo import config as C
from demo import typo
from demo.capture import SimSpec, Sim
from demo.compositor import Canvas, Cam


# ---------------------------------------------------------------------------
# Numbers, straight from the experiment CSVs
# ---------------------------------------------------------------------------

def _agg(axis):
    path = os.path.join(C.CSV_DIR, "agg_%s.csv" % axis)
    if not os.path.exists(path):
        raise SystemExit(
            "demo/scenes/s6_outro.py needs %s. Run the evaluation sweep "
            "(testing/evaluate_policy.py --axis %s) first." % (path, axis))
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(rows, value):
    for r in rows:
        if abs(float(r["value"]) - value) < 1e-9:
            return r
    raise SystemExit("no row for value=%r in that sweep" % value)


def facts():
    """The three claims the outro makes, with their supporting rows."""
    n_rows = _agg("n_agents")
    f_rows = _agg("fault")

    six    = float(_row(n_rows, 6)["success_rate"])
    f_zero = float(_row(f_rows, 0.0)["success_rate"])
    f_max  = float(_row(f_rows, 0.003)["success_rate"])
    f_ood  = float(_row(f_rows, 0.01)["success_rate"])
    broken = float(_row(f_rows, 0.003)["n_broken"])

    return [
        ("%.0f%%" % (six * 100),
         "Six drones",
         "A team size it never trained on"),
        ("%.1f pts" % (-(f_zero - f_max) * 100),
         "Across the trained fault range",
         "%.1f%% to %.1f%% success" % (f_zero * 100, f_max * 100)),
        ("%.0f%%" % (f_ood * 100),
         "At triple that failure rate",
         "Well outside training, and still no cliff"),
    ]


CARD = (420, 232)


# ---------------------------------------------------------------------------

def render(ctx):
    """A slow, heavily dimmed mission runs behind the numbers."""
    spec = SimSpec(n_agents=4, vision=3, comm=5, density=0.22,
                   fault_prob=0.003, max_steps=200, seed=7)
    sim = Sim(spec, ctx.policy).reset()
    sim.frame(1.0)

    cards = facts()
    canvas = Canvas()

    b60, b61 = ctx.beat("b60"), ctx.beat("b61")
    n60, n61 = ctx.frames(b60), ctx.frames(b61)

    # -- b60: the first two cards -----------------------------------------
    for k in range(n60):
        t = k / float(n60)
        _bed(canvas, sim)
        typo.chapter(canvas.surf, 6, "Results", (k / float(C.FPS)) / 5.0)
        typo.stat_cards(canvas.surf, cards[:2], typo.clamp(t / 0.6),
                        y=C.H // 2 - 160, card=CARD)
        yield canvas.surf

    # -- b61: the third card, then the sign-off ---------------------------
    for k in range(n61):
        t = k / float(n61)
        _bed(canvas, sim)

        # The first two cards stay up, then everything gives way to the title.
        fade = typo.clamp((t - 0.44) / 0.14)
        if fade < 1.0:
            typo.stat_cards(canvas.surf, cards[:2], 1.0, y=C.H // 2 - 160,
                            card=CARD, alpha=1.0 - fade)
            typo.stat_cards(canvas.surf, cards[2:], typo.clamp(t / 0.26),
                            y=C.H // 2 + 90, card=CARD, alpha=1.0 - fade)
        if fade > 0.0:
            canvas.scrim(int(235 * fade))
            typo.title_card(
                canvas.surf,
                ["Multi-Drone Cooperative Search", "with Random Faults"],
                "Simone Rimondi  ·  Multi-Agent Systems  ·  "
                "University of Bologna  ·  2025/2026",
                typo.clamp((t - 0.50) / 0.44),
                kicker="one policy, any team, any failure rate")
        yield canvas.surf

    sim.close()


def _bed(canvas, sim):
    """The dimmed mission the cards sit on."""
    src = sim.tick(1.1)
    canvas.backdrop()
    canvas.place(src, Cam(), shadow=False, border=False)
    canvas.scrim(168)
    canvas.vignette(0.6)

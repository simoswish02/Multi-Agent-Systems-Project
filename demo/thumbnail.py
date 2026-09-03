"""YouTube thumbnails, generated from the real mission.

1280x720, drawn with the film's own palette and type so the thumbnail and the
video look like the same thing. The map, the wreck and the channel planes are
all rendered from a live episode -- nothing here is a mock-up.

    python -m demo.thumbnail            # all variants -> demo/out/thumbs/
    python -m demo.thumbnail --variant b

A thumbnail has to survive being shown 200 px wide in a sidebar, so each
variant keeps to one focal image, at most four words of headline, and a single
accent colour.
"""

import argparse
import os

import pygame

from gui import theme
from demo import config as C
from demo import seeds as S
from demo import typo
from demo.tensorviz import planes as P
from demo.tensorviz import verify as V

W, H = 1280, 720
OUT = os.path.join(C.OUT, "thumbs")


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def _map_crop(sim, centre, span, size, shift=(0.0, 0.0)):
    """A region of the live map, scaled up for the thumbnail.

    `shift` moves the framing in grid cells, so the subject can be pushed out
    of the left third where the headline goes.
    """
    src = sim.frame(1.0)
    cs = sim.renderer.CELL
    grid = sim.grid_px
    aspect = size[1] / float(size[0])

    # A widely spread team can ask for more than the map holds, so the window
    # is clamped to fit on both axes before it is positioned.
    span = min(span, grid, grid / aspect)
    half_w, half_h = span / 2.0, span * aspect / 2.0

    cx = (centre[1] + shift[1]) * cs + cs / 2.0
    cy = (centre[0] + shift[0]) * cs + cs / 2.0 + sim.renderer.TOOLBAR_H
    x = min(max(cx - half_w, 0.0), max(0.0, grid - 2 * half_w))
    y = min(max(cy - half_h, float(sim.renderer.TOOLBAR_H)),
            sim.renderer.TOOLBAR_H + max(0.0, grid - 2 * half_h))
    rect = pygame.Rect(int(x), int(y), int(2 * half_w), int(2 * half_h))
    return pygame.transform.smoothscale(src.subsurface(rect), size)


def _frame_agents(sim, pad=7.0, min_span=280.0):
    """Centre and span (in pixels) that actually contain the live drones.

    A thumbnail of an empty grid says nothing, so the crop is derived from
    where the team is rather than from a fixed point.
    """
    cs = sim.renderer.CELL
    live = [sim.env.agent_pos[i] for i in range(sim.env.n_agents)
            if sim.env.agent_alive[i]]
    if not live:
        return (16, 16), min_span
    rs = [p[0] for p in live]
    cls = [p[1] for p in live]
    centre = ((min(rs) + max(rs)) / 2.0, (min(cls) + max(cls)) / 2.0)
    extent = max(max(rs) - min(rs), max(cls) - min(cls)) + pad
    return centre, max(min_span, extent * cs)


def _scrim(surf, rect, alpha_left=232, alpha_right=0):
    """Horizontal gradient wash, so type on one side always has contrast."""
    grad = pygame.Surface(rect.size, pygame.SRCALPHA)
    steps = 64
    w = max(1, rect.width // steps)
    for k in range(steps):
        a = int(alpha_left + (alpha_right - alpha_left) * (k / float(steps - 1)))
        grad.fill((9, 10, 14, max(0, a)), pygame.Rect(k * w, 0, w + 1, rect.height))
    surf.blit(grad, rect.topleft)


def _tag(surf, text, pos, color=None):
    f = theme.font(19, bold=True)
    typo.tracked(surf, text.upper(), f, color or theme.ACCENT, pos, tracking=4)


def _headline(surf, lines, pos, *, size=92, colors=None, lead=8):
    f = theme.font(size, bold=True)
    x, y = pos
    for k, ln in enumerate(lines):
        col = (colors or {}).get(k, theme.TEXT)
        g = f.render(ln, True, col)
        # A soft drop shadow keeps the type readable over any footage.
        sh = f.render(ln, True, (6, 7, 10))
        surf.blit(sh, (x + 3, y + 3))
        surf.blit(g, (x, y))
        y += f.get_height() + lead
    return y


def _rule(surf, pos, width=110, color=None):
    pygame.draw.rect(surf, color or theme.ACCENT,
                     pygame.Rect(pos[0], pos[1], width, 5))


def _sub(surf, text, pos, *, size=26, color=None):
    f = theme.font(size)
    g = f.render(text, True, color or theme.TEXT_DIM)
    sh = f.render(text, True, (6, 7, 10))
    surf.blit(sh, (pos[0] + 2, pos[1] + 2))
    surf.blit(g, pos)


def _canvas():
    s = pygame.Surface((W, H))
    s.fill((10, 11, 15))
    return s


def _finish(surf):
    surf.blit(typo.vignette((W, H), 0.55), (0, 0))
    return surf


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def variant_a(policy):
    """One policy, any swarm -- the map, with the channel planes alongside."""
    facts = S.hero_facts()
    brk = facts.get("breaks", [[0, 56]])[0][1]
    # Early, while the team is still leaving the spawn corner together: this
    # variant sells the healthy swarm, so the drones have to be both alive and
    # close enough to each other to read as a group.
    sim = S.hero_sim(policy, upto_round=18)

    surf = _canvas()
    live = [i for i in range(sim.env.n_agents) if sim.env.agent_alive[i]]
    centre, span = _frame_agents(sim)
    # Push the team into the right two thirds; the headline owns the left.
    surf.blit(_map_crop(sim, centre, span, (W, H), shift=(0, -2.0)), (0, 0))
    _scrim(surf, pygame.Rect(0, 0, W, H), 246, 30)

    # Two channels of that drone's belief, tucked into the top right.
    i = live[0] if live else 0
    g = V.reshape(sim.obs[i]["global"], 6, sim.env.grid_size)
    for k, ch in enumerate((0, 2)):
        card = P.plane_card(g[ch], P.GLOBAL_CHANNELS[ch][1], 4).copy()
        card.set_alpha(226)
        surf.blit(card, (W - 190 - k * 26, 40 + k * 150))

    _tag(surf, "multi-agent reinforcement learning", (58, 62))
    y = _headline(surf, ["ONE POLICY.", "ANY SWARM."], (54, 336), size=96,
                  colors={1: theme.ACCENT})
    _rule(surf, (58, y + 16))
    _sub(surf, "32x32 grid  ·  partial observability  ·  random faults",
         (58, y + 40))
    sim.close()
    return _finish(surf)


def variant_b(policy):
    """The failure: a wreck, close, with the beacon still going."""
    facts = S.hero_facts()
    victim, brk = facts.get("breaks", [[0, 56]])[0]
    sim = S.hero_sim(policy, upto_round=brk + 6)

    surf = _canvas()
    crash = sim.env.crash_pos.get(victim, sim.env.agent_pos[victim])
    # The wreck sits in the right third, clear of the headline.
    surf.blit(_map_crop(sim, crash, 300, (W, H), shift=(-1.0, -4.0)), (0, 0))
    _scrim(surf, pygame.Rect(0, 0, W, H), 242, 24)

    _tag(surf, "no oracle  ·  no coordinator", (58, 62), theme.DANGER)
    y = _headline(surf, ["WHEN A DRONE", "JUST STOPS"], (54, 330), size=94,
                  colors={1: theme.DANGER})
    _rule(surf, (58, y + 16), color=theme.DANGER)
    _sub(surf, "The others are not told. They find out.", (58, y + 40))
    sim.close()
    return _finish(surf)


def variant_c(policy):
    """What the network actually sees: the six belief planes."""
    facts = S.hero_facts()
    brk = facts.get("breaks", [[0, 56]])[0][1]
    sim = S.hero_sim(policy, upto_round=brk + 34)

    surf = _canvas()
    live = [i for i in range(sim.env.n_agents) if sim.env.agent_alive[i]]
    i = live[0] if live else 0
    g = V.reshape(sim.obs[i]["global"], 6, sim.env.grid_size)

    grid = P.Grid(P.GLOBAL_CHANNELS, cols=3, cell=6, gap_x=22, gap_y=22)
    gw, gh = grid.size(6)
    grid.draw(surf, g, (W - gw - 26, (H - gh) // 2 - 6), 1.0, alpha=250,
              labels=False)
    # The wash has to clear the headline's full width, or the planes read
    # through the letters.
    _scrim(surf, pygame.Rect(0, 0, 700, H), 252, 96)

    _tag(surf, "what one drone actually sees", (58, 66), theme.INFO)
    y = _headline(surf, ["INSIDE", "THE POLICY"], (54, 300), size=88,
                  colors={1: theme.INFO})
    _rule(surf, (58, y + 16), color=theme.INFO)
    _sub(surf, "6 x 32 x 32 of belief, not the world.", (58, y + 40))
    sim.close()
    return _finish(surf)


VARIANTS = {"a": variant_a, "b": variant_b, "c": variant_c}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), nargs="*",
                    default=sorted(VARIANTS))
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    os.makedirs(OUT, exist_ok=True)

    from demo.capture import make_policy
    policy = make_policy()

    for key in args.variant:
        surf = VARIANTS[key](policy)
        path = os.path.join(OUT, "thumb_%s.png" % key)
        pygame.image.save(surf, path)
        print("wrote %s  (%dx%d)" % (path, W, H))


if __name__ == "__main__":
    main()

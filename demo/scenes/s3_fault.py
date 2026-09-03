"""Scene 3 -- a drone fails.

Picks up the hero mission exactly where scene 2 left it, a few rounds short of
the first fault, and slows almost to a stop for it. Nothing about the failure
is staged: the seed decides which drone breaks and when, and the news reaches
the others only as fast as the beacon and their own movement allow.

The pace of the opening shot is computed so the fault lands on the cue rather
than whenever it happens to come round.
"""

import pygame

from gui import theme
from demo import config as C
from demo import director as D
from demo import seeds as S
from demo import typo
from demo.compositor import frame_rect
from demo.scenes.s2_search import rounds_covered


def render(ctx):
    facts = S.hero_facts()
    victim, brk = facts.get("breaks", [[0, 60]])[0]

    sim = S.hero_sim(ctx.policy, upto_round=rounds_covered())
    start_round = rounds_covered()

    # b30 must deliver the team to the fault exactly as the words run out.
    b30 = ctx.beat("b30")
    lead_rps = max(0.35, (brk - start_round) / max(1.0, b30["duration"]))

    # The overlays read the environment directly every frame, so the round
    # callback only has to exist for `Shot.on_round`.
    def watch(info):
        pass

    def look_at_victim(s):
        r, c = s.env.crash_pos.get(victim, s.env.agent_pos[victim])
        return frame_rect(s.cell_rect(r, c, pad=150), s.size, pad=1.0,
                          max_zoom=2.6)

    shots = [
        D.Shot(
            "b30", look=lambda s: frame_rect(
                s.cell_rect(*s.env.agent_pos[victim], pad=210), s.size,
                pad=1.0, max_zoom=1.9),
            rps=lead_rps, on_round=watch,
            overlay=D.track_agent(victim, "still flying", side="right",
                                  window=(0.30, 0.98)),
        ),
        D.Shot(
            "b31", look=look_at_victim, rps=0.55, on_round=watch,
            accent=theme.DANGER,
            lower=("D%d is gone" % victim,
                   "It will not move, sense or transmit again. Its map is "
                   "frozen at the moment it failed, and its turn still comes "
                   "round every round."),
            overlay=_wreck_marks(victim),
        ),
        D.Shot(
            "b32", look="map", rps=1.0, on_round=watch, pad=1.02,
            accent=theme.WARNING,
            lower=("No oracle",
                   "Nobody is told. The wreck emits a distress beacon, and "
                   "each teammate learns of it only by coming close enough, "
                   "or by hearing it from another."),
            overlay=_knower_counter(victim),
        ),
        D.Shot(
            "b33", look="all", rps=2.4, on_round=watch,
            lower=("The team absorbs it",
                   "Coverage keeps climbing with one drone fewer. Losing a "
                   "drone costs time as well as sensors -- the wreck still "
                   "burns a turn in every round."),
            overlay=_alive_badge(),
        ),
    ]

    for surf in D.play(ctx, sim, shots, chapter=(3, "Failure")):
        yield surf

    sim.close()


# ---------------------------------------------------------------------------

def _knowers(sim, victim):
    """How many still-flying drones know that `victim` crashed."""
    env = sim.env
    return sum(1 for i in range(env.n_agents)
               if env.agent_alive[i] and victim in env.agent_known_crashed[i])


def _live_others(sim, victim):
    env = sim.env
    return [i for i in range(env.n_agents)
            if env.agent_alive[i] and i != victim]


def _wreck_marks(victim):
    """Label the wreck itself once it exists."""
    def _draw(canvas, t, sim, k, n):
        if sim.env.agent_alive[victim]:
            return
        r, c = sim.env.crash_pos.get(victim, sim.env.agent_pos[victim])
        cs = sim.renderer.CELL
        px = c * cs + cs / 2.0
        py = r * cs + sim.renderer.TOOLBAR_H + cs / 2.0
        x, y = canvas.to_canvas((px, py))
        typo.callout(canvas.surf, (int(x), int(y)),
                     "crash site -- a wreck, and a beacon",
                     typo.span(t, 0.18), side="right", color=theme.BROKEN_X,
                     title="D%d" % victim, dist=150)
    return _draw


def _knower_counter(victim):
    """A running count of who has learned about the crash."""
    def _draw(canvas, t, sim, k, n):
        a = typo.inout(t, 0.12, 0.10)
        if a <= 0.02:
            return
        others = _live_others(sim, victim)
        known = [i for i in others if victim in sim.env.agent_known_crashed[i]]

        x, y = 330, 132          # over the map, clear of the sidebar
        f_h = theme.font(17, bold=True)
        f_b = theme.font(44, bold=True, mono=True)
        f_s = theme.font(16)
        panel = pygame.Surface((390, 168), pygame.SRCALPHA)
        pygame.draw.rect(panel, theme.with_alpha((12, 13, 18), 232),
                         panel.get_rect(), border_radius=7)
        pygame.draw.rect(panel, theme.with_alpha(theme.WARNING, 190),
                         panel.get_rect(), 1, border_radius=7)
        typo.blit_alpha(canvas.surf, panel, (x, y), int(255 * a))

        typo.blit_alpha(canvas.surf,
                        f_h.render("WHO KNOWS D%d IS DOWN" % victim, True,
                                   theme.WARNING), (x + 20, y + 16),
                        int(255 * a))
        typo.blit_alpha(canvas.surf,
                        f_b.render("%d of %d" % (len(known), len(others)),
                                   True, theme.TEXT), (x + 20, y + 44),
                        int(255 * a))
        # One chip per surviving teammate, filled once it has heard.
        for j, i in enumerate(others):
            chip = pygame.Rect(x + 20 + j * 46, y + 112, 38, 34)
            col = theme.AGENT_COLORS[i % len(theme.AGENT_COLORS)]
            hot = i in known
            pygame.draw.rect(canvas.surf, col if hot else theme.SURFACE, chip,
                             0 if hot else 1, border_radius=5)
            g = f_s.render("D%d" % i, True,
                           (16, 18, 24) if hot else theme.TEXT_DIM)
            canvas.surf.blit(g, (chip.centerx - g.get_width() // 2,
                                 chip.centery - g.get_height() // 2))
    return _draw


def _alive_badge():
    """Team strength, as the mission carries on."""
    def _draw(canvas, t, sim, k, n):
        a = typo.inout(t, 0.14, 0.12)
        if a <= 0.02:
            return
        env = sim.env
        alive = int(env.agent_alive.sum())
        x, y = 330, 132          # over the map, clear of the sidebar
        f_h = theme.font(16, bold=True)
        f_b = theme.font(48, bold=True, mono=True)
        panel = pygame.Surface((260, 128), pygame.SRCALPHA)
        pygame.draw.rect(panel, theme.with_alpha((12, 13, 18), 232),
                         panel.get_rect(), border_radius=7)
        pygame.draw.rect(panel, theme.with_alpha(theme.ACCENT, 170),
                         panel.get_rect(), 1, border_radius=7)
        typo.blit_alpha(canvas.surf, panel, (x, y), int(255 * a))
        typo.blit_alpha(canvas.surf,
                        f_h.render("STILL FLYING", True, theme.ACCENT),
                        (x + 20, y + 16), int(255 * a))
        typo.blit_alpha(canvas.surf,
                        f_b.render("%d / %d" % (alive, env.n_agents), True,
                                   theme.TEXT), (x + 20, y + 44),
                        int(255 * a))
    return _draw

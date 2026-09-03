"""Scene 2 -- cooperative search.

The opening act of the hero mission, from spawn up to just before the fault
that scene 3 opens on. Five shots: the fog, the dispersion, a map merge, the
four private beliefs, the live telemetry.

The pace is derived, not guessed: the shot list must deliver the team to the
round before the fault exactly as the narration runs out, whatever seed the
casting picked.
"""

from gui import theme
from demo import config as C
from demo import director as D
from demo import seeds as S
from demo import typo


# Scene 2 hands over to scene 3 this many rounds before the fault.
HANDOVER_LEAD = 3


def rounds_covered():
    """Rounds this scene plays: spawn up to `HANDOVER_LEAD` before the fault."""
    facts = S.hero_facts()
    brk = facts.get("breaks", [[0, 60]])[0][1]
    return max(10, brk - HANDOVER_LEAD)


def render(ctx):
    sim = S.hero_sim(ctx.policy).reset()
    sim.frame(1.0)                      # episode-start banner into the log

    seconds = sum(b["duration"] for b in ctx.beats)
    rps = rounds_covered() / max(1.0, seconds)

    # Remember where drones started, so the dispersion callouts point at the
    # two that actually went furthest apart.
    tracked = _tracked_pair(sim)

    # Catch the first comm event of the merge shot and flag it on the map.
    merge = {"pair": None, "at": None}

    def watch_comm(info):
        if merge["pair"] is None and info.comm_pairs:
            merge["pair"] = info.comm_pairs[0]
            merge["at"] = info.step

    shots = [
        D.Shot(
            "b20", look="map", rps=rps, pad=1.02,
            lower=("Team knowledge",
                   "The map is the union of what every drone has seen. "
                   "Dark cells have never been observed by anyone."),
        ),
        D.Shot(
            "b21", look="map", rps=rps, pad=1.02,
            lower=("No territory, no coordinator",
                   "Dispersion is stigmergic: revisiting a swept cell earns "
                   "less than breaking new ground, so the team splits itself."),
            overlay=D.track_agent(tracked[0], "sweeping east", side="right",
                                  window=(0.22, 0.9)),
        ),
        D.Shot(
            "b22", look="map", rps=rps, pad=1.02, accent=theme.COMM,
            lower=("Map fusion",
                   "Inside communication range, two drones take the union of "
                   "their beliefs: visited cells, walls, target and wrecks."),
            on_round=watch_comm,
            overlay=_merge_callout(merge),
        ),
        D.Shot(
            "b23", look="minimaps", rps=rps, pad=1.04, accent=theme.INFO,
            lower=("Four private beliefs",
                   "One panel per drone -- what that drone alone knows. They "
                   "are not the same map, and none of them is the world."),
        ),
        D.Shot(
            "b24", look="charts", rps=rps, pad=1.04,
            lower=("Live telemetry",
                   "Per-drone reward, cumulative return, coverage, new cells "
                   "per step and messages per step."),
        ),
    ]

    for surf in D.play(ctx, sim, shots, chapter=(2, "Cooperative search")):
        yield surf

    sim.close()


# ---------------------------------------------------------------------------

def _tracked_pair(sim):
    """Two drones far enough apart in index to have taken different corners of
    the map -- the clearest illustration of the team splitting itself up."""
    n = sim.env.n_agents
    return (1 % n, (n - 1) % n)


def _merge_callout(merge):
    """Flag the first comm link of the shot, on the link itself."""
    def _draw(canvas, t, sim, k, n):
        if merge["pair"] is None or t < 0.3:
            return
        st = typo.span(t, 0.3)
        i, j = merge["pair"][0], merge["pair"][1]
        if max(i, j) >= sim.env.n_agents:
            return
        pi, pj = sim.agent_px(i), sim.agent_px(j)
        mid = canvas.to_canvas(((pi[0] + pj[0]) / 2.0, (pi[1] + pj[1]) / 2.0))
        typo.callout(canvas.surf, (int(mid[0]), int(mid[1])),
                     "D%d and D%d merged maps" % (i, j),
                     st, side="right", color=theme.COMM,
                     title="link up", dist=120)
    return _draw

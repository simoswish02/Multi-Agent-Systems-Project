"""Scene 0 -- cold open.

The hero mission at speed: the fog burning away, comm links firing, and then
the first drone failing. The pace is set so the failure lands on the second
beat rather than wherever it happens to fall, and the title arrives on top of
it.
"""

from gui import theme
from demo import config as C
from demo import seeds as S
from demo import typo
from demo.compositor import Canvas, Cam


def render(ctx):
    facts = S.hero_facts()
    victim, brk = facts.get("breaks", [[0, 56]])[0]

    sim = S.hero_sim(ctx.policy).reset()
    sim.frame(1.0)

    canvas = Canvas()
    b00, b01 = ctx.beat("b00"), ctx.beat("b01")
    n00, n01 = ctx.frames(b00), ctx.frames(b01)

    # Reach the fault exactly as the first beat ends.
    rps_fast = max(1.0, brk / max(1.0, b00["duration"]))

    # A slow push in over the whole open.
    cam_a, cam_b = Cam(zoom=1.0), Cam(zoom=1.28)

    for k in range(n00):
        t = k / float(n00)
        src = sim.tick(rps_fast)
        canvas.backdrop()
        canvas.place(src, cam_a.lerp(cam_b, (k / float(n00 + n01))),
                     shadow=False, border=False)
        canvas.vignette(0.62)
        canvas.scrim(int(70 + 40 * (1 - t)))
        typo.lower_third(
            canvas.surf, None,
            "32 x 32 grid  ~  4 drones  ~  1 hidden target  ~  "
            "no drone can see more than three cells",
            typo.span(t, 0.10), pos=(64, C.H - 132), width=900)
        yield canvas.surf

    for k in range(n01):
        t = k / float(n01)
        # Slam on the brakes for the failure, then hold for the title.
        rps = 1.4 if t < 0.34 else 0.0
        src = sim.tick(rps)
        canvas.backdrop()
        canvas.place(src, cam_a.lerp(cam_b, ((n00 + k) / float(n00 + n01))),
                     shadow=False, border=False)
        canvas.vignette(0.62)
        canvas.scrim(int(70 + 150 * typo.clamp((t - 0.30) / 0.22)))

        if 0.06 < t < 0.42 and not sim.env.agent_alive[victim]:
            typo.lower_third(canvas.surf, "D%d has failed" % victim,
                             "No warning, no handover, and nobody has been "
                             "told.", typo.span(t, 0.06, 0.44),
                             pos=(64, C.H - 168), width=640,
                             accent=theme.DANGER)

        typo.title_card(
            canvas.surf,
            ["Multi-Drone Cooperative Search", "with Random Faults"],
            "Deep reinforcement learning for a swarm that keeps working "
            "when its members do not",
            typo.span(t, 0.34), kicker="one shared policy")
        yield canvas.surf

    sim.close()

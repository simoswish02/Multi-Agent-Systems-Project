"""Scene 4 -- inside the policy.

The mission freezes on one round and one drone, and the film opens up exactly
what that drone is about to decide with: its six belief planes, the crop the
network takes for itself, the five context numbers, the trunk, and the four
Q-values that pick the move.

Every array drawn is the live observation, and every number is a live forward
pass. ``demo.tensorviz.verify`` re-derives all of it from the environment and
from ``CnnQNetwork`` before a single frame is rendered; if the picture and the
tensors ever disagree the build stops here.
"""

import math


import pygame
import torch

from agents.networks import GLOBAL_CHANNELS, LOCAL_CHANNELS
from gui import theme
from demo import config as C
from demo import seeds as S
from demo import typo
from demo.compositor import Canvas, Cam, frame_rect
from demo.tensorviz import netdiagram as ND
from demo.tensorviz import planes as P
from demo.tensorviz import verify as V


# Where to freeze the mission. Far enough past the first fault that the map is
# well explored -- and then a little further if need be, until the news of the
# crash has actually reached a drone that is still flying. Channel 5 is one of
# the six the narration walks through, so it has to have something in it; we
# choose *when* to look, never what is there.
FREEZE_AFTER_FAULT = 26
FREEZE_SEARCH_MAX  = 60

# One geometry for every beat that shows the observation, so the picture never
# jumps: the six planes on the left, the crop on the right, context underneath.
GRID_ORIGIN = (60, 170)
GRID_CELL   = 7
PATCH_XY    = (1150, 230)
PATCH_CELL  = 26
CHIPS_XY    = (345, 800)
NOTE_XY     = (1150, 618)   # copy for the context beats: right of the planes,
NOTE_W      = 700           # under the crop, above the chips


# ---------------------------------------------------------------------------
# Preparing the frozen moment
# ---------------------------------------------------------------------------

def _informed(sim):
    """Live drones that know about at least one crash site."""
    return [i for i in range(sim.env.n_agents)
            if sim.env.agent_alive[i] and sim.env.agent_broken[i].sum() > 0]


def _advance_to_informed(sim, limit=None):
    """Step on until a still-flying drone has learned of a crash.

    A beacon only reaches a teammate that comes close enough, so how long this
    takes is a property of the run, not something the film sets.
    """
    limit = FREEZE_SEARCH_MAX if limit is None else limit
    for _ in range(limit):
        if _informed(sim) or sim.done:
            break
        sim.step_round()
        sim.frame(1.0)          # keep the charts and log accumulating
    return sim


def _pick_drone(sim):
    """A live drone that knows about the wreck -- so channel 5 is not empty."""
    live = [i for i in range(sim.env.n_agents) if sim.env.agent_alive[i]]
    if not live:
        return 0
    return max(live, key=lambda i: (float(sim.env.agent_broken[i].sum()),
                                    float(sim.env.agent_visited[i].sum())))


class Moment:
    """Everything the scene draws, captured once and checked once."""

    def __init__(self, sim, policy):
        self.sim, self.policy = sim, policy
        env = sim.env
        self.i = i = _pick_drone(sim)
        grid = env.grid_size

        self.g = V.reshape(sim.obs[i]["global"], GLOBAL_CHANNELS, grid)
        self.l = V.reshape(sim.obs[i]["local"], LOCAL_CHANNELS, grid)

        # The position the network recovers for itself, from channel 4.
        self.pos = V.observation(env, sim.obs[i], i, self.g, self.l)

        net = policy.agent.q_net
        lm = torch.tensor(self.l, dtype=torch.float32).unsqueeze(0)
        ap = torch.tensor([[self.pos[0], self.pos[1]]], dtype=torch.long)
        with torch.no_grad():
            self.patch = net._extract_local_patch(lm, ap)[0].numpy()
        V.patch(net, self.l, self.pos, self.patch)

        self.q, self.ctx = policy.q_values(env, sim.obs[i], i, sim.ctx_params)
        V.context(policy.ctx_norm, sim.ctx_params, i, env.n_alive_belief(i),
                  self.ctx)

        self.mask = env._get_action_mask(i)
        self.action = policy.act(env, sim.obs[i], i, sim.ctx_params)
        V.decision(self.q, self.mask, self.action)

        # The raw values behind each normalised context entry.
        n = policy.ctx_norm
        cp = sim.ctx_params
        self.ctx_raw = [
            "%d / %d" % (cp["vision_radius"], n["vision_radius"]),
            "%d / %d" % (cp["comm_range"], n["comm_range"]),
            "%d / %d" % (cp["n_agents"], n["n_agents"]),
            "%d / %d" % (i, n["n_agents"]),
            "%d / %d" % (env.n_alive_belief(i), n["n_agents"]),
        ]
        self.n_alive_true = int(env.agent_alive.sum())
        self.believes = env.n_alive_belief(i)

    @property
    def color(self):
        return theme.AGENT_COLORS[self.i % len(theme.AGENT_COLORS)]


# ---------------------------------------------------------------------------

def render(ctx):
    V.channel_tables()

    facts = S.hero_facts()
    brk = facts.get("breaks", [[0, 55]])[0][1]
    sim = S.hero_sim(ctx.policy, upto_round=brk + FREEZE_AFTER_FAULT)
    _advance_to_informed(sim)
    m = Moment(sim, ctx.policy)

    canvas = Canvas()
    beats = {b["id"]: ctx.frames(b) for b in ctx.beats}
    clock = {"f": 0}

    def chrome(t_scene):
        typo.chapter(canvas.surf, 4, "Inside the policy", t_scene / 5.0)

    def now():
        return clock["f"] * (1000.0 / C.FPS)

    # -- b40: pick the drone, then dissolve the map into its planes --------
    n = beats["b40"]
    look = frame_rect(sim.cell_rect(*sim.env.agent_pos[m.i], pad=170),
                      sim.size, pad=1.0, max_zoom=2.4)
    cam0 = Cam()
    for k in range(n):
        t = k / float(n)
        src = sim.tick(0.0)                       # the mission is frozen
        cam = cam0.lerp(look, typo.clamp(t / 0.45))
        canvas.backdrop()
        canvas.place(src, cam)
        canvas.vignette(0.45)
        _reticle(canvas, sim, m, typo.clamp((t - 0.30) / 0.25), now())
        chrome(clock["f"] / float(C.FPS))
        typo.lower_third(canvas.surf, "One drone, frozen mid-mission",
                         "Everything from here is what D%d alone holds -- not "
                         "the world, and not the team's combined map." % m.i,
                         t, width=720)
        # The last stretch cross-dissolves into the collapsed stack.
        fade = typo.clamp((t - 0.74) / 0.26)
        if fade > 0:
            _planes_frame(canvas, m, spread=0.0, labels=False,
                          alpha=int(255 * fade), scrim=int(235 * fade))
        clock["f"] += 1
        yield canvas.surf

    # -- b41: the six planes fan out --------------------------------------
    n = beats["b41"]
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        _planes_frame(canvas, m, spread=typo.clamp(t / 0.66))
        chrome(clock["f"] / float(C.FPS))
        typo.lower_third(canvas.surf, "The global stream",
                         "Six 32x32 planes. This is the drone's entire memory "
                         "of where it has been and what it found.",
                         t, pos=(64, C.H - 150), width=690)
        clock["f"] += 1
        yield canvas.surf

    # -- b42: how much that is, and whose it is ---------------------------
    n = beats["b42"]
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        _planes_frame(canvas, m, spread=1.0)
        _brace_total(canvas, m, typo.clamp(t / 0.4))
        chrome(clock["f"] / float(C.FPS))
        typo.lower_third(canvas.surf, "A belief, not the world",
                         "Nothing here is ground truth. Every plane is what "
                         "this drone has seen for itself or been told.",
                         t, pos=(64, C.H - 150), width=690)
        clock["f"] += 1
        yield canvas.surf

    # -- b43: channel 4 -> argmax -> the crop -----------------------------
    n = beats["b43"]
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        _planes_frame(canvas, m, spread=1.0,
                      focus=4 if t > 0.12 else None)
        _argmax_callout(canvas, m, typo.span(t, 0.22))
        _patch_panel(canvas, m, typo.clamp((t - 0.52) / 0.4), now(),
                     show_note=False)
        chrome(clock["f"] / float(C.FPS))
        typo.lower_third(canvas.surf, "It finds itself",
                         "Channel 4 holds a single 1.0. The network takes its "
                         "argmax and crops a 13x13 window centred there.",
                         t, pos=(64, C.H - 150), width=690)
        clock["f"] += 1
        yield canvas.surf

    # -- b44: why thirteen -------------------------------------------------
    n = beats["b44"]
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        _planes_frame(canvas, m, spread=1.0, focus=4, dim=0.5)
        _patch_panel(canvas, m, 1.0, now(), show_note=typo.clamp(t / 0.3))
        chrome(clock["f"] / float(C.FPS))
        clock["f"] += 1
        yield canvas.surf

    # -- b45 / b46: the context vector -------------------------------------
    for bid, focus_note in (("b45", False), ("b46", True)):
        n = beats[bid]
        for k in range(n):
            t = k / float(n)
            canvas.backdrop()
            _planes_frame(canvas, m, spread=1.0, dim=0.5)
            _patch_panel(canvas, m, 1.0, now(), show_note=0.0, dim=0.5)
            hot = 4 if (focus_note and t > 0.18) else None
            P.ctx_chips(canvas.surf, CHIPS_XY, m.ctx, m.ctx_raw,
                        1.0 if focus_note else t, focus=hot)
            chrome(clock["f"] / float(C.FPS))
            if focus_note:
                _belief_note(canvas, m, typo.span(t, 0.22))
            else:
                typo.lower_third(canvas.surf, "Five numbers of context",
                                 "Who it is, and what it is working with, "
                                 "normalised by the training ranges.",
                                 t, pos=NOTE_XY, width=NOTE_W)
            clock["f"] += 1
            yield canvas.surf

    # -- b47: the trunk ----------------------------------------------------
    n = beats["b47"]
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        _diagram(canvas, m, t, now(), stage="trunk")
        chrome(clock["f"] / float(C.FPS))
        clock["f"] += 1
        yield canvas.surf

    # -- b48: the dueling head --------------------------------------------
    n = beats["b48"]
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        _diagram(canvas, m, 1.0, now(), stage="head", head_t=t)
        chrome(clock["f"] / float(C.FPS))
        clock["f"] += 1
        yield canvas.surf

    # -- b49: the four numbers, and the move ------------------------------
    n = beats["b49"]
    moved = {"done": False}
    for k in range(n):
        t = k / float(n)
        canvas.backdrop()
        if t < 0.62:
            _diagram(canvas, m, 1.0, now(), stage="q", head_t=1.0,
                     q_t=typo.clamp(t / 0.5))
        else:
            # Back to the map, and the drone takes exactly that move.
            if not moved["done"]:
                sim.step_round()
                moved["done"] = True
            src = sim.tick(0.0)
            fade = typo.clamp((t - 0.62) / 0.14)
            canvas.place(src, look)
            canvas.vignette(0.45)
            canvas.scrim(int(200 * (1 - fade)))
            if fade < 1.0:
                _diagram(canvas, m, 1.0, now(), stage="q", head_t=1.0, q_t=1.0,
                         alpha=int(255 * (1 - fade)))
            _move_flag(canvas, sim, m, typo.span(t, 0.72), now())
        chrome(clock["f"] / float(C.FPS))
        clock["f"] += 1
        yield canvas.surf

    sim.close()


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def _reticle(canvas, sim, m, t, now):
    """A targeting ring that settles on the chosen drone."""
    if t <= 0.01:
        return
    cx, cy = canvas.to_canvas(sim.agent_px(m.i))
    e = typo.ease_out(t)
    r = int(150 - 96 * e)
    a = int(235 * e)
    layer = pygame.Surface((C.W, C.H), pygame.SRCALPHA)
    pygame.draw.circle(layer, theme.with_alpha(m.color, a), (int(cx), int(cy)),
                       r, 2)
    for ang in (0, 90, 180, 270):
        rad = math.radians(ang + now * 0.02)
        x1 = cx + math.cos(rad) * (r - 12)
        y1 = cy + math.sin(rad) * (r - 12)
        x2 = cx + math.cos(rad) * (r + 12)
        y2 = cy + math.sin(rad) * (r + 12)
        pygame.draw.line(layer, theme.with_alpha(m.color, a), (x1, y1), (x2, y2), 2)
    canvas.surf.blit(layer, (0, 0))
    if e > 0.6:
        f = theme.font(24, bold=True)
        g = f.render("D%d" % m.i, True, m.color)
        typo.blit_alpha(canvas.surf, g, (cx - g.get_width() // 2, cy - r - 36),
                        int(255 * (e - 0.6) / 0.4))


def _grid():
    return P.Grid(P.GLOBAL_CHANNELS, cols=3, cell=GRID_CELL, gap_x=50,
                  gap_y=92)


def _planes_frame(canvas, m, spread, *, labels=True, alpha=255, focus=None,
                  scrim=0, dim=1.0):
    """The six global channels, fanning out of the map they came from."""
    if scrim:
        canvas.scrim(scrim)
    g = _grid()
    g.draw(canvas.surf, m.g, GRID_ORIGIN, spread, alpha=int(alpha * dim),
           focus=focus, labels=labels)
    if labels and spread > 0.55:
        f = theme.font(17, bold=True, mono=True)
        t = f.render("obs['global']  --  6 x 32 x 32", True, theme.TEXT_DIM)
        typo.blit_alpha(canvas.surf, t,
                        (GRID_ORIGIN[0], GRID_ORIGIN[1] - 40),
                        int(alpha * dim))


def _brace_total(canvas, m, t):
    """Total the stack up: how many numbers this actually is."""
    if t <= 0.01:
        return
    g = _grid()
    w, h = g.size(len(P.GLOBAL_CHANNELS))
    bx = GRID_ORIGIN[0] + w + 54
    top, bot = GRID_ORIGIN[1] + 10, GRID_ORIGIN[1] + h - 10
    P.brace(canvas.surf, bx, top, bot, theme.ACCENT, None, t, side=-1)
    if t < 0.5:
        return
    a = int(255 * (t - 0.5) / 0.5)
    f_b = theme.font(52, bold=True, mono=True)
    f_s = theme.font(19)
    gg = f_b.render("6,144", True, theme.TEXT)
    typo.blit_alpha(canvas.surf, gg, (bx + 36, (top + bot) // 2 - 54), a)
    for j, line in enumerate(["numbers, rebuilt every round,",
                              "privately, for every drone"]):
        gs = f_s.render(line, True, theme.TEXT_DIM)
        typo.blit_alpha(canvas.surf, gs,
                        (bx + 38, (top + bot) // 2 + 8 + j * 26), int(a * 0.9))


def _argmax_callout(canvas, m, t):
    """Point at the single 1.0 in channel 4 and say what is done with it."""
    if t <= 0.01:
        return
    r, c = int(m.pos[0]), int(m.pos[1])
    px, py = _grid().cell_px(4, GRID_ORIGIN, r, c)
    pulse = 0.5 + 0.5 * math.sin(t * 13.0)
    pygame.draw.circle(canvas.surf, theme.INFO, (int(px), int(py)),
                       int(10 + 6 * pulse), 2)
    typo.callout(canvas.surf, (int(px), int(py)),
                 "one 1.0 among 1,024 zeros -> row %d, column %d" % (r, c),
                 t, side="right", color=theme.INFO, title="argmax", dist=140)


def _patch_panel(canvas, m, t, now, *, show_note=0.0, dim=1.0):
    """The 13x13 crop, lifted out of the local stream and enlarged."""
    if t <= 0.01:
        return
    x, y = PATCH_XY
    a = int(255 * typo.ease_out(t) * dim)
    surf = P.patch_surface(m.patch, P.LOCAL_CHANNELS, cell=PATCH_CELL)
    sw, sh = surf.get_size()
    e = typo.ease_out(t)
    if e < 1.0:
        surf = pygame.transform.smoothscale(
            surf, (max(2, int(sw * (0.62 + 0.38 * e))),
                   max(2, int(sh * (0.62 + 0.38 * e)))))
    typo.blit_alpha(canvas.surf, surf, (x, y), a)

    f = theme.font(17, bold=True, mono=True)
    g = f.render("local patch  --  5 x 13 x 13", True, theme.TEXT_DIM)
    typo.blit_alpha(canvas.surf, g, (x, y - 34), a)

    if show_note:
        na = int(255 * typo.ease_out(show_note))
        f_b = theme.font(36, bold=True, mono=True)
        f_s = theme.font(19)
        by = y + sh + 28
        typo.blit_alpha(canvas.surf, f_b.render("13 = 2 x 6 + 1", True,
                                                theme.ACCENT), (x, by), na)
        for j, line in enumerate(["the largest vision radius",
                                  "the policy ever trained on"]):
            gs = f_s.render(line, True, theme.TEXT_DIM)
            typo.blit_alpha(canvas.surf, gs, (x, by + 48 + j * 26),
                            int(na * 0.9))


def _belief_note(canvas, m, t):
    """Spell out that `alive` is a belief, and whether it is currently right."""
    if t <= 0.01:
        return
    right = (m.believes == m.n_alive_true)
    head = "A belief that can be wrong"
    body = ("D%d believes %d of %d drones are still flying. %s"
            % (m.i, m.believes, m.sim.env.n_agents,
               "Right now it happens to be correct -- it has no way to know that."
               if right else
               "It is out of date: %d are actually left. The policy acts on the "
               "belief anyway." % m.n_alive_true))
    typo.lower_third(canvas.surf, head, body, t, pos=NOTE_XY, width=NOTE_W,
                     accent=theme.WARNING)


# ---------------------------------------------------------------------------
# The schematic
# ---------------------------------------------------------------------------

def _diagram(canvas, m, t, now, *, stage="trunk", head_t=0.0, q_t=0.0,
             alpha=255):
    """Layout C: inputs down the left, the two trunks across, the head at the
    right -- and, when the Q-values arrive, the whole schematic steps back and
    hands them the frame."""
    surf = canvas.surf
    dim = 1.0
    if stage == "q":
        # The four numbers are the point of the scene; everything else recedes.
        dim = 1.0 - 0.62 * typo.clamp(q_t / 0.35)
    a = int(alpha * dim)

    _inputs_column(surf, m, a, now)
    g_rects, l_rects = _trunks(surf, m, t, now, a)
    jx = _join(surf, g_rects, l_rects, t, a)

    if stage != "trunk":
        _head(surf, jx, head_t, a)

    if stage == "q":
        ND.q_bars(surf, (700, 660), m.q, q_t, mask=m.mask,
                  chosen=int(m.action), width=470, row_h=60, gap=16, now=now)


IN_X = 60
G_TRUNK_Y, L_TRUNK_Y = 300, 560
BUS_Y = 196                      # the context bus, above the global trunk
CHIPS_Y = 806                    # where the context chips sit in that column


def _inputs_column(surf, m, alpha, now):
    """The three things the network is handed, as small reminders."""
    f = theme.font(15, bold=True)
    y = 150
    stack = P.Grid(P.GLOBAL_CHANNELS, cols=2, cell=3, gap_x=14, gap_y=30)
    typo.blit_alpha(surf, f.render("global  6 x 32 x 32", True, theme.TEXT_DIM),
                    (IN_X, y - 24), alpha)
    stack.draw(surf, m.g, (IN_X, y), 1.0, alpha=int(alpha * 0.9), labels=False)
    y += stack.size(6)[1] + 54

    typo.blit_alpha(surf, f.render("patch  5 x 13 x 13", True, theme.TEXT_DIM),
                    (IN_X, y - 24), alpha)
    pat = P.patch_surface(m.patch, P.LOCAL_CHANNELS, cell=9)
    typo.blit_alpha(surf, pat, (IN_X, y), int(alpha * 0.95))
    y += pat.get_height() + 54

    y = CHIPS_Y
    typo.blit_alpha(surf, f.render("context  5", True, theme.TEXT_DIM),
                    (IN_X, y - 24), alpha)
    fs = theme.font(13, bold=True, mono=True)
    for k in range(5):
        chip = pygame.Rect(IN_X + k * 36, y, 30, 30)
        pygame.draw.rect(surf, theme.PANEL_ALT, chip, border_radius=4)
        pygame.draw.rect(surf, theme.WARNING if k == 4 else theme.BORDER,
                         chip, 1, border_radius=4)
        g = fs.render("%.1f" % m.ctx[k], True, theme.TEXT)
        surf.blit(g, (chip.centerx - g.get_width() // 2,
                      chip.centery - g.get_height() // 2))
    return y


def _trunks(surf, m, t, now, alpha):
    """The two convolutional paths, and the context bus feeding the global one."""
    tx = 300
    g_rects = ND.trunk(surf, tx, G_TRUNK_Y, ND.GLOBAL_TRUNK, t, height=74)
    l_rects = ND.trunk(surf, tx, L_TRUNK_Y, ND.LOCAL_TRUNK,
                       typo.clamp(t / 0.9), height=74)

    # FiLM: one bus above the trunk, dropping into each conditioned stage.
    ft = typo.clamp((t - 0.45) / 0.42)
    if ft > 0.02:
        pulse = 0.45 + 0.35 * math.sin(now / 260.0)
        col = theme.blend(theme.WARNING, theme.BG, 1.0 - pulse)
        x_end = int(tx + (g_rects[3].centerx - tx) * typo.smooth(ft))
        bus_x = IN_X - 22
        pygame.draw.line(surf, col, (bus_x, BUS_Y), (x_end, BUS_Y), 2)
        pygame.draw.line(surf, col, (bus_x, BUS_Y), (bus_x, CHIPS_Y + 15), 2)
        pygame.draw.line(surf, col, (bus_x, CHIPS_Y + 15),
                         (IN_X - 6, CHIPS_Y + 15), 2)
        for r in g_rects[1:4]:
            if r.centerx <= x_end:
                ND.arrow(surf, (r.centerx, BUS_Y + 4), (r.centerx, r.top - 6),
                         1.0, color=col, width=2, head=7)
        if ft > 0.75:
            fs = theme.font(15, bold=True)
            g = fs.render("FiLM -- the context scales and shifts every stage",
                          True, theme.WARNING)
            typo.blit_alpha(surf, g, (tx, BUS_Y - 26),
                            int(alpha * (ft - 0.75) / 0.25))
    return g_rects, l_rects


def _join(surf, g_rects, l_rects, t, alpha):
    """Where the two streams and the context meet."""
    jx = 1272
    jt = typo.clamp((t - 0.58) / 0.42)
    mid = (G_TRUNK_Y + L_TRUNK_Y) // 2
    ND.arrow(surf, (g_rects[-1].right, G_TRUNK_Y), (jx - 10, mid - 42), jt)
    ND.arrow(surf, (l_rects[-1].right, L_TRUNK_Y), (jx - 10, mid + 2), jt)

    ND.tensor_bar(surf, pygame.Rect(jx, mid - 54, 150, 24), "1024", jt,
                  color=theme.INFO, alpha=alpha)
    ND.tensor_bar(surf, pygame.Rect(jx, mid - 10, 96, 24), "384",
                  typo.clamp((t - 0.64) / 0.36), color=theme.ACCENT,
                  alpha=alpha)
    ND.tensor_bar(surf, pygame.Rect(jx, mid + 34, 26, 24), "5",
                  typo.clamp((t - 0.70) / 0.30), color=theme.WARNING,
                  alpha=alpha)
    if jt > 0.7:
        aa = int(alpha * (jt - 0.7) / 0.3)
        f = theme.font(22, bold=True, mono=True)
        typo.blit_alpha(surf, f.render("concat -> 1413", True, theme.TEXT),
                        (jx, mid + 80), aa)
        fs = theme.font(15)
        typo.blit_alpha(surf, fs.render("1536 -> 768 -> 384", True,
                                        theme.TEXT_DIM), (jx, mid + 110), aa)
    return jx


def _head(surf, jx, head_t, alpha):
    """The dueling split, and the identity that puts it back together."""
    hx = 1478
    mid = (G_TRUNK_Y + L_TRUNK_Y) // 2
    ND.arrow(surf, (jx + 190, mid + 92), (hx - 12, mid), typo.clamp(head_t / 0.22))
    ND.dueling(surf, pygame.Rect(hx, 214, 340, 330), head_t)
    ND.equation(surf, (hx + 170, 600), typo.clamp((head_t - 0.55) / 0.4),
                alpha=alpha)


def _move_flag(canvas, sim, m, t, now):
    """Back on the map: label the move the network just took."""
    if t <= 0.01:
        return
    cx, cy = canvas.to_canvas(sim.agent_px(m.i))
    typo.callout(canvas.surf, (int(cx), int(cy)),
                 "D%d moved %s -- the argmax, executed"
                 % (m.i, ND.ACTION_NAMES[int(m.action)].lower()),
                 t, side="right", color=m.color, title="decision", dist=150)

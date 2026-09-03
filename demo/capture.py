"""Headless capture: drive a real episode and emit video frames.

The rollout mirrors ``main.py::run_simulate`` exactly -- sequential per-drone
turns, wrecks still burning a turn, the ``set_domain_params`` -> ``reset``
invariant -- so what the film shows is what the system does. The only addition
is that the GUI is rendered several times per round, gliding the drones between
cells, instead of once per round.

Invariant kept from docs/ARCHITECTURE.md: BFS auto-nav is eval/simulate-only, and it stays
off unless a scene explicitly asks for it.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

from demo import config as C


# ---------------------------------------------------------------------------
# Specification of one run
# ---------------------------------------------------------------------------

@dataclass
class SimSpec:
    """Everything that makes one episode reproducible."""
    n_agents:     int   = 4
    vision:       int   = 3
    comm:         int   = 5
    density:      float = 0.20
    fault_prob:   float = 0.0
    max_steps:    int   = 200
    seed:         int   = 0
    auto_nav:     bool  = False
    spawn_corner: int   = 0        # 0 = top-left, as the setup screen forces


@dataclass
class RoundInfo:
    """What happened during one round -- the hooks scenes direct on."""
    step:       int  = 0
    broke:      list = field(default_factory=list)   # drones that just failed
    comm_pairs: list = field(default_factory=list)
    found:      bool = False
    done:       bool = False
    n_alive:    int  = 0


# ---------------------------------------------------------------------------
# The shared policy
# ---------------------------------------------------------------------------

class Policy:
    """The trained shared network, loaded once and reused by every Sim."""

    def __init__(self, weights=None, device=None):
        from main import build_agent
        from training.train import build_ctx_norm

        with open(C.CONFIG) as f:
            self.config = yaml.safe_load(f)

        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx_norm = build_ctx_norm(self.config)
        self.agent    = build_agent(self.config, device=self.device)
        self.agent.load(weights or C.WEIGHTS)
        self.agent.epsilon = 0.0

    def _ctx(self, env, i, ctx_params):
        from training.train import ctx_to_numpy
        ctx_np = ctx_to_numpy(ctx_params, self.ctx_norm, i, env.n_alive_belief(i))
        ctx_t = torch.tensor(
            ctx_np, dtype=torch.float32, device=torch.device(self.device)
        ).unsqueeze(0)
        return ctx_np, ctx_t

    def act(self, env, obs_i, i, ctx_params):
        _, ctx_t = self._ctx(env, i, ctx_params)
        return self.agent.select_action(
            obs_i["global"], obs_i["local"], ctx_t,
            greedy=True, action_mask=env._get_action_mask(i),
        )

    def act_round(self, env, obs, ctx_params, idx):
        """Greedy actions for several drones in one forward pass.

        Every input a drone acts on is already fixed when the round starts:
        ``obs[i]`` was written at the end of that drone's own last turn and
        nothing else touches it, and the action mask depends only on the
        static grid. The one exception is ``n_alive_belief``, which can change
        mid-round when a teammate fails. So the beliefs used are returned too,
        and the caller re-runs any drone whose belief moved -- making this
        exactly equal to acting one at a time, about four times faster.
        """
        from training.train import ctx_to_numpy

        beliefs = {i: env.n_alive_belief(i) for i in idx}
        ctxs  = [ctx_to_numpy(ctx_params, self.ctx_norm, i, beliefs[i])
                 for i in idx]
        obss  = [obs[i] for i in idx]
        masks = [env._get_action_mask(i) for i in idx]
        acts  = self.agent.select_actions_batch(obss, ctxs, masks)
        return dict(zip(idx, acts)), beliefs

    def q_values(self, env, obs_i, i, ctx_params):
        """Raw Q-values for one drone. ``select_action`` throws these away after
        the argmax, so scene 4 reads them off the network directly."""
        ctx_np, ctx_t = self._ctx(env, i, ctx_params)
        dev = torch.device(self.device)
        og = torch.tensor(obs_i["global"], dtype=torch.float32,
                          device=dev).unsqueeze(0)
        ol = torch.tensor(obs_i["local"], dtype=torch.float32,
                          device=dev).unsqueeze(0)
        was_training = self.agent.q_net.training
        self.agent.q_net.eval()
        with torch.no_grad():
            q = self.agent.q_net(og, ol, ctx_t)[0].cpu().numpy()
        self.agent.q_net.train(was_training)
        return q, ctx_np


# ---------------------------------------------------------------------------
# One recordable episode
# ---------------------------------------------------------------------------

class Sim:
    """One environment plus its offscreen renderer, steppable and renderable."""

    def __init__(self, spec, policy, cell=None, render=True):
        import copy
        from env.grid_env import DroneSearchEnv
        from gui.renderer import DroneRenderer

        self.spec   = spec
        self.policy = policy

        cfg = copy.deepcopy(policy.config)
        cfg["env"]["n_agents"]     = spec.n_agents
        cfg["env"]["max_steps"]    = spec.max_steps
        cfg["env"]["spawn_corner"] = spec.spawn_corner

        self.ctx_params = {
            "vision_radius": spec.vision,
            "comm_range":    spec.comm,
            "n_agents":      spec.n_agents,
        }

        self.env = DroneSearchEnv(cfg, render_mode=None)
        self._apply_domain()

        # The headless renderer sizes itself from CELL_DEFAULT at construction.
        # Seed scouting runs thousands of episodes it never shows, so it skips
        # the renderer entirely.
        if render:
            DroneRenderer.CELL_DEFAULT = cell or C.CELL
            self.renderer = DroneRenderer(self.env, headless=True)
            self.env.renderer = self.renderer
        else:
            self.renderer = None

        self.t_ms      = 0.0
        self.done      = False
        self.found     = False
        self.obs       = None
        self._prev_pos = None
        self._cur_pos  = None
        self._fir      = 0.0     # video frames elapsed inside the current round

    def _apply_domain(self):
        # Invariant 5: domain params are set before every reset().
        self.env.set_domain_params(
            **self.ctx_params,
            obstacle_density=self.spec.density,
            fault_prob=self.spec.fault_prob,
        )

    # -- geometry ----------------------------------------------------------
    @property
    def size(self):
        """(width, height) of a captured frame."""
        return self.renderer.screen.get_size()

    @property
    def grid_px(self):
        """Width of the map area; the sidebar starts at this x."""
        return self.env.W * self.renderer.CELL

    def regions(self):
        """Named rects of the captured frame, for the camera to target.

        Taken from the renderer's own layout so they can never drift from what
        is actually drawn.
        """
        import pygame

        r = self.renderer
        w, h = self.size
        sx, sy = self.grid_px, r.TOOLBAR_H
        sw, sh = w - sx, h - sy
        out = {
            "all":     pygame.Rect(0, 0, w, h),
            "toolbar": pygame.Rect(0, 0, w, r.TOOLBAR_H),
            "map":     pygame.Rect(0, sy, sx, h - sy),
            "sidebar": pygame.Rect(sx, sy, sw, sh),
        }
        lay = r._layout_sidebar(sx, sy, sw, sh)
        for key, name in (("minimaps", "minimaps"), ("charts", "charts"),
                          ("log", "log")):
            body, hdr = lay.get(key + "_body"), lay.get(key + "_header")
            if body is not None and hdr is not None:
                out[name] = pygame.Rect(hdr.x, hdr.y, hdr.width,
                                        body.bottom - hdr.y)
        return out

    def cell_rect(self, r, c, pad=0):
        """Captured-frame rect of one grid cell -- anchors callouts on drones."""
        import pygame
        cs = self.renderer.CELL
        return pygame.Rect(int(c * cs) - pad, int(r * cs) + self.renderer.TOOLBAR_H - pad,
                           cs + 2 * pad, cs + 2 * pad)

    def agent_px(self, i):
        """Captured-frame centre pixel of drone `i`."""
        cs = self.renderer.CELL
        r, c = self._cur_pos[i]
        return (c * cs + cs / 2.0, r * cs + self.renderer.TOOLBAR_H + cs / 2.0)

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self._apply_domain()
        self.obs, _ = self.env.reset(seed=self.spec.seed)
        self.env.debug_nav_paths = {}
        self.done  = False
        self.found = False
        self._cur_pos  = [(float(r), float(c)) for r, c in self.env.agent_pos]
        self._prev_pos = list(self._cur_pos)
        self._fir      = 0.0
        return self

    def step_round(self):
        """One full round: every drone takes its turn, wrecks included."""
        from env.utils import auto_nav_action

        env  = self.env
        info = RoundInfo(step=int(env.step_count),
                         n_alive=int(env.agent_alive.sum()))
        self._prev_pos = list(self._cur_pos)

        if self.done:
            info.done = True
            return info

        alive_before = env.agent_alive.copy()

        # One shared forward pass for the whole round (see Policy.act_round).
        plan, beliefs = {}, {}
        live = [i for i in range(env.n_agents) if env.agent_alive[i]]
        if live and not self.spec.auto_nav:
            plan, beliefs = self.policy.act_round(
                env, self.obs, self.ctx_params, live)

        last = {}
        for i in range(env.n_agents):
            if self.done:
                break
            if not env.agent_alive[i]:
                env.debug_nav_paths.pop(i, None)
                obs_i, _, term, trunc, last = env.step_agent(i, 0)
                self.obs[i] = obs_i
                self.done = term or trunc
                continue

            action = None
            if self.spec.auto_nav:
                action, path = auto_nav_action(env, i)
                if action is None:
                    env.debug_nav_paths.pop(i, None)
                else:
                    env.debug_nav_paths[i] = path
            if action is None:
                if i in plan and env.n_alive_belief(i) == beliefs[i]:
                    action = plan[i]          # batch is still valid for i
                else:
                    action = self.policy.act(env, self.obs[i], i,
                                             self.ctx_params)

            obs_i, _, term, trunc, last = env.step_agent(i, action)
            self.obs[i] = obs_i
            self.done = term or trunc

        self._cur_pos = [(float(r), float(c)) for r, c in env.agent_pos]
        self.found = bool(last.get("found", self.found))

        info.step       = int(env.step_count)
        info.broke      = [i for i in range(env.n_agents)
                           if alive_before[i] and not env.agent_alive[i]]
        info.comm_pairs = (self.renderer._comm_pairs()
                           if self.renderer is not None else [])
        info.found      = self.found
        info.done       = self.done
        info.n_alive    = int(env.agent_alive.sum())
        return info

    # -- rendering ---------------------------------------------------------
    def frame(self, alpha=1.0):
        """Render the current state with the drones eased ``alpha`` of the way
        from their previous cells to their current ones, then advance the
        animation clock by one video frame.

        Returns the renderer's own surface, which is redrawn on every call --
        blit it before asking for the next frame.
        """
        from gui import effects

        te  = effects.ease_smoothstep(float(alpha))
        pos = [effects.lerp_pos(a, b, te)
               for a, b in zip(self._prev_pos, self._cur_pos)]
        surf = self.renderer.draw_offscreen(now_ms=int(self.t_ms), display_pos=pos)
        self.t_ms += 1000.0 / C.FPS
        return surf

    def tick(self, rps=None, on_round=None):
        """Advance the film by exactly one video frame.

        Rounds are stepped only when enough frame-time has accumulated, so a
        scene can vary its pace (or freeze at ``rps=0``) without ever losing
        sync between the picture and the episode. ``on_round`` receives each
        RoundInfo as it happens.
        """
        rps = C.RPS_NORMAL if rps is None else rps
        if rps <= 0:                       # frozen: clock runs, episode does not
            return self.frame(1.0)

        fpr = C.FPS / float(rps)           # video frames per env round
        if self._fir >= fpr - 1e-9:
            self._fir -= fpr
            if not self.done:
                info = self.step_round()
                if on_round is not None:
                    on_round(info)
        self._fir += 1.0
        alpha = 1.0 if fpr <= 1.0 else min(1.0, self._fir / fpr)
        return self.frame(alpha)

    def hold(self, seconds, fps=None):
        """Emit frames without stepping. The clock still runs, so beacons,
        comm links and rotors keep moving."""
        fps = fps or C.FPS
        for _ in range(max(1, int(round(seconds * fps)))):
            yield self.frame(1.0)

    def play(self, seconds, rps=None, fps=None, on_round=None):
        """Convenience wrapper over `tick`: run for `seconds` of film."""
        fps = fps or C.FPS
        for _ in range(max(1, int(round(seconds * fps)))):
            yield self.tick(rps, on_round)

    def close(self):
        """Release this run without shutting pygame down.

        ``DroneSearchEnv.close`` forwards to ``DroneRenderer.close``, which
        calls ``pygame.quit()`` -- fine for a one-shot script, fatal in a build
        that renders seven scenes back to back, because it tears the font
        subsystem out from under every scene that follows. A headless renderer
        owns nothing but a Surface, so detaching it is enough.
        """
        self.env.renderer = None
        self.renderer = None
        self.env.close()


def make_policy(weights=None, device=None):
    """Cached policy factory -- the checkpoint is ~270 MB, load it once."""
    key = (weights or C.WEIGHTS, device)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = Policy(weights=weights, device=device)
    return _POLICY_CACHE[key]


_POLICY_CACHE = {}

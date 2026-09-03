"""Pygame renderer for DroneSearchEnv.

Layout
------
    +--------------------------------------------------+
    |  toolbar: [Play][Pause][Step]  status   fps/step |
    +-----------------------------+--------------------+
    |                             |  A. drone minimaps |
    |         main grid           |  B. realtime charts|
    |   (merged team knowledge)   |  C. episode log    |
    +-----------------------------+--------------------+

The renderer is purely a *view*: it does not drive the environment. The
simulation loop in `main.py` calls `env.render()` once per round and then
advances every drone. The playback controls work by gating that loop from
inside `render()`: while PAUSED the call blocks (keeping the window responsive)
until the user presses Play or requests a single Step. One Step therefore
advances exactly one call's worth of environment stepping (one round in the
eval loop). See docs/REFACTOR_REPORT.md ("GUI — What changed").

All UI strings are in English. Comments are in English.
"""

import math
from collections import deque

import numpy as np

try:
    import pygame
    from gui import charts
    from gui import effects
    from gui import icons
    from gui import terminal as terminal_widget
    from gui import theme
    from gui.theme import (
        COLOR_UNKNOWN, COLOR_FREE, COLOR_OBSTACLE, COLOR_TARGET,
        COLOR_GRID_LINE, COLOR_BG, COLOR_TEXT, COLOR_COMM_FLASH,
        COLOR_REWARD_POS, COLOR_REWARD_NEG, COLOR_BROKEN, COLOR_BROKEN_X,
        COLOR_NAV_PATH,
        COLOR_TOOLBAR_BG, COLOR_SIDEBAR_BG, COLOR_BORDER, COLOR_SECTION_HDR,
        COLOR_BTN, COLOR_BTN_ACTIVE, COLOR_BTN_DISABLED, COLOR_PLAYING,
        COLOR_PAUSED, COLOR_INACTIVE_PANEL, COLOR_LABEL,
        COLOR_LOG_HEADER, COLOR_LOG_NORMAL, COLOR_LOG_COMM, COLOR_LOG_TARGET,
        COLOR_LOG_NEG, COLOR_LOG_BROKEN, AGENT_COLORS,
    )
    PYGAME_AVAILABLE = True
except ImportError:                      # pragma: no cover - exercised only without pygame
    PYGAME_AVAILABLE = False
    charts = None
    terminal_widget = None

_SCREEN_MARGIN = 80
MINIMAP_SLOTS = 4


class DroneRenderer:
    """Pygame-based renderer for DroneSearchEnv.

    Public interface (unchanged):
        __init__(env, headless=False)
        render()            -> draw one human-mode frame (gated by Play/Pause)
        get_rgb_array()     -> draw to an offscreen surface and return HxWx3
        close()             -> shut pygame down

    Extra public helper:
        reset_episode_data() -> clear chart history / per-episode trackers
    """

    CELL_DEFAULT = 22
    FPS = 10                       # stepped-mode frames (= env rounds) per second
    FPS_SMOOTH = 60                # sub-loop frame rate in smooth mode
    ROUND_MS = 100                 # smooth glide duration per round at speed 1.0x
    SPEED_STEPS = (0.5, 1.0, 2.0, 4.0)

    SIDEBAR_W    = 500
    TOOLBAR_H    = 44
    MINIMAP_BODY = 250
    LOG_BODY     = 150
    DESIRED_SIDEBAR_H = 820

    def __init__(self, env, headless: bool = False):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required. Install with: pip install pygame")

        self.env = env
        self.headless = headless
        self._closed = False
        self.CELL = self.CELL_DEFAULT

        # Playback state
        self.playing = True            # default state on open
        self.step_pending = False
        self.show_comm_range = False   # toggle: overlay each drone's comm range
        self.smooth = False            # toggle (M): glide between cells vs stepped
        self.show_trails = True        # toggle (T): fading recent-position trails
        self.show_legend = False       # toggle (L): symbol legend overlay
        self.speed_idx = 1             # index into SPEED_STEPS (1.0x)

        # Unconsumed KEYDOWNs forwarded to external loops (e.g. play mode).
        self.key_events = deque(maxlen=8)

        # Cached terrain layer (rebuilt per round, blitted per frame) and a
        # reusable full-window SRCALPHA overlay for translucent drawing.
        self._terrain_surf = None
        self._terrain_dirty = True
        self._overlay = None

        # Drawn (possibly tweened) agent positions in float grid coords.
        self._display_pos = None
        self._heading = []
        self._teleport = False         # set on episode boundaries: snap, no glide

        # Optional virtual animation clock in ms. When set it overrides both
        # the wall clock and the step-derived headless clock, so an offscreen
        # recorder can emit several video frames per env round.
        self._vclock_ms = None

        # Collapsible sidebar sections (all expanded by default)
        self.sections = {"minimaps": True, "charts": True, "log": True}

        # Widgets / per-frame UI bookkeeping
        self.log = terminal_widget.TerminalLog()
        self.fx = effects.EffectManager()
        self._comm_flash_until = {}    # drone idx -> ms deadline (minimap border)
        self._buttons = {}
        self._section_headers = {}
        self._log_rect = None

        # Episode tracking
        self._prev_step = None
        self._episode_active = False
        self._episode_ended = False
        self.reset_episode_data()

        pygame.init()
        if not headless:
            pygame.display.set_caption("Multi-Drone Search - RL Project")
        self._init_surface(offscreen=headless)

    # ------------------------------------------------------------------
    # Surface / layout setup
    # ------------------------------------------------------------------

    def _compute_cell_size(self, grid_h: int, grid_w: int) -> int:
        if self.headless:
            return self.CELL_DEFAULT
        info = pygame.display.Info()
        max_w = info.current_w - _SCREEN_MARGIN
        max_h = info.current_h - _SCREEN_MARGIN
        cell_by_w = (max_w - self.SIDEBAR_W) // max(grid_w, 1)
        cell_by_h = (max_h - self.TOOLBAR_H) // max(grid_h, 1)
        return max(4, min(self.CELL_DEFAULT, cell_by_w, cell_by_h))

    def _init_surface(self, offscreen: bool = False):
        H, W = self.env.H, self.env.W
        self.CELL = self._compute_cell_size(H, W)
        grid_w = W * self.CELL
        grid_h = H * self.CELL
        width = grid_w + self.SIDEBAR_W
        desired = self.TOOLBAR_H + self.DESIRED_SIDEBAR_H
        if offscreen:
            height = max(self.TOOLBAR_H + grid_h, desired)
            self.screen = pygame.Surface((width, height))
        else:
            info = pygame.display.Info()
            cap = info.current_h - 40
            height = min(max(self.TOOLBAR_H + grid_h, desired), cap)
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

        self.clock = pygame.time.Clock()
        self.font_big   = theme.font(15, bold=True)
        self.font_small = theme.font(12)
        self.font_mono  = theme.font(12, mono=True)
        self.font_tiny  = theme.font(10, mono=True)
        self.log.line_h = self.font_mono.get_height() + 2

        # Cell size may have changed: rebuild size-dependent caches.
        icons.clear_cache()
        self._terrain_surf = None
        self._overlay = None

    @property
    def speed(self) -> float:
        return self.SPEED_STEPS[self.speed_idx]

    def _now(self) -> int:
        """Animation clock in milliseconds. An explicit virtual clock wins when
        set (offscreen recording drives it per video frame); otherwise headless
        frames derive it from step_count so rgb_array output stays
        deterministic."""
        if self._vclock_ms is not None:
            return int(self._vclock_ms)
        if self.headless:
            return int(getattr(self.env, "step_count", 0)) * 100
        return pygame.time.get_ticks()

    # ------------------------------------------------------------------
    # Public per-frame entry points
    # ------------------------------------------------------------------

    def render(self):
        """Draw one round's worth of frames. Blocks while PAUSED until Play /
        Step. In smooth mode a short 60 FPS sub-loop glides the drones from
        their previous to their current cells; one render() call still gates
        exactly one caller round."""
        if self.headless or self._closed:
            return
        self._tick_data()
        self._handle_events()
        if self._closed:
            return

        targets = [(float(r), float(c)) for r, c in self.env.agent_pos]
        prev = self._display_pos
        can_glide = (
            self.smooth and self.playing and not self._teleport
            and prev is not None and len(prev) == len(targets)
            and 0.0 < self._max_jump(prev, targets) <= 2.0
        )
        self._teleport = False
        if can_glide:
            self._animate_round(targets)
        else:
            self._display_pos = targets
            self._draw_flip()
            self.clock.tick(self.FPS * self.speed)
        if self._closed:
            return

        if not self.playing:
            self._pause_gate()

    def _draw_flip(self):
        self._draw()
        pygame.display.flip()

    def _pause_gate(self):
        """Keep the window alive without advancing the simulation; a single
        Step breaks out so the caller advances exactly one round."""
        while not self.playing and not self._closed:
            self._draw_flip()
            self.clock.tick(self.FPS)
            self._handle_events()
            if self.step_pending:        # single-step requested -> let caller advance once
                self.step_pending = False
                break

    @staticmethod
    def _max_jump(a, b):
        """Largest per-drone Manhattan distance between two position lists."""
        return max((abs(ar - br) + abs(ac - bc)
                    for (ar, ac), (br, bc) in zip(a, b)), default=0.0)

    def _animate_round(self, targets):
        """Smooth-mode sub-loop: tween _display_pos toward `targets` over
        ROUND_MS/speed. Pausing mid-glide finishes the move instantly (paused
        positions always reflect the true env state), then render() gates."""
        start = list(self._display_pos)
        period = max(1.0, self.ROUND_MS / self.speed)
        t0 = pygame.time.get_ticks()
        while not self._closed:
            t = (pygame.time.get_ticks() - t0) / period
            te = effects.ease_smoothstep(t)
            self._display_pos = [effects.lerp_pos(a, b, te)
                                 for a, b in zip(start, targets)]
            self._draw_flip()
            self.clock.tick(self.FPS_SMOOTH)
            self._handle_events()
            if t >= 1.0 or not self.playing:
                break
        self._display_pos = targets

    def draw_offscreen(self, now_ms=None, display_pos=None):
        """Draw one frame and return the surface it was drawn on.

        Both arguments are optional and default to the historical behaviour
        (true positions, step-derived clock). A recorder emitting several video
        frames per env round passes ``now_ms`` to advance the animation clock
        and ``display_pos`` (float grid coords) to glide the drones between
        cells. ``_tick_data`` runs first because an episode boundary resets the
        drawn positions.

        The surface is reused every call -- blit or copy it before drawing
        again.
        """
        self._vclock_ms = now_ms
        self._tick_data()
        self._display_pos = display_pos   # None: draw true positions
        self._draw()
        return self.screen

    def get_rgb_array(self, now_ms=None, display_pos=None):
        """Render to the offscreen surface and return an (H, W, 3) array."""
        surf = self.draw_offscreen(now_ms, display_pos)
        return np.transpose(np.array(pygame.surfarray.array3d(surf)), axes=(1, 0, 2))

    def close(self):
        self._closed = True
        pygame.quit()

    # ------------------------------------------------------------------
    # Episode data / history buffers
    # ------------------------------------------------------------------

    def reset_episode_data(self):
        """Clear chart history and per-episode trackers (call on env reset)."""
        n = max(1, int(getattr(self.env, "n_agents", 1)))
        self._hist_step          = []
        self._hist_drone_reward  = [[] for _ in range(n)]
        self._hist_cum_reward    = []
        self._hist_coverage      = []
        self._hist_new_cells     = []
        self._hist_comms         = []

        self._prev_pos           = None
        self._prev_visited_sum   = None
        self._prev_known         = 0
        self._prev_comm_pairs    = set()
        self._prev_alive         = None
        self._comm_flash_until   = {}
        self._found_logged       = False

        self._display_pos        = None
        self._heading            = [None] * n
        self._terrain_dirty      = True
        self._trails             = [deque(maxlen=10) for _ in range(n)]
        self._active_pairs       = []
        self._prev_merged_mask   = None
        self._target_discovered  = False
        self.fx.clear()

        self._last_total = 0.0
        self._last_cov   = 0.0
        self._last_step  = 0

    def _tick_data(self):
        """Detect episode boundaries and update history/log from env state."""
        env = self.env
        if getattr(env, "agent_pos", None) is None:
            return                       # env not reset yet
        step = int(getattr(env, "step_count", 0))

        # A reset rewinds step_count to 0 -> a new episode started.
        if self._prev_step is None or step < self._prev_step:
            if self._episode_active and not self._episode_ended:
                self._log_episode_end()  # best-effort close of the previous episode
            self.reset_episode_data()
            self._log_episode_start()
            self._episode_active = True
            self._episode_ended = False
            self._teleport = True        # never glide across an episode boundary
            self._snapshot(step)
            self._record_sample(step)
            self._on_world_changed(spawn_effects=False)
            self._prev_step = step
            return

        if step == self._prev_step:
            return                       # no new env step since last frame

        # One or more rounds happened since the last frame.
        self._log_round(step)
        self._record_sample(step)
        self._update_headings()
        self._update_trails()
        self._on_world_changed()
        if getattr(env, "done", False) and not self._episode_ended:
            self._log_episode_end()
            self._episode_ended = True
        self._snapshot(step)
        self._prev_step = step

    def _on_world_changed(self, spawn_effects: bool = True):
        """Round bookkeeping for the drawing layer: terrain refresh, comm-pair
        cache, fog fade-in spawns and target-discovery detection."""
        env = self.env
        now = self._now()
        self._terrain_dirty = True
        self._active_pairs = self._comm_pairs()

        mask = self._merged_maps()[0] > 0
        if spawn_effects and self._prev_merged_mask is not None:
            new = np.argwhere(mask & ~self._prev_merged_mask)
            if len(new) <= 220:          # no fireworks on giant one-round reveals
                for r, c in new:
                    self.fx.spawn(effects.CellFade(now, (int(r), int(c))))
        self._prev_merged_mask = mask

        # Latched team-level discovery: any 1.0 in any drone's target map.
        tgt = getattr(env, "agent_target", None)
        if tgt is not None and not self._target_discovered and np.any(tgt > 0):
            self._target_discovered = True
            if spawn_effects:
                self.fx.spawn(effects.DiscoveryFlash(now, tuple(env.target_pos)))
                self.log.add(
                    f"[Step {int(getattr(env, 'step_count', 0)):4d}] [target] "
                    f"target spotted — position now known to the team",
                    COLOR_LOG_TARGET,
                )

    def _update_trails(self):
        env = self.env
        if len(self._trails) != env.n_agents:
            self._trails = [deque(maxlen=10) for _ in range(env.n_agents)]
        for i in range(env.n_agents):
            if not self._is_alive(i):
                continue                 # wrecks stop leaving a trail
            cur = tuple(env.agent_pos[i])
            prev = (self._prev_pos[i]
                    if (self._prev_pos and i < len(self._prev_pos)) else None)
            if cur != prev:
                self._trails[i].append(cur)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def _merged_maps(self):
        env = self.env
        mv = np.zeros((env.H, env.W), dtype=np.float32)
        mo = np.zeros((env.H, env.W), dtype=np.float32)
        for vis, obs in zip(env.agent_visited, env.agent_obstacle):
            np.maximum(mv, vis, out=mv)
            np.maximum(mo, obs, out=mo)
        return mv, mo

    def _known_count(self):
        mv, _ = self._merged_maps()
        return int((mv > 0).sum())

    def _traversable(self):
        env = self.env
        if getattr(env, "grid", None) is None:
            return max(1, env.H * env.W)
        return max(1, int((env.grid != 1).sum()))

    def _is_alive(self, i) -> bool:
        alive = getattr(self.env, "agent_alive", None)
        return True if alive is None else bool(alive[i])

    def _comm_pairs(self):
        """Pairs of LIVE drones currently within comm range (mirrors
        env._communicate — wrecks neither transmit nor receive)."""
        env = self.env
        rng = getattr(env, "comm_range", getattr(env, "comm_radius", 0))
        metric = getattr(env, "comm_metric", "manhattan")
        pairs = []
        n = env.n_agents
        for i in range(n):
            if not self._is_alive(i):
                continue
            ri, ci = env.agent_pos[i]
            for j in range(i + 1, n):
                if not self._is_alive(j):
                    continue
                rj, cj = env.agent_pos[j]
                if metric == "manhattan":
                    d = abs(ri - rj) + abs(ci - cj)
                else:
                    d = ((ri - rj) ** 2 + (ci - cj) ** 2) ** 0.5
                if d <= rng:
                    pairs.append((i, j, float(d)))
        return pairs

    def _drone_new_cells(self, i):
        cur = float(self.env.agent_visited[i].sum())
        if self._prev_visited_sum is None or i >= len(self._prev_visited_sum):
            return 0
        return max(0, int(round(cur - self._prev_visited_sum[i])))

    def _update_headings(self):
        """Remember each live drone's last move direction (grid delta)."""
        env = self.env
        n = env.n_agents
        if len(self._heading) != n:
            self._heading = [None] * n
        if not self._prev_pos:
            return
        for i in range(min(n, len(self._prev_pos))):
            pr, pc = self._prev_pos[i]
            cr, cc = env.agent_pos[i]
            if (cr, cc) != (pr, pc):
                self._heading[i] = (cr - pr, cc - pc)

    # ------------------------------------------------------------------
    # History recording + logging
    # ------------------------------------------------------------------

    def _record_sample(self, step):
        env = self.env
        known = self._known_count()
        trav = self._traversable()
        last_r = list(getattr(env, "last_rewards", []))

        self._hist_step.append(step)
        for i in range(len(self._hist_drone_reward)):
            if i < len(last_r):
                self._hist_drone_reward[i].append(float(last_r[i]))
        self._hist_cum_reward.append(float(getattr(env, "episode_reward", 0.0)))
        self._hist_coverage.append(100.0 * known / trav)
        self._hist_new_cells.append(max(0, known - self._prev_known))
        self._hist_comms.append(len(self._comm_pairs()))

    def _snapshot(self, step):
        env = self.env
        self._prev_pos = [tuple(p) for p in env.agent_pos]
        alive = getattr(env, "agent_alive", None)
        self._prev_alive = None if alive is None else [bool(a) for a in alive]
        self._prev_visited_sum = [float(v.sum()) for v in env.agent_visited]
        known = self._known_count()
        self._prev_known = known
        self._prev_comm_pairs = {(i, j) for (i, j, _d) in self._comm_pairs()}
        self._last_total = float(getattr(env, "episode_reward", 0.0))
        self._last_cov = 100.0 * known / self._traversable()
        self._last_step = step

    @staticmethod
    def _fmt_pos(p):
        return f"({p[0]:>2},{p[1]:>2})"

    def _log_round(self, step):
        env = self.env
        last_r = list(getattr(env, "last_rewards", []))

        # Fault events: a drone alive at the previous frame is now down.
        for i in range(env.n_agents):
            if not self._is_alive(i) and (
                self._prev_alive is None
                or (i < len(self._prev_alive) and self._prev_alive[i])
            ):
                self.log.add(
                    f"[Step {step:4d}] [fault] D{i} BROKE DOWN at "
                    f"{self._fmt_pos(env.agent_pos[i])} — inactive from now on",
                    COLOR_LOG_BROKEN,
                )
                # One-shot red flash at the crash site.
                self.fx.spawn(effects.CommRipple(
                    self._now(), tuple(env.agent_pos[i]), theme.DANGER))

        for i in range(env.n_agents):
            if not self._is_alive(i):
                continue                 # wrecks do not move: skip their rows
            cur = tuple(env.agent_pos[i])
            prev = self._prev_pos[i] if (self._prev_pos and i < len(self._prev_pos)) else cur
            nc = self._drone_new_cells(i)
            rw = last_r[i] if i < len(last_r) else 0.0
            line = (f"[Step {step:4d}] D{i}: {self._fmt_pos(prev)}->{self._fmt_pos(cur)}"
                    f"  new_cells={nc}  reward={rw:+.3f}")
            color = COLOR_LOG_NEG if rw < 0 else COLOR_LOG_NORMAL
            self.log.add(line, color)

        # Communication: log only newly-formed pairs to avoid per-step spam.
        pairs = self._comm_pairs()
        now = self._now()
        for (i, j, d) in pairs:
            if (i, j) not in self._prev_comm_pairs:
                self.log.add(
                    f"[Step {step:4d}] [comm] D{i} <-> D{j} communicated  (distance={d:.1f})",
                    COLOR_LOG_COMM,
                )
                # Radio ripples at both endpoints when the link forms.
                self.fx.spawn(effects.CommRipple(
                    now, tuple(env.agent_pos[i]), AGENT_COLORS[i % len(AGENT_COLORS)]))
                self.fx.spawn(effects.CommRipple(
                    now, tuple(env.agent_pos[j]), AGENT_COLORS[j % len(AGENT_COLORS)]))
        for (i, j, _d) in pairs:
            self._comm_flash_until[i] = now + 400
            self._comm_flash_until[j] = now + 400

        # Target found (best-effort: the env's terminal round may not be rendered
        # in the eval loop; see module docstring).
        if not self._found_logged:
            for i in range(env.n_agents):
                if tuple(env.agent_pos[i]) == tuple(env.target_pos):
                    self.log.add(
                        f"[Step {step:4d}] [target] D{i} found the target at "
                        f"{self._fmt_pos(env.target_pos)}!",
                        COLOR_LOG_TARGET,
                    )
                    self._found_logged = True
                    break

    def _log_episode_start(self):
        env = self.env
        vr = getattr(env, "vision_radius", "N/A")
        cr = getattr(env, "comm_range", getattr(env, "comm_radius", "N/A"))
        od = getattr(env, "obstacle_density", 0.0)
        ms = getattr(env, "max_steps", "N/A")
        fp = getattr(env, "fault_prob", 0.0)
        rule = "=" * 34
        self.log.add_block([
            rule,
            " Episode start",
            f" Grid: {env.H}x{env.W}   Agents: {env.n_agents}",
            f" Vision radius: {vr}",
            f" Comm radius:   {cr}",
            f" Obstacle density: {od:.2f}",
            f" Fault prob/step:  {fp:.4f}",
            f" Max steps: {ms}",
            rule,
        ], COLOR_LOG_HEADER)

    def _log_episode_end(self):
        rule = "-" * 34
        self.log.add_block([
            rule,
            f" Episode end - total reward: {self._last_total:+.3f}",
            f" Coverage: {self._last_cov:.1f}%   Steps: {self._last_step}",
            rule,
        ], COLOR_LOG_HEADER)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type == pygame.VIDEORESIZE:
                self._on_resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.playing = not self.playing
                    if self.playing:
                        self.step_pending = False   # drop a stale queued step
                elif event.key == pygame.K_RIGHT and not self.playing:
                    self.step_pending = True
                elif event.key == pygame.K_c:
                    self.show_comm_range = not self.show_comm_range
                elif event.key == pygame.K_m:
                    self.smooth = not self.smooth
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_l:
                    self.show_legend = not self.show_legend
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed_idx = min(len(self.SPEED_STEPS) - 1, self.speed_idx + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_idx = max(0, self.speed_idx - 1)
                else:
                    # Not ours: forward to the embedding loop (play mode arrows).
                    self.key_events.append(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._on_click(event.pos)
                elif event.button == 4:                 # legacy wheel up
                    self._wheel(event.pos, +3)
                elif event.button == 5:                 # legacy wheel down
                    self._wheel(event.pos, -3)
            elif event.type == pygame.MOUSEWHEEL:
                self._wheel(pygame.mouse.get_pos(), event.y * 3)

    def _on_click(self, pos):
        for key, rect in self._buttons.items():
            if rect.collidepoint(pos):
                if key == "play":
                    self.playing = True
                    self.step_pending = False       # drop a stale queued step
                elif key == "pause":
                    self.playing = False
                elif key == "step" and not self.playing:
                    self.step_pending = True
                elif key == "comm":
                    self.show_comm_range = not self.show_comm_range
                elif key == "smooth":
                    self.smooth = not self.smooth
                elif key == "trails":
                    self.show_trails = not self.show_trails
                elif key == "legend":
                    self.show_legend = not self.show_legend
                elif key == "speed_down":
                    self.speed_idx = max(0, self.speed_idx - 1)
                elif key == "speed_up":
                    self.speed_idx = min(len(self.SPEED_STEPS) - 1, self.speed_idx + 1)
                return
        for key, rect in self._section_headers.items():
            if rect.collidepoint(pos):
                self.sections[key] = not self.sections[key]
                return

    def _wheel(self, pos, amount):
        if self._log_rect is not None and self._log_rect.collidepoint(pos):
            self.log.scroll_by(amount)

    def _on_resize(self, w, h):
        grid_w = max(80, w - self.SIDEBAR_W)
        cell_w = grid_w // max(self.env.W, 1)
        cell_h = (h - self.TOOLBAR_H) // max(self.env.H, 1)
        self.CELL = max(4, min(self.CELL_DEFAULT, cell_w, cell_h))
        self.screen = pygame.display.set_mode(
            (max(w, self.SIDEBAR_W + 120), max(h, 320)), pygame.RESIZABLE
        )
        # Cell size changed: rebuild size-dependent caches.
        icons.clear_cache()
        self._terrain_surf = None
        self._overlay = None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_toolbar()
        self._draw_grid()
        self._draw_sidebar()
        if self.show_legend:
            self._draw_legend()

    def _draw_legend(self):
        """Symbol legend overlay (toggle L), anchored top-right of the grid.
        Glyphs come from the same icon factory as the live view."""
        rows = [
            ("drone",    "Drone (numbered, rotors spin)"),
            ("wreck",    "Wreck — broken drone"),
            ("distress", "Distress beacon (until all know)"),
            ("link",     "Comm link (within range)"),
            ("ripple",   "Comm ripple (new contact)"),
            ("ghost",    "Target — not yet discovered"),
            ("beacon",   "Target — discovered"),
            ("nav",      "Auto-nav BFS path"),
            ("trail",    "Trail (recent cells)"),
            ("fog",      "Fog / explored cell"),
        ]
        gs = 18                          # glyph box size
        row_h, pad = 24, 10
        width = 252
        height = pad * 2 + 20 + row_h * len(rows)
        gw = self.env.W * self.CELL
        panel = pygame.Rect(max(4, gw - width - 8), self.TOOLBAR_H + 8,
                            width, height)
        theme.draw_panel(self.screen, panel, alpha=232)
        t = self.font_small.render("Legend", True, COLOR_TEXT)
        self.screen.blit(t, (panel.x + pad, panel.y + 5))
        now = self._now()
        y = panel.y + pad + 20
        for kind, label in rows:
            self._draw_legend_glyph(kind, panel.x + pad + gs // 2,
                                    y + row_h // 2, gs, now)
            lt = self.font_tiny.render(label, True, COLOR_LABEL)
            self.screen.blit(lt, (panel.x + pad + gs + 10,
                                  y + row_h // 2 - lt.get_height() // 2))
            y += row_h

    def _draw_legend_glyph(self, kind, cx, cy, s, now):
        surf = self.screen
        col0 = AGENT_COLORS[0]
        if kind == "drone":
            icon = icons.get_icon("drone", s, col0,
                                  phase=int((now // 40) % icons.DRONE_PHASES))
            surf.blit(icon, (cx - s // 2, cy - s // 2))
        elif kind == "wreck":
            surf.blit(icons.get_icon("wreck", s), (cx - s // 2, cy - s // 2))
        elif kind == "distress":
            pygame.draw.circle(surf, theme.DANGER, (cx, cy), s // 2, 2)
            pygame.draw.circle(surf, theme.DANGER, (cx, cy), s // 4, 1)
        elif kind == "link":
            effects.draw_dashed_line(surf, theme.COMM, (cx - s + 2, cy),
                                     (cx + s - 2, cy), width=2, dash=5, gap=3,
                                     offset=now * 0.04)
            pygame.draw.circle(surf, (255, 246, 205), (cx, cy), 3)
        elif kind == "ripple":
            pygame.draw.circle(surf, theme.COMM, (cx, cy), s // 2, 1)
            pygame.draw.circle(surf, theme.COMM, (cx, cy), s // 3, 1)
        elif kind == "ghost":
            surf.blit(icons.get_icon("target_ghost", s), (cx - s // 2, cy - s // 2))
        elif kind == "beacon":
            surf.blit(icons.get_icon("target_core", s), (cx - s // 2, cy - s // 2))
        elif kind == "nav":
            pygame.draw.line(surf, COLOR_NAV_PATH, (cx - s + 2, cy), (cx + s - 4, cy), 3)
            pygame.draw.circle(surf, COLOR_NAV_PATH, (cx + s - 4, cy), 4, 2)
        elif kind == "trail":
            for k in range(3):
                col = theme.blend(theme.BG, col0, 0.30 + 0.30 * k)
                pygame.draw.circle(surf, col, (cx - s // 2 + k * (s // 2), cy), 3)
        elif kind == "fog":
            pygame.draw.rect(surf, theme.FOG,
                             pygame.Rect(cx - s // 2, cy - s // 2 + 1, s // 2, s - 2))
            pygame.draw.rect(surf, theme.FREE,
                             pygame.Rect(cx, cy - s // 2 + 1, s // 2, s - 2))

    def _draw_toolbar(self):
        w = self.screen.get_width()
        bar = pygame.Rect(0, 0, w, self.TOOLBAR_H)
        pygame.draw.rect(self.screen, COLOR_TOOLBAR_BG, bar)

        self._buttons = {}
        bh = self.TOOLBAR_H - 16
        y = (self.TOOLBAR_H - 4 - bh) // 2       # leave the bottom 4px to the strip
        x = 8

        # Transport: square icon buttons (triangle / bars / bar+triangle).
        for key in ("play", "pause", "step"):
            rect = pygame.Rect(x, y, 34, bh)
            active = (key == "play" and self.playing) or (key == "pause" and not self.playing)
            disabled = (key == "step" and self.playing)
            theme.draw_button(self.screen, rect, "", self.font_small,
                              active=active, disabled=disabled)
            self._draw_transport_glyph(key, rect, active, disabled)
            self._buttons[key] = rect
            x += 38

        x += 8
        # Toggle chips.
        chips = (("comm",   "Comm",  self.show_comm_range),
                 ("smooth", "Move",  self.smooth),
                 ("trails", "Trail", self.show_trails),
                 ("legend", "Key",   self.show_legend))
        for key, label, on in chips:
            cw = self.font_small.size(label)[0] + 22
            rect = pygame.Rect(x, y + (bh - 24) // 2, cw, 24)
            theme.draw_chip(self.screen, rect, label, self.font_small, on)
            self._buttons[key] = rect
            x += cw + 6

        x += 8
        # Speed group: [-] 1.0x [+]
        sb = 22
        sy = y + (bh - sb) // 2
        down = pygame.Rect(x, sy, sb, sb)
        theme.draw_button(self.screen, down, "-", self.font_small,
                          disabled=self.speed_idx == 0)
        self._buttons["speed_down"] = down
        x += sb + 2
        ts = self.font_small.render(f"{self.speed:g}x", True, COLOR_TEXT)
        self.screen.blit(ts, (x + (34 - ts.get_width()) // 2,
                              self.TOOLBAR_H // 2 - 2 - ts.get_height() // 2))
        x += 36
        up = pygame.Rect(x, sy, sb, sb)
        theme.draw_button(self.screen, up, "+", self.font_small,
                          disabled=self.speed_idx == len(self.SPEED_STEPS) - 1)
        self._buttons["speed_up"] = up
        x += sb + 14

        # Status indicator (text only when there is room).
        status = "PLAYING" if self.playing else "PAUSED"
        scol = COLOR_PLAYING if self.playing else COLOR_PAUSED
        cy = (self.TOOLBAR_H - 4) // 2
        pygame.draw.circle(self.screen, scol, (x + 7, cy), 6)
        if w >= 1000:
            st = self.font_big.render(status, True, scol)
            self.screen.blit(st, (x + 18, cy - st.get_height() // 2))

        # Right-aligned info string.
        alive = getattr(self.env, "agent_alive", None)
        alive_txt = ""
        if alive is not None:
            n_alive = int(sum(bool(a) for a in alive))
            alive_txt = f"alive {n_alive}/{self.env.n_agents}  ·  "
        mode_txt = (f"{self.FPS_SMOOTH}fps smooth" if self.smooth
                    else f"{self.FPS}fps stepped")
        info = (f"{alive_txt}step {int(getattr(self.env, 'step_count', 0))}"
                f"  ·  {mode_txt} x{self.speed:g}")
        it = self.font_small.render(
            info, True,
            COLOR_BROKEN_X if (alive is not None and not all(alive)) else COLOR_LABEL)
        self.screen.blit(it, (w - it.get_width() - 10, cy - it.get_height() // 2))

        # Episode progress strip along the toolbar's bottom edge.
        strip = pygame.Rect(0, self.TOOLBAR_H - 4, w, 4)
        pygame.draw.rect(self.screen, COLOR_BTN_DISABLED, strip)
        ms = getattr(self.env, "max_steps", None)
        if ms:
            denom = int(ms) * max(1, int(getattr(self.env, "n_agents", 1)))
            frac = min(1.0, int(getattr(self.env, "step_count", 0)) / max(1, denom))
            if frac < 0.8:
                fill = theme.ACCENT
            else:
                fill = theme.blend(theme.WARNING, theme.DANGER, (frac - 0.8) / 0.2)
            pygame.draw.rect(self.screen, fill,
                             pygame.Rect(0, strip.y, int(w * frac), 4))

    def _draw_transport_glyph(self, key, rect, active, disabled):
        col = (18, 24, 20) if active else (COLOR_LABEL if disabled else COLOR_TEXT)
        cx, cy = rect.centerx, rect.centery
        if key == "play":
            pygame.draw.polygon(self.screen, col,
                                [(cx - 4, cy - 6), (cx - 4, cy + 6), (cx + 6, cy)])
        elif key == "pause":
            pygame.draw.rect(self.screen, col, pygame.Rect(cx - 6, cy - 6, 4, 12))
            pygame.draw.rect(self.screen, col, pygame.Rect(cx + 2, cy - 6, 4, 12))
        else:  # step: triangle + bar
            pygame.draw.polygon(self.screen, col,
                                [(cx - 7, cy - 6), (cx - 7, cy + 6), (cx + 3, cy)])
            pygame.draw.rect(self.screen, col, pygame.Rect(cx + 5, cy - 6, 3, 12))

    def _draw_grid(self):
        env = self.env
        cs = self.CELL
        y_off = self.TOOLBAR_H
        now = self._now()
        view = effects.GridView(cs, y_off)
        grid_rect = pygame.Rect(0, y_off, env.W * cs, env.H * cs)

        if (self._terrain_dirty or self._terrain_surf is None
                or self._terrain_surf.get_size() != grid_rect.size):
            self._build_terrain()
        self.screen.blit(self._terrain_surf, (0, y_off))

        # Under-pass: translucent layers below the entity icons.
        overlay = self._get_overlay()
        overlay.fill((0, 0, 0, 0))
        overlay.set_clip(grid_rect)
        self.fx.draw_under(overlay, view, now)        # fog fade-ins
        if self.show_trails:
            self._draw_trails(overlay, view)
        if self.show_comm_range:
            self._draw_comm_ranges(overlay, cs, y_off)
        self._draw_links(overlay, view, now)
        overlay.set_clip(None)
        self.screen.blit(overlay, (0, 0))

        self._draw_target(cs, y_off, now)
        self._draw_nav_paths(cs, y_off)
        self._draw_entities(cs, y_off, now)

        # Over-pass: beacons, smoke, ripples and flashes above the entities.
        overlay.fill((0, 0, 0, 0))
        overlay.set_clip(grid_rect)
        self._draw_beacons(overlay, view, now)
        self.fx.draw_over(overlay, view, now)
        overlay.set_clip(None)
        self.screen.blit(overlay, (0, 0))

    def _build_terrain(self):
        """Rebuild the cached terrain layer (fog / explored / obstacles).

        Rebuilt only when the merged team knowledge changed (once per round)
        or on resize; per frame the whole grid is a single blit."""
        env = self.env
        H, W, cs = env.H, env.W, self.CELL
        size = (W * cs, H * cs)
        if self._terrain_surf is None or self._terrain_surf.get_size() != size:
            self._terrain_surf = pygame.Surface(size)
        surf = self._terrain_surf
        merged_vis, merged_obs = self._merged_maps()
        surf.fill(theme.FOG)
        hairline = cs >= 6
        bevel = cs >= 8
        for r in range(H):
            for c in range(W):
                x, y = c * cs, r * cs
                rect = pygame.Rect(x, y, cs, cs)
                if merged_obs[r, c] > 0:
                    pygame.draw.rect(surf, theme.OBSTACLE, rect)
                    if bevel:
                        pygame.draw.line(surf, theme.OBSTACLE_HI,
                                         (x + 1, y + 1), (x + cs - 2, y + 1))
                        pygame.draw.line(surf, theme.OBSTACLE_HI,
                                         (x + 1, y + 1), (x + 1, y + cs - 2))
                        pygame.draw.line(surf, theme.OBSTACLE_LO,
                                         (x + 1, y + cs - 2), (x + cs - 2, y + cs - 2))
                        pygame.draw.line(surf, theme.OBSTACLE_LO,
                                         (x + cs - 2, y + 1), (x + cs - 2, y + cs - 2))
                elif merged_vis[r, c] > 0:
                    pygame.draw.rect(surf, theme.FREE, rect)
                    if hairline:
                        pygame.draw.rect(surf, theme.FREE_GRID, rect, 1)
                elif hairline:
                    pygame.draw.rect(surf, theme.FOG_GRID, rect, 1)
        self._terrain_dirty = False

    def _get_overlay(self):
        """Reusable full-window SRCALPHA surface (avoids per-frame allocation)."""
        size = self.screen.get_size()
        if self._overlay is None or self._overlay.get_size() != size:
            self._overlay = pygame.Surface(size, pygame.SRCALPHA)
        return self._overlay

    def _display_positions(self):
        """Agent positions actually drawn (float grid coords; tweened in
        smooth mode). Snaps to the env state when unset or team size changed."""
        cur = [(float(r), float(c)) for r, c in self.env.agent_pos]
        if self._display_pos is None or len(self._display_pos) != len(cur):
            self._display_pos = cur
        return self._display_pos

    def _cell_center(self, r, c, cs, y_off):
        return int(c * cs + cs / 2), int(r * cs + y_off + cs / 2)

    def _draw_target(self, cs, y_off, now):
        """Ghost marker while undiscovered (spectator-only knowledge), full
        beacon core once any drone has spotted the target."""
        tr, tc = self.env.target_pos
        px, py = self._cell_center(tr, tc, cs, y_off)
        if self._target_discovered:
            icon = icons.get_icon("target_core", max(8, int(cs * 1.15)))
            icon.set_alpha(255)
        else:
            icon = icons.get_icon("target_ghost", max(8, int(cs * 1.05)))
            icon.set_alpha(120 + int(50 * math.sin(now / 480.0)))
        self.screen.blit(icon, (px - icon.get_width() // 2,
                                py - icon.get_height() // 2))

    def _draw_trails(self, overlay, view):
        """Fading dots over each drone's recently visited cells."""
        rad = max(2, view.cs // 5)
        for i, trail in enumerate(self._trails):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            n = len(trail)
            for k, cell in enumerate(trail):
                alpha = int(15 + 85 * (k + 1) / n)
                pygame.draw.circle(overlay, theme.with_alpha(color, alpha),
                                   view.center(cell), rad)

    def _draw_links(self, overlay, view, now):
        """Animated dashed link + traveling pulse between drones in contact."""
        pos = self._display_positions()
        for (i, j, _d) in self._active_pairs:
            if i >= len(pos) or j >= len(pos):
                continue
            if not (self._is_alive(i) and self._is_alive(j)):
                continue
            effects.draw_link(overlay, view.center(pos[i]), view.center(pos[j]),
                              AGENT_COLORS[i % len(AGENT_COLORS)],
                              AGENT_COLORS[j % len(AGENT_COLORS)],
                              view.cs, now)

    def _draw_beacons(self, overlay, view, now):
        """Wreck smoke + distress rings + knower pips, and the target beacon.

        Fault knowledge is read per drone (agent_known_crashed) — the beacon
        stays loud until every live drone knows about that crash."""
        env = self.env
        cs = view.cs
        crash_pos = getattr(env, "crash_pos", None) or {}
        known = getattr(env, "agent_known_crashed", None)
        alive_idx = [i for i in range(env.n_agents) if self._is_alive(i)]
        for k, (kr, kc) in crash_pos.items():
            center = view.center((kr, kc))
            effects.draw_smoke(overlay, center, cs, now, seed=int(k))
            knowers = ([i for i in alive_idx if k in known[i]]
                       if known is not None else alive_idx)
            if alive_idx and len(knowers) < len(alive_idx):
                # Active distress: loud double ring until the team knows.
                effects.draw_expanding_rings(
                    overlay, center, 0.3 * cs, 2.2 * cs, theme.DANGER,
                    1100, now, width=max(2, cs // 9), n_rings=2, alpha_max=190)
            else:
                # Acknowledged (or nobody left): quiet slow ring.
                effects.draw_expanding_rings(
                    overlay, center, 0.3 * cs, 1.4 * cs, theme.DANGER,
                    2200, now, width=2, n_rings=1, alpha_max=70)
            if cs >= 16 and alive_idx:
                self._draw_knower_pips(overlay, center, alive_idx, knowers, cs)
        if self._target_discovered:
            effects.draw_expanding_rings(
                overlay, view.center(tuple(env.target_pos)), 0.35 * cs,
                1.8 * cs, theme.TARGET, 1400, now, width=2, n_rings=2,
                alpha_max=150)

    def _draw_knower_pips(self, overlay, center, alive_idx, knowers, cs):
        """Tiny dots above a wreck: one per live teammate, filled if it knows
        about this crash, hollow if the distress beacon hasn't reached it."""
        cx, cy = center
        sp = max(6, int(cs * 0.30))
        rad = max(2, cs // 8)
        x0 = cx - sp * (len(alive_idx) - 1) / 2
        y = cy - int(cs * 0.95)
        for slot, i in enumerate(alive_idx):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            p = (int(x0 + slot * sp), y)
            if i in knowers:
                pygame.draw.circle(overlay, theme.with_alpha(color, 235), p, rad)
            else:
                pygame.draw.circle(overlay, theme.with_alpha((15, 16, 22), 200), p, rad)
                pygame.draw.circle(overlay, theme.with_alpha(color, 160), p, rad, 1)

    def _draw_entities(self, cs, y_off, now):
        env = self.env
        pos = self._display_positions()
        for i, (r, c) in enumerate(pos):
            px, py = self._cell_center(r, c, cs, y_off)
            broken = not self._is_alive(i)
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            if cs >= 12:
                isz = int(cs * 1.3)
                phase = 0 if broken else int((now // 40) % icons.DRONE_PHASES)
                icon = icons.get_icon("wreck" if broken else "drone",
                                      isz, color, phase=phase)
                self.screen.blit(icon, (px - isz // 2, py - isz // 2))
                if not broken and i < len(self._heading) and self._heading[i]:
                    self._draw_heading_marker(px, py, self._heading[i], cs, color)
            else:
                # Tiny cells: fall back to the plain circle (+ X for wrecks).
                rad = max(3, cs // 2 - 2)
                pygame.draw.circle(self.screen,
                                   COLOR_BROKEN if broken else color, (px, py), rad)
                if broken:
                    lw = max(2, cs // 8)
                    pygame.draw.line(self.screen, COLOR_BROKEN_X,
                                     (px - rad, py - rad), (px + rad, py + rad), lw)
                    pygame.draw.line(self.screen, COLOR_BROKEN_X,
                                     (px - rad, py + rad), (px + rad, py - rad), lw)
            if cs >= 14:
                label = icons.get_label(str(i), cs)
                self.screen.blit(label, (px - label.get_width() // 2,
                                         py - label.get_height() // 2))

    def _draw_heading_marker(self, px, py, head, cs, color):
        """Small triangle at the body edge pointing along the last move."""
        dr, dc = head
        n = (dr * dr + dc * dc) ** 0.5
        if n == 0:
            return
        ux, uy = dc / n, dr / n
        tip = (px + ux * cs * 0.62, py + uy * cs * 0.62)
        left = (px + ux * cs * 0.34 - uy * cs * 0.16,
                py + uy * cs * 0.34 + ux * cs * 0.16)
        right = (px + ux * cs * 0.34 + uy * cs * 0.16,
                 py + uy * cs * 0.34 - ux * cs * 0.16)
        pygame.draw.polygon(self.screen, theme.brighten(color, 45),
                            [tip, left, right])

    def _draw_nav_paths(self, cs, y_off):
        """Overlay the BFS auto-nav path of each homing drone (set by the
        eval/simulate loop on env.debug_nav_paths). Drawn under the markers."""
        paths = getattr(self.env, "debug_nav_paths", None)
        if not paths:
            return
        for i, path in paths.items():
            if not path or len(path) < 2:
                continue
            pts = [(c * cs + cs // 2, r * cs + y_off + cs // 2) for (r, c) in path]
            pygame.draw.lines(self.screen, COLOR_NAV_PATH, False, pts, max(2, cs // 8))
            # Ring around the homing drone so the mode switch is visible.
            pygame.draw.circle(self.screen, COLOR_NAV_PATH, pts[0], max(4, cs // 2), 2)

    def _draw_comm_ranges(self, overlay, cs, y_off):
        """Translucent comm-range shapes on the shared overlay — a diamond for
        the Manhattan metric, a circle for Euclidean. Toggled with (C) / the
        Comm chip. Drawn under the drone markers."""
        env = self.env
        rng = int(getattr(env, "comm_range", 0))
        if rng <= 0:
            return
        metric = getattr(env, "comm_metric", "manhattan")
        for i, (r, c) in enumerate(self._display_positions()):
            if not self._is_alive(i):
                continue                 # wrecks have no radio
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            cx, cy = self._cell_center(r, c, cs, y_off)
            fill = (color[0], color[1], color[2], 32)
            line = (color[0], color[1], color[2], 150)
            R = rng * cs
            if metric == "manhattan":
                pts = [(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy)]
                pygame.draw.polygon(overlay, fill, pts)
                pygame.draw.polygon(overlay, line, pts, 2)
            else:
                pygame.draw.circle(overlay, fill, (cx, cy), R)
                pygame.draw.circle(overlay, line, (cx, cy), R, 2)

    # --- Sidebar ----------------------------------------------------------

    def _layout_sidebar(self, x, y, w, h):
        HDR = 22
        out = {}
        mini_h = self.MINIMAP_BODY if self.sections["minimaps"] else 0
        log_h  = self.LOG_BODY if self.sections["log"] else 0
        fixed = 3 * HDR + mini_h + log_h
        charts_h = max(0, (h - fixed)) if self.sections["charts"] else 0

        cy = y
        out["minimaps_header"] = pygame.Rect(x, cy, w, HDR); cy += HDR
        if self.sections["minimaps"]:
            out["minimaps_body"] = pygame.Rect(x, cy, w, mini_h); cy += mini_h
        out["charts_header"] = pygame.Rect(x, cy, w, HDR); cy += HDR
        if self.sections["charts"]:
            out["charts_body"] = pygame.Rect(x, cy, w, charts_h); cy += charts_h
        out["log_header"] = pygame.Rect(x, cy, w, HDR); cy += HDR
        if self.sections["log"]:
            out["log_body"] = pygame.Rect(x, cy, w, log_h); cy += log_h
        return out

    def _draw_sidebar(self):
        env = self.env
        sx = env.W * self.CELL
        sy = self.TOOLBAR_H
        sw = self.screen.get_width() - sx
        sh = self.screen.get_height() - sy
        pygame.draw.rect(self.screen, COLOR_SIDEBAR_BG, pygame.Rect(sx, sy, sw, sh))
        pygame.draw.line(self.screen, COLOR_BORDER, (sx, sy), (sx, sy + sh), 1)

        layout = self._layout_sidebar(sx + 4, sy + 4, sw - 8, sh - 8)
        self._section_headers = {}

        self._draw_section_header(layout["minimaps_header"], "A.  Drone mini-maps", "minimaps")
        if "minimaps_body" in layout:
            self._draw_minimaps(layout["minimaps_body"])

        self._draw_section_header(layout["charts_header"], "B.  Real-time charts", "charts")
        if "charts_body" in layout:
            self._draw_charts(layout["charts_body"])

        self._draw_section_header(layout["log_header"], "C.  Episode log", "log")
        if "log_body" in layout:
            self._log_rect = layout["log_body"]
            self.log.render(self.screen, self._log_rect, self.font_mono)
        else:
            self._log_rect = None

    def _draw_section_header(self, rect, title, key):
        expanded = self.sections[key]
        pygame.draw.rect(self.screen, COLOR_SECTION_HDR, rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, rect, 1)
        arrow = "v" if expanded else ">"
        self.screen.blit(self.font_small.render(f"[{arrow}] {title}", True, COLOR_TEXT),
                         (rect.x + 6, rect.y + 4))
        self._section_headers[key] = rect

    def _draw_minimaps(self, body):
        env = self.env
        pad = 6
        cw = (body.width - pad * 3) // 2
        ch = (body.height - pad * 3) // 2
        for slot in range(MINIMAP_SLOTS):
            gr, gc = divmod(slot, 2)
            x = body.x + pad + gc * (cw + pad)
            y = body.y + pad + gr * (ch + pad)
            cell = pygame.Rect(x, y, cw, ch)
            if slot >= env.n_agents:
                pygame.draw.rect(self.screen, COLOR_INACTIVE_PANEL, cell)
                pygame.draw.rect(self.screen, COLOR_BORDER, cell, 1)
                t = self.font_small.render("-- inactive --", True, (120, 120, 120))
                self.screen.blit(t, (cell.centerx - t.get_width() // 2, cell.centery - 7))
            else:
                self._draw_one_minimap(cell, slot)

    def _draw_one_minimap(self, cell, i):
        env = self.env
        H, W = env.H, env.W
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        # Inner drawing area, leaving the top strip for the label.
        inner = pygame.Rect(cell.x + 3, cell.y + 14, cell.width - 6, cell.height - 17)
        csz = max(1, min(inner.width // W, inner.height // H))
        ox = inner.x + (inner.width - csz * W) // 2
        oy = inner.y + (inner.height - csz * H) // 2

        pygame.draw.rect(self.screen, (10, 10, 12), pygame.Rect(ox, oy, csz * W, csz * H))

        vis = env.agent_visited[i]
        obs = env.agent_obstacle[i]
        tgt = env.agent_target[i]

        for r, c in zip(*np.where(obs > 0)):
            pygame.draw.rect(self.screen, COLOR_OBSTACLE,
                             pygame.Rect(ox + c * csz, oy + r * csz, csz, csz))
        for r, c in zip(*np.where(vis > 0)):
            pygame.draw.rect(self.screen, COLOR_FREE,
                             pygame.Rect(ox + c * csz, oy + r * csz, csz, csz))
        for r, c in zip(*np.where(tgt > 0)):
            pygame.draw.rect(self.screen, COLOR_TARGET,
                             pygame.Rect(ox + c * csz, oy + r * csz, csz, csz))

        # Crash sites this drone knows about (its own Broken map).
        brk = getattr(env, "agent_broken", None)
        if brk is not None:
            for r, c in zip(*np.where(brk[i] > 0)):
                pygame.draw.rect(self.screen, COLOR_BROKEN_X,
                                 pygame.Rect(ox + c * csz, oy + r * csz, csz, csz))

        broken = not self._is_alive(i)
        r, c = env.agent_pos[i]
        pygame.draw.circle(self.screen, COLOR_BROKEN if broken else color,
                           (ox + c * csz + csz // 2, oy + r * csz + csz // 2), max(2, csz))

        if broken:
            border_col = COLOR_BROKEN_X
        elif self._comm_flash_until.get(i, 0) > self._now():
            border_col = COLOR_COMM_FLASH
        else:
            border_col = color
        pygame.draw.rect(self.screen, border_col, cell, 2)
        tag = f"D{i} BROKEN" if broken else f"D{i}"
        self.screen.blit(
            self.font_small.render(tag, True, COLOR_BROKEN_X if broken else color),
            (cell.x + 4, cell.y + 1),
        )

    def _draw_charts(self, body):
        n_charts = 5
        gap = 4
        ch = max(28, (body.height - gap * (n_charts + 1)) // n_charts)
        x = body.x + 4
        w = body.width - 8
        y = body.y + gap

        # 1 - per-drone step reward (one line per drone)
        series = [(AGENT_COLORS[i % len(AGENT_COLORS)], self._hist_drone_reward[i])
                  for i in range(len(self._hist_drone_reward))]
        legend = [(AGENT_COLORS[i % len(AGENT_COLORS)], f"D{i}")
                  for i in range(len(self._hist_drone_reward))]
        charts.draw_line_chart(self.screen, pygame.Rect(x, y, w, ch), series, self.font_tiny,
                               title="Per-drone step reward", window=200, legend=legend)
        y += ch + gap

        # 2 - episode cumulative reward
        charts.draw_line_chart(self.screen, pygame.Rect(x, y, w, ch),
                               [((220, 220, 220), self._hist_cum_reward)], self.font_tiny,
                               title="Episode cumulative reward", window=400)
        y += ch + gap

        # 3 - coverage %
        charts.draw_line_chart(self.screen, pygame.Rect(x, y, w, ch),
                               [((120, 200, 255), self._hist_coverage)], self.font_tiny,
                               title="Coverage %", window=400, y_min=0, y_max=100, hline=100)
        y += ch + gap

        # 4 - new cells explored per step (green bars)
        charts.draw_bar_chart(self.screen, pygame.Rect(x, y, w, ch), self._hist_new_cells,
                              self.font_tiny, title="New cells / step", window=100,
                              color=COLOR_REWARD_POS)
        y += ch + gap

        # 5 - communications per step (yellow bars)
        charts.draw_bar_chart(self.screen, pygame.Rect(x, y, w, ch), self._hist_comms,
                              self.font_tiny, title="Communications / step", window=100,
                              color=COLOR_COMM_FLASH)

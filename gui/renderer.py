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

import numpy as np

try:
    import pygame
    from gui import charts
    from gui import terminal as terminal_widget
    PYGAME_AVAILABLE = True
except ImportError:                      # pragma: no cover - exercised only without pygame
    PYGAME_AVAILABLE = False
    charts = None
    terminal_widget = None

# --- Map colours -----------------------------------------------------------
COLOR_UNKNOWN  = (30, 30, 30)
COLOR_FREE     = (200, 200, 200)
COLOR_OBSTACLE = (60, 60, 60)
COLOR_TARGET   = (255, 80, 80)
COLOR_GRID_LINE = (100, 100, 100)
COLOR_BG       = (15, 15, 15)
COLOR_TEXT     = (240, 240, 240)
COLOR_COMM_FLASH = (255, 220, 0)
COLOR_REWARD_POS = (80, 220, 120)
COLOR_REWARD_NEG = (220, 80, 80)

# --- Chrome colours --------------------------------------------------------
COLOR_TOOLBAR_BG    = (28, 28, 34)
COLOR_SIDEBAR_BG    = (18, 18, 22)
COLOR_BORDER        = (60, 60, 66)
COLOR_SECTION_HDR   = (40, 40, 48)
COLOR_BTN           = (55, 55, 64)
COLOR_BTN_ACTIVE    = (90, 200, 120)
COLOR_BTN_DISABLED  = (38, 38, 42)
COLOR_PLAYING       = (90, 220, 120)
COLOR_PAUSED        = (240, 190, 60)
COLOR_INACTIVE_PANEL = (34, 34, 38)
COLOR_LABEL         = (140, 140, 140)

# --- Episode-log line colours ---------------------------------------------
COLOR_LOG_HEADER = (240, 240, 240)
COLOR_LOG_NORMAL = (180, 180, 180)
COLOR_LOG_COMM   = (255, 220, 0)
COLOR_LOG_TARGET = (255, 120, 40)
COLOR_LOG_NEG    = (210, 140, 140)

AGENT_COLORS = [
    (80, 160, 255),
    (80, 255, 160),
    (255, 160, 80),
    (200, 80, 255),
    (255, 80, 200),
]

_SCREEN_MARGIN = 80
COMM_FLASH_FRAMES = 4
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
    FPS = 10

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

        # Collapsible sidebar sections (all expanded by default)
        self.sections = {"minimaps": True, "charts": True, "log": True}

        # Widgets / per-frame UI bookkeeping
        self.log = terminal_widget.TerminalLog()
        self._comm_flash = {}
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
        self.font_big   = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12)
        self.font_mono  = pygame.font.SysFont("monospace", 12)
        self.font_tiny  = pygame.font.SysFont("monospace", 10)
        self.log.line_h = 14

    # ------------------------------------------------------------------
    # Public per-frame entry points
    # ------------------------------------------------------------------

    def render(self):
        """Draw one human-mode frame. Blocks while PAUSED until Play / Step."""
        if self.headless or self._closed:
            return
        self._tick_data()
        self._handle_events()
        if self._closed:
            return

        # Pause gate: keep the window alive without advancing the simulation.
        while not self.playing and not self._closed:
            self._draw()
            self._decay_flash()
            pygame.display.flip()
            self.clock.tick(self.FPS)
            self._handle_events()
            if self.step_pending:        # single-step requested -> let caller advance once
                self.step_pending = False
                break
        if self._closed:
            return

        self._draw()
        self._decay_flash()
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def get_rgb_array(self):
        """Render to the offscreen surface and return an (H, W, 3) array."""
        self._tick_data()
        self._draw()
        self._decay_flash()
        return np.transpose(np.array(pygame.surfarray.array3d(self.screen)), axes=(1, 0, 2))

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
        self._comm_flash         = {}
        self._found_logged       = False

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
            self._snapshot(step)
            self._record_sample(step)
            self._prev_step = step
            return

        if step == self._prev_step:
            return                       # no new env step since last frame

        # One or more rounds happened since the last frame.
        self._log_round(step)
        self._record_sample(step)
        if getattr(env, "done", False) and not self._episode_ended:
            self._log_episode_end()
            self._episode_ended = True
        self._snapshot(step)
        self._prev_step = step

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

    def _comm_pairs(self):
        """Pairs of drones currently within comm range (mirrors env._communicate)."""
        env = self.env
        rng = getattr(env, "comm_range", getattr(env, "comm_radius", 0))
        metric = getattr(env, "comm_metric", "manhattan")
        pairs = []
        n = env.n_agents
        for i in range(n):
            ri, ci = env.agent_pos[i]
            for j in range(i + 1, n):
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

        for i in range(env.n_agents):
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
        for (i, j, d) in pairs:
            if (i, j) not in self._prev_comm_pairs:
                self.log.add(
                    f"[Step {step:4d}] [comm] D{i} <-> D{j} communicated  (distance={d:.1f})",
                    COLOR_LOG_COMM,
                )
        for (i, j, _d) in pairs:
            self._comm_flash[i] = COMM_FLASH_FRAMES
            self._comm_flash[j] = COMM_FLASH_FRAMES

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
        rule = "=" * 34
        self.log.add_block([
            rule,
            " Episode start",
            f" Grid: {env.H}x{env.W}   Agents: {env.n_agents}",
            f" Vision radius: {vr}",
            f" Comm radius:   {cr}",
            f" Obstacle density: {od:.2f}",
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
                elif event.key == pygame.K_RIGHT and not self.playing:
                    self.step_pending = True
                elif event.key == pygame.K_c:
                    self.show_comm_range = not self.show_comm_range
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
                elif key == "pause":
                    self.playing = False
                elif key == "step" and not self.playing:
                    self.step_pending = True
                elif key == "comm":
                    self.show_comm_range = not self.show_comm_range
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

    def _decay_flash(self):
        for k in list(self._comm_flash.keys()):
            if self._comm_flash[k] > 0:
                self._comm_flash[k] -= 1

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_toolbar()
        self._draw_grid()
        self._draw_sidebar()

    def _draw_toolbar(self):
        w = self.screen.get_width()
        bar = pygame.Rect(0, 0, w, self.TOOLBAR_H)
        pygame.draw.rect(self.screen, COLOR_TOOLBAR_BG, bar)
        pygame.draw.line(self.screen, COLOR_BORDER,
                         (0, self.TOOLBAR_H - 1), (w, self.TOOLBAR_H - 1), 1)

        self._buttons = {}
        x, y = 8, 7
        bh = self.TOOLBAR_H - 14
        glyphs = {"play": "> Play", "pause": "|| Pause", "step": ">| Step"}
        for key in ("play", "pause", "step"):
            rect = pygame.Rect(x, y, 80, bh)
            active = (key == "play" and self.playing) or (key == "pause" and not self.playing)
            disabled = (key == "step" and self.playing)
            col = COLOR_BTN_ACTIVE if active else (COLOR_BTN_DISABLED if disabled else COLOR_BTN)
            pygame.draw.rect(self.screen, col, rect)
            pygame.draw.rect(self.screen, COLOR_BORDER, rect, 1)
            tcol = (20, 20, 20) if active else (COLOR_LABEL if disabled else COLOR_TEXT)
            self.screen.blit(self.font_small.render(glyphs[key], True, tcol),
                             (rect.x + 7, rect.y + (bh - 12) // 2))
            self._buttons[key] = rect
            x += 86

        # Comm-range toggle button
        comm_rect = pygame.Rect(x, y, 88, bh)
        on = self.show_comm_range
        ccol = COLOR_BTN_ACTIVE if on else COLOR_BTN
        pygame.draw.rect(self.screen, ccol, comm_rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, comm_rect, 1)
        ctcol = (20, 20, 20) if on else COLOR_TEXT
        self.screen.blit(self.font_small.render("(C) Comm", True, ctcol),
                         (comm_rect.x + 7, comm_rect.y + (bh - 12) // 2))
        self._buttons["comm"] = comm_rect
        x += 94

        # Status indicator
        status = "PLAYING" if self.playing else "PAUSED"
        scol = COLOR_PLAYING if self.playing else COLOR_PAUSED
        cx = x + 12
        pygame.draw.circle(self.screen, scol, (cx, self.TOOLBAR_H // 2), 7)
        self.screen.blit(self.font_big.render(status, True, scol),
                         (cx + 14, (self.TOOLBAR_H - 15) // 2))

        info = f"FPS {self.FPS}   step {int(getattr(self.env, 'step_count', 0))}"
        iw = self.font_small.size(info)[0]
        self.screen.blit(self.font_small.render(info, True, COLOR_LABEL),
                         (w - iw - 10, (self.TOOLBAR_H - 12) // 2))

    def _draw_grid(self):
        env = self.env
        H, W, cs = env.H, env.W, self.CELL
        y_off = self.TOOLBAR_H
        merged_vis, merged_obs = self._merged_maps()

        for r in range(H):
            for c in range(W):
                x, y = c * cs, r * cs + y_off
                rect = pygame.Rect(x, y, cs, cs)
                if merged_obs[r, c] > 0:
                    color = COLOR_OBSTACLE
                elif merged_vis[r, c] > 0:
                    color = COLOR_FREE
                else:
                    color = COLOR_UNKNOWN
                pygame.draw.rect(self.screen, color, rect)
                if cs >= 6:
                    pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)

        tr, tc = env.target_pos
        margin = max(2, cs // 6)
        pygame.draw.rect(
            self.screen, COLOR_TARGET,
            pygame.Rect(tc * cs + margin, tr * cs + y_off + margin,
                        cs - 2 * margin, cs - 2 * margin),
        )

        if self.show_comm_range:
            self._draw_comm_ranges(cs, y_off)

        for i, (r, c) in enumerate(env.agent_pos):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            if self._comm_flash.get(i, 0) > 0:
                color = COLOR_COMM_FLASH
            cx = c * cs + cs // 2
            cy = r * cs + y_off + cs // 2
            pygame.draw.circle(self.screen, color, (cx, cy), max(3, cs // 2 - 2))
            if cs >= 14:
                label = self.font_small.render(str(i), True, (0, 0, 0))
                self.screen.blit(label, (cx - 5, cy - 7))

    def _draw_comm_ranges(self, cs, y_off):
        """Translucent overlay of each drone's communication range — a diamond for
        the Manhattan metric, a circle for Euclidean. Toggled with (C) / the
        Comm button. Drawn under the drone markers."""
        env = self.env
        rng = int(getattr(env, "comm_range", 0))
        if rng <= 0:
            return
        metric = getattr(env, "comm_metric", "manhattan")
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for i, (r, c) in enumerate(env.agent_pos):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            cx = c * cs + cs // 2
            cy = r * cs + y_off + cs // 2
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
        self.screen.blit(overlay, (0, 0))

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

        r, c = env.agent_pos[i]
        pygame.draw.circle(self.screen, color,
                           (ox + c * csz + csz // 2, oy + r * csz + csz // 2), max(2, csz))

        border_col = COLOR_COMM_FLASH if self._comm_flash.get(i, 0) > 0 else color
        pygame.draw.rect(self.screen, border_col, cell, 2)
        self.screen.blit(self.font_small.render(f"D{i}", True, color), (cell.x + 4, cell.y + 1))

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

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.utils import generate_grid
from collections import defaultdict

# ---------------------------------------------------------------------------
# Channel counts (must match agents/dqn_agent.py)
# ---------------------------------------------------------------------------
# Global stream: coarse spatial memory for the whole map
#   0  Visited      – cells seen as free by this drone (or comm-shared)
#   1  Obstacle     – discovered obstacles
#   2  Trajectory   – normalised visit counts (cap at _TRAJ_CAP)
#   3  Target       – target position if ever seen
#   4  Own_Position – this drone's own cell (single 1.0); the network reads it
#                     back (argmax) to crop the local patch
#   5  Broken       – crash sites of broken teammates KNOWN to this drone
#                     (learned via distress beacon / comm fusion, not an oracle)
GLOBAL_CHANNELS = 6

# Index of the Broken channel within the global map.
BROKEN_CHANNEL = 5

# Local stream: fine-grained detail fed to the local CNN
#   0  Visited
#   1  Obstacle
#   2  Trajectory
#   3  Target
#   4  Other_Position – 1 at cells occupied by other drones within vision
LOCAL_CHANNELS  = 5

# Movement actions
ACTIONS    = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTIONS_NP = np.array(ACTIONS, dtype=np.int32)
N_ACTIONS  = len(ACTIONS)

# Fixed cap for trajectory normalisation.
_TRAJ_CAP = 5.0

# Four spawn corners
_SPAWN_CORNERS = [(0, 0), (0, -1), (-1, 0), (-1, -1)]


class DroneSearchEnv(gym.Env):
    """
    Sequential-execution multi-drone search on a fixed 32x32 grid.

    Observation (per agent): dict with two flat arrays
    -------------------------------------------------------
      'global' : shape (GLOBAL_CHANNELS * grid_size^2,)  = (6 * 1024,) = (6144,)
                 channels: [Visited, Obstacle, Trajectory, Target, Own_Position,
                            Broken]
      'local'  : shape (LOCAL_CHANNELS  * grid_size^2,)  = (5 * 1024,) = (5120,)
                 channels: [Visited, Obstacle, Trajectory, Target, Other_Position]

    The drone's own position IS encoded, as the Own_Position global channel
    (a single 1.0 at env.agent_pos[i]). The network recovers the crop
    coordinates from that channel, so position is a spatial signal rather than
    a context scalar.

    Fault model (random drone failures)
    -----------------------------------
    At the start of its turn each live drone breaks with probability
    ``fault_prob`` (per step). A broken drone stops for the rest of the
    episode: it no longer moves, senses, or communicates; its turns become
    no-ops (the global clock still ticks, so the time budget is unaffected).
    The wreck does NOT block movement and is excluded from collisions and
    from the Other_Position channel.

    Fault knowledge is decentralized: the wreck emits a distress beacon.
    A live drone passing within ``comm_range`` of the crash site (or seeing
    it inside its vision box) records it in its OWN Broken map, and that
    knowledge then spreads through the usual comm map-fusion — exactly like
    Visited/Obstacle/Target knowledge. ``n_alive_belief(i)`` exposes the
    team size drone ``i`` believes is still operational.

    When every drone is broken the episode is truncated early (non-terminal).

    Reward per agent step:
      +target_reward         if this drone reaches the target
      +exploration_bonus     for newly discovered cells
      -revisit_penalty * k   linear in the prior visit count k of the entered
                             cell; the cumulative cost of repeatedly revisiting
                             a cell therefore grows quadratically, discouraging
                             oscillation loops
      -step_penalty          every move
      -wall_penalty          if invalid move
      -collision_penalty     if two or more drones occupy the same cell
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict, render_mode: str = None):
        super().__init__()
        env_cfg = config["env"]

        # Single source of truth: fixed grid size
        self.grid_size         = env_cfg["grid_size"]
        self.obstacle_density  = env_cfg["obstacle_density"]
        self.n_agents          = env_cfg["n_agents"]
        self.vision_radius     = env_cfg["vision_radius"]
        self.comm_range        = env_cfg["comm_range"]
        self.comm_metric       = env_cfg["comm_metric"]
        self.max_steps         = env_cfg["max_steps"]
        self.step_penalty      = env_cfg["step_penalty"]
        self.wall_penalty      = env_cfg.get("wall_penalty",      -0.02)
        self.revisit_penalty   = env_cfg.get("revisit_penalty",   -0.05)
        self.target_reward     = env_cfg["target_reward"]
        self.exploration_bonus = env_cfg.get("exploration_bonus",  0.05)
        self.collision_penalty = env_cfg.get("collision_penalty",  -0.5)
        # Per-step probability that a live drone breaks at the start of its
        # turn (0 disables faults). Randomized per episode via DR in training;
        # set a fixed value from CLI/GUI in eval/simulate.
        self.fault_prob        = env_cfg.get("fault_prob", 0.0)
        # Optional fixed spawn corner index (0=top-left .. 3=bottom-right); None
        # keeps the per-episode random corner used during training.
        self.spawn_corner      = env_cfg.get("spawn_corner", None)
        self.seed_val          = env_cfg["seed"]

        # H and W are always grid_size; kept as explicit attributes for
        # renderer and utility functions that reference env.H / env.W
        self.H = self.grid_size
        self.W = self.grid_size
        mg     = self.grid_size

        # Observation space: Dict matching the {'global', 'local'} dict that
        # _get_obs / reset / step_agent actually return (per-agent).
        global_dim = GLOBAL_CHANNELS * mg * mg
        local_dim  = LOCAL_CHANNELS  * mg * mg
        self.observation_space = spaces.Dict({
            "global": spaces.Box(low=0.0, high=1.0, shape=(global_dim,), dtype=np.float32),
            "local":  spaces.Box(low=0.0, high=1.0, shape=(local_dim,),  dtype=np.float32),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Precompute vision offsets for the current vision_radius.
        self._compute_vision_offsets()

        self.render_mode = render_mode
        self.renderer    = None

        # State (populated in reset)
        self.grid             = None
        self.target_pos       = None
        self.agent_pos        = None
        self.agent_visited    = None
        self.agent_obstacle   = None
        self.agent_target     = None
        self.agent_trajectory = None
        self.agent_alive      = None   # (na,) bool — False once a drone broke
        self.crash_pos        = None   # dict idx -> (r, c) ground-truth wrecks
        self.agent_broken     = None   # (na, H, W) crash sites KNOWN per drone
        self.agent_known_crashed = None  # list[set[int]] crashed ids per drone
        self._known_mask      = None
        self.step_count       = 0
        self._current_agent   = 0      # round-robin pointer for the gym step()
        self.done             = False
        self.last_rewards     = [0.0] * self.n_agents
        self.episode_reward   = 0.0
        self.rng              = np.random.default_rng(self.seed_val)

    # ------------------------------------------------------------------
    # Domain-randomisation parameters
    # ------------------------------------------------------------------

    def _compute_vision_offsets(self):
        """Precompute the (dr, dc) offsets covering the current vision_radius."""
        vr = self.vision_radius
        dr_grid, dc_grid = np.meshgrid(
            np.arange(-vr, vr + 1), np.arange(-vr, vr + 1), indexing="ij"
        )
        self._vis_dr = dr_grid.ravel()
        self._vis_dc = dc_grid.ravel()

    def set_domain_params(self, vision_radius: int = None,
                          comm_range: int = None, n_agents: int = None,
                          obstacle_density: float = None,
                          fault_prob: float = None):
        """Set the per-episode domain-randomisation parameters.

        Call this BEFORE ``reset()`` so the next episode's observations are
        actually generated with these values — keeping what the network observes
        consistent with the context vector it is given. ``reset()`` reallocates
        the per-agent map arrays from ``self.n_agents``, so changing the agent
        count here is safe.

        Args:
            vision_radius: New field-of-view radius (recomputes vision offsets).
            comm_range: New communication range for map fusion.
            n_agents: New number of drones for the next episode.
            obstacle_density: New obstacle density for the next generated grid.
            fault_prob: New per-step drone fault probability.
        """
        if vision_radius is not None and vision_radius != self.vision_radius:
            self.vision_radius = int(vision_radius)
            self._compute_vision_offsets()
        if comm_range is not None:
            self.comm_range = int(comm_range)
        if n_agents is not None:
            self.n_agents = int(n_agents)
        if obstacle_density is not None:
            self.obstacle_density = float(obstacle_density)
        if fault_prob is not None:
            self.fault_prob = float(fault_prob)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mg = self.grid_size
        na = self.n_agents

        self.grid, self.target_pos = generate_grid(
            self.H, self.W, self.obstacle_density, self.rng
        )

        # Pick one of the four corners at random each episode.
        # Negative indices wrap to the last row/column (gs-1).
        corner_idx = self.spawn_corner if self.spawn_corner is not None \
            else int(self.rng.integers(0, 4))
        cr, cc = _SPAWN_CORNERS[corner_idx]
        start = (int(cr % mg), int(cc % mg))
        self.agent_pos = [start for _ in range(na)]

        self.agent_visited    = np.zeros((na, mg, mg), dtype=np.float32)
        self.agent_obstacle   = np.zeros((na, mg, mg), dtype=np.float32)
        self.agent_target     = np.zeros((na, mg, mg), dtype=np.float32)
        self.agent_trajectory = np.zeros((na, mg, mg), dtype=np.float32)

        # Fault state: everyone starts operational, no known crash sites.
        self.agent_alive         = np.ones(na, dtype=bool)
        self.crash_pos           = {}
        self.agent_broken        = np.zeros((na, mg, mg), dtype=np.float32)
        self.agent_known_crashed = [set() for _ in range(na)]

        self._known_mask    = np.zeros((self.H, self.W), dtype=bool)
        self.step_count     = 0
        self._current_agent = 0
        self.done           = False
        self.last_rewards   = [0.0] * na
        self.episode_reward = 0.0

        for i in range(na):
            r, c = self.agent_pos[i]
            self.agent_trajectory[i, r, c] = 1.0
            self._update_map(i)
        self._communicate()
        self._sync_known_mask()

        obs   = [self._get_obs(i)         for i in range(na)]
        masks = [self._get_action_mask(i) for i in range(na)]
        info  = {
            "H": self.H, "W": self.W,
            "target": self.target_pos,
            "action_masks": masks,
            "found": False,
            "step": 0,
            "n_alive": na,
        }
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_agent(self, agent_idx: int, action: int):
        assert not self.done, "Episode ended – call reset()"

        # --- Random fault model ------------------------------------------
        # A broken drone's turn is a no-op (it cannot move, sense, or
        # communicate) but the clock still ticks — losing drones does not
        # buy the team extra time.
        if not self.agent_alive[agent_idx]:
            return self._noop_turn(agent_idx, just_broke=False)

        # A live drone may break at the start of its turn, before moving.
        # Drawn from self.rng AFTER map generation, so a fixed reset seed
        # reproduces both the layout and the fault sequence.
        if self.fault_prob > 0 and self.rng.random() < self.fault_prob:
            self.agent_alive[agent_idx] = False
            self.crash_pos[agent_idx]   = self.agent_pos[agent_idx]
            return self._noop_turn(agent_idx, just_broke=True)

        dr, dc = ACTIONS[action]
        r, c   = self.agent_pos[agent_idx]
        nr, nc = r + dr, c + dc

        reward = self.step_penalty
        valid  = (
            0 <= nr < self.H and 0 <= nc < self.W
            and self.grid[nr, nc] != 1
        )

        if not valid:
            reward += self.wall_penalty
        else:
            prev_count = self.agent_trajectory[agent_idx, nr, nc]
            if prev_count > 0:
                reward += self.revisit_penalty * prev_count
            self.agent_pos[agent_idx] = (nr, nc)
            self.agent_trajectory[agent_idx, nr, nc] += 1.0

            # Wrecks are on the ground and do not collide with flying drones.
            collision_count = sum(
                1 for k in range(self.n_agents)
                if k != agent_idx and self.agent_alive[k]
                and self.agent_pos[k] == self.agent_pos[agent_idx]
            )
            if collision_count > 0:
                reward += self.collision_penalty * collision_count

        old_known = self._known_mask.copy()
        self._update_map(agent_idx)
        self._detect_wrecks(agent_idx)
        self._communicate()
        self._sync_known_mask()

        new_cells = int((self._known_mask & ~old_known).sum())
        reward   += self.exploration_bonus * new_cells

        self.step_count += 1

        terminated = (self.agent_pos[agent_idx] == self.target_pos)
        if terminated:
            reward += self.target_reward

        truncated   = self.step_count >= self.max_steps * self.n_agents
        self.done   = terminated or truncated

        self.last_rewards[agent_idx]  = reward
        self.episode_reward          += reward

        obs_i  = self._get_obs(agent_idx)
        mask_i = self._get_action_mask(agent_idx)
        info = {
            "step":            self.step_count,
            "found":           terminated,
            "new_cells":       new_cells,
            "agent_positions": list(self.agent_pos),
            "known_cells":     int(self._known_mask.sum()),
            "action_mask":     mask_i,
            "broken":          False,
            "just_broke":      False,
            "n_alive":         int(self.agent_alive.sum()),
        }
        return obs_i, reward, terminated, truncated, info

    def _noop_turn(self, agent_idx: int, just_broke: bool):
        """Consume a broken drone's turn: only the global clock advances.

        Returns the standard 5-tuple with reward 0 and terminated=False —
        a fault is a truncation-like event for that drone, never a terminal
        one (it is independent of state/action, so its future value must
        still be bootstrapped by learners). Once every drone is down the
        episode is truncated early: nothing can change anymore.
        """
        self.step_count += 1
        truncated = (
            self.step_count >= self.max_steps * self.n_agents
            or not self.agent_alive.any()
        )
        self.done = self.done or truncated
        self.last_rewards[agent_idx] = 0.0

        obs_i = self._get_obs(agent_idx)
        info = {
            "step":            self.step_count,
            "found":           False,
            "new_cells":       0,
            "agent_positions": list(self.agent_pos),
            "known_cells":     int(self._known_mask.sum()),
            "action_mask":     self._get_action_mask(agent_idx),
            "broken":          True,
            "just_broke":      just_broke,
            "n_alive":         int(self.agent_alive.sum()),
        }
        return obs_i, 0.0, False, truncated, info

    def step(self, action: int):
        """Gymnasium-standard single-action step

        The real training loop drives each drone explicitly through
        ``step_agent(i, action)``
        """
        if self.done:
            # Auto-reset for the canonical gym loop convenience.
            self.reset()
        idx = self._current_agent
        obs_i, reward, terminated, truncated, info = self.step_agent(idx, action)
        info = dict(info)
        info["agent_idx"] = idx
        self._current_agent = (idx + 1) % self.n_agents
        return obs_i, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_all_obs(self):
        return [self._get_obs(i) for i in range(self.n_agents)]

    def get_all_masks(self):
        return [self._get_action_mask(i) for i in range(self.n_agents)]

    def n_alive_belief(self, agent_idx: int) -> int:
        """Team size drone ``agent_idx`` BELIEVES is still operational.

        ``n_agents`` minus the crashes this drone knows about — its own
        (beacon/fusion-acquired) knowledge, not a global oracle. Feeds the
        ``n_alive`` slot of the context vector.
        """
        return self.n_agents - len(self.agent_known_crashed[agent_idx])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_action_mask(self, agent_idx: int) -> np.ndarray:
        r, c = self.agent_pos[agent_idx]
        nrs  = r + ACTIONS_NP[:, 0]
        ncs  = c + ACTIONS_NP[:, 1]
        valid = (
            (nrs >= 0) & (nrs < self.H) &
            (ncs >= 0) & (ncs < self.W)
        )
        for a in range(N_ACTIONS):
            if valid[a] and self.grid[nrs[a], ncs[a]] == 1:
                valid[a] = False
        if not valid.any():
            valid[:] = True
        return valid

    def _sync_known_mask(self):
        union_vis        = np.any(self.agent_visited[:, :self.H, :self.W] > 0, axis=0)
        self._known_mask |= union_vis

    def _update_map(self, agent_idx: int):
        r, c   = self.agent_pos[agent_idx]
        tr, tc = self.target_pos

        nrs = r + self._vis_dr
        ncs = c + self._vis_dc

        in_bounds = (
            (nrs >= 0) & (nrs < self.H) &
            (ncs >= 0) & (ncs < self.W)
        )
        ib_nrs = nrs[in_bounds].astype(np.int32)
        ib_ncs = ncs[in_bounds].astype(np.int32)

        cell_vals = self.grid[ib_nrs, ib_ncs]
        is_obs  = cell_vals == 1
        is_tgt  = (~is_obs) & (ib_nrs == tr) & (ib_ncs == tc)
        is_free = ~is_obs

        self.agent_obstacle[agent_idx, ib_nrs[is_obs],  ib_ncs[is_obs]]  = 1.0
        self.agent_visited [agent_idx, ib_nrs[is_free], ib_ncs[is_free]] = 1.0
        self.agent_target  [agent_idx, ib_nrs[is_tgt],  ib_ncs[is_tgt]]  = 1.0

    def _detect_wrecks(self, agent_idx: int):
        """Record crash sites whose distress beacon reaches this drone.

        A wreck is detected when its cell is within ``comm_range`` (the
        beacon uses the same radio metric as live communication) or inside
        the drone's vision box (the wreck is physically visible). Knowledge
        is stored per-drone and spreads through ``_communicate`` afterwards,
        exactly like map knowledge — there is no global fault oracle.
        """
        if not self.crash_pos:
            return
        r, c = self.agent_pos[agent_idx]
        for k, (kr, kc) in self.crash_pos.items():
            if k in self.agent_known_crashed[agent_idx]:
                continue
            d = (abs(r - kr) + abs(c - kc)) if self.comm_metric == "manhattan" \
                else ((r - kr) ** 2 + (c - kc) ** 2) ** 0.5
            seen = max(abs(r - kr), abs(c - kc)) <= self.vision_radius
            if d <= self.comm_range or seen:
                self.agent_known_crashed[agent_idx].add(k)
                self.agent_broken[agent_idx, kr, kc] = 1.0

    def _communicate(self):
        # Broken drones neither transmit nor receive: only live drones can
        # form comm groups. Their frozen maps keep whatever was fused before
        # the fault.
        alive = [i for i in range(self.n_agents) if self.agent_alive[i]]
        parent = list(range(self.n_agents))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ii, i in enumerate(alive):
            for j in alive[ii + 1:]:
                ri, ci = self.agent_pos[i]
                rj, cj = self.agent_pos[j]
                d = (abs(ri - rj) + abs(ci - cj)) if self.comm_metric == "manhattan" \
                    else ((ri - rj)**2 + (ci - cj)**2) ** 0.5
                if d <= self.comm_range:
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj

        groups = defaultdict(list)
        for i in alive:
            groups[find(i)].append(i)

        for members in groups.values():
            if len(members) > 1:
                idxs       = list(members)
                merged_vis = self.agent_visited [idxs].max(axis=0)
                merged_obs = self.agent_obstacle[idxs].max(axis=0)
                merged_tgt = self.agent_target  [idxs].max(axis=0)
                merged_brk = self.agent_broken  [idxs].max(axis=0)
                self.agent_visited [idxs] = merged_vis
                self.agent_obstacle[idxs] = merged_obs
                self.agent_target  [idxs] = merged_tgt
                self.agent_broken  [idxs] = merged_brk
                # Fault knowledge fuses like the maps do (set union).
                merged_crashed = set().union(
                    *(self.agent_known_crashed[k] for k in idxs)
                )
                for k in idxs:
                    self.agent_known_crashed[k] = set(merged_crashed)

    def _get_obs(self, agent_idx: int) -> dict:
        """
        Build the observation for agent_idx as a dict with two flat arrays.

        Returns
        -------
        {
          'global': np.ndarray, shape (GLOBAL_CHANNELS * grid_size^2,)
                    channels: [Visited, Obstacle, Trajectory, Target,
                               Own_Position, Broken]
          'local':  np.ndarray, shape (LOCAL_CHANNELS * grid_size^2,)
                    channels: [Visited, Obstacle, Trajectory, Target, Other_Position]
        }

        Note: the drone's own position IS encoded here, as the Own_Position
        global channel (a single 1.0 at agent_pos[agent_idx]). The network
        recovers the crop coordinates from it via argmax.
        """
        r, c = self.agent_pos[agent_idx]

        vis_map  = self.agent_visited [agent_idx]  # (H, W)
        obs_map  = self.agent_obstacle[agent_idx]  # (H, W)
        tgt_map  = self.agent_target  [agent_idx]  # (H, W)
        traj_map = np.clip(
            self.agent_trajectory[agent_idx] / _TRAJ_CAP, 0.0, 1.0
        )                                           # (H, W), normalised

        # Other-drone positions within vision radius (local stream only).
        # Only LIVE teammates appear here; wrecks show up in the Broken
        # channel instead (once known to this drone).
        mg = self.grid_size
        others_pos = np.zeros((mg, mg), dtype=np.float32)
        for k in range(self.n_agents):
            if k == agent_idx or not self.agent_alive[k]:
                continue
            rk, ck = self.agent_pos[k]
            if abs(rk - r) <= self.vision_radius and abs(ck - c) <= self.vision_radius:
                others_pos[rk, ck] = 1.0

        # Own position: single 1.0 at this drone's cell (the network reads it
        # back to crop the local patch).
        own_pos = np.zeros((mg, mg), dtype=np.float32)
        own_pos[r, c] = 1.0

        # Global observation: 6 channels
        obs_global = np.concatenate([
            vis_map.ravel(),
            obs_map.ravel(),
            traj_map.ravel(),
            tgt_map.ravel(),
            own_pos.ravel(),
            self.agent_broken[agent_idx].ravel(),
        ])  # shape: (GLOBAL_CHANNELS * mg^2,)

        # Local observation: 5 channels (same 4 + other-drone positions)
        obs_local = np.concatenate([
            vis_map.ravel(),
            obs_map.ravel(),
            traj_map.ravel(),
            tgt_map.ravel(),
            others_pos.ravel(),
        ])  # shape: (LOCAL_CHANNELS * mg^2,)

        return {"global": obs_global, "local": obs_local}

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from gui.renderer import DroneRenderer
                self.renderer = DroneRenderer(self)
            self.renderer.render()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                from gui.renderer import DroneRenderer
                self.renderer = DroneRenderer(self, headless=True)
            return self.renderer.get_rgb_array()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

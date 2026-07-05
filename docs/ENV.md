# Environment — `DroneSearchEnv`

The custom Gymnasium environment in `env/grid_env.py` (grid generation and BFS
utilities in `env/utils.py`). Sequential-execution multi-drone search on a
fixed **32×32** grid **with random drone faults**. This document covers the
observation layout, the full reward function, the fault model, domain
randomization, the communication fusion, the trajectory channel, the action
mask, and termination.

> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) (network, invariants),
> [TRAINING.md](TRAINING.md) (training loop), [QUICK_REFERENCE.md](QUICK_REFERENCE.md).

---

## 1. Overview & lifecycle

```
DroneSearchEnv(config, render_mode=None)
  set_domain_params(vision_radius, comm_range, n_agents,
                    obstacle_density, fault_prob)           # BEFORE reset()
  obs, info = reset(seed=None)                              # new map + spawn
  obs_i, reward, terminated, truncated, info = step_agent(i, action)  # one drone
  obs_i, reward, terminated, truncated, info = step(action)          # gym wrapper
  n_alive_belief(i)                                         # drone i's believed team size
  render() / close()
```

The environment is **multi-agent and sequential**: the training loop drives the
drones round-robin via `step_agent(i, action)`, one call per drone per round, and
the caller owns the loop and the action order. There is **no vectorized
`step(actions)`** that advances the whole team in one call. A standard Gymnasium
`step(action)` *does* exist as a thin wrapper: it applies the action to the next
drone in round-robin order (`self._current_agent`), auto-resets when the episode
is done, and returns the canonical 5-tuple — so the env passes the canonical
`env.step(env.action_space.sample())` smoke-test. `set_domain_params` must be
called *before* `reset()` so the next episode's FOV / comm range / team size match
the context vector fed to the network.

**Spaces.** `action_space = Discrete(4)`. `observation_space = Dict({"global":
Box, "local": Box})` matching the per-agent dict returned by `reset` /
`step_agent`.

**Actions** (`ACTIONS`, index → delta): `0:up (-1,0)`, `1:down (1,0)`,
`2:left (0,-1)`, `3:right (0,1)`.

---

## 2. Observation layout

Each `step_agent` / `reset` returns, **per agent**, a dict of two flat `float32`
arrays in `[0,1]` (own position **is** encoded, as the `Own_Position` global
channel — see [ARCHITECTURE.md §4](ARCHITECTURE.md)):

| Key | Flat shape (gs=32) | Channels |
|-----|--------------------|----------|
| `global` | `6 · 32² = 6144` | `[Visited, Obstacle, Trajectory, Target, Own_Position, Broken]` |
| `local`  | `5 · 32² = 5120` | `[Visited, Obstacle, Trajectory, Target, Other_Position]` |

| Channel | Built from | Encoding |
|---------|-----------|----------|
| Visited | `agent_visited[i]` | `{0,1}` free cells this drone has seen (or comm-shared) |
| Obstacle | `agent_obstacle[i]` | `{0,1}` discovered obstacles |
| Trajectory | `agent_trajectory[i]` | `clip(count / _TRAJ_CAP, 0, 1)`, `_TRAJ_CAP = 5.0` |
| Target | `agent_target[i]` | `{0,1}` target cell once seen (persists) |
| Own_Position | `agent_pos[i]` | `{0,1}` this drone's own cell (**global** only; argmax → patch crop) |
| Broken | `agent_broken[i]` | `{0,1}` crash sites **known to this drone** (**global** only; beacon + fusion, see §4) |
| Other_Position | other **live** drones | `{0,1}` live teammates within `vision_radius` (**local** only; wrecks excluded) |

The map is updated in `_update_map(i)`: cells within the agent's Chebyshev vision
radius are revealed — obstacles → Obstacle, the target (if in view) → Target, all
other in-bounds cells → Visited.

---

## 3. Reward function

Computed inside `step_agent`. The reward **starts at `step_penalty`** and
accumulates terms. Default values are from `configs/default.yaml` (the `.get(...)`
fallbacks in code are only used if a key is absent):

| Term | Config value | When applied |
|------|--------------|--------------|
| `step_penalty` | `-0.3` | **every** move (the base reward) |
| `wall_penalty` | `-0.02` | invalid move (out-of-bounds or into an obstacle); the drone does **not** move |
| `revisit_penalty · k` | `-0.05 · k` | entering a cell with prior visit count `k > 0` (linear per step) |
| `collision_penalty · m` | `-0.5 · m` | entering a cell shared by `m` other **live** drones (wrecks are on the ground and don't collide) |
| `exploration_bonus · c` | `+0.02 · c` | `c` = number of newly globally-discovered cells this step |
| `target_reward` | `+200.0` | the drone's new cell **is** the target (also terminates) |

### Exact step logic

```python
if not agent_alive[i]:            # wreck: no-op turn, clock still ticks
    return obs_i, 0.0, False, truncated, info          # info["broken"]=True
if rng.random() < fault_prob:     # the drone breaks BEFORE moving
    agent_alive[i] = False; crash_pos[i] = agent_pos[i]
    return obs_i, 0.0, False, truncated, info          # info["just_broke"]=True

reward = step_penalty
valid  = in_bounds(nr, nc) and grid[nr, nc] != 1
if not valid:
    reward += wall_penalty                      # stays in place
else:
    prev_count = trajectory[i, nr, nc]
    if prev_count > 0:
        reward += revisit_penalty * prev_count  # linear in prior visits
    move to (nr, nc); trajectory[i, nr, nc] += 1
    m = #(other LIVE drones currently on (nr, nc))
    if m > 0:
        reward += collision_penalty * m
update_map(i); detect_wrecks(i); communicate(); sync_known_mask()
reward += exploration_bonus * (#newly known cells)
terminated = (pos == target_pos)
if terminated: reward += target_reward
```

Note the exploration bonus uses the **team** known-mask (union of all drones'
visited cells), so it rewards *globally* new discoveries, not per-drone novelty.

### Why the revisit penalty grows quadratically

The penalty is **linear per step** (`revisit_penalty · k`, with `k` the visit
count *before* entering), but the **cumulative** cost of repeatedly visiting one
cell is quadratic. Visiting a cell `n` times (after the first free visit) costs

```
Σ_{k=1}^{n-1} revisit_penalty · k = revisit_penalty · (n-1)·n / 2   →  O(n²)
```

This makes tight oscillation loops increasingly expensive, discouraging the agent
from pacing back and forth. The trajectory channel saturates at `_TRAJ_CAP = 5`
(lowered from 10) so the signal becomes visible to the network sooner.

---

## 4. Fault model (random drone failures)

Configured by `env.fault_prob` — the per-step probability that a live drone
breaks at the start of its own turn (0 disables faults; DR-randomized per
episode in training, fixed via CLI/GUI in eval).

- **Breaking.** The coin is flipped from `env.rng` at the start of the drone's
  turn, *before* it moves. Because map generation consumed the RNG first, a
  fixed `reset(seed=...)` reproduces both the layout **and** the fault
  sequence — evaluation stays comparable across checkpoints.
- **A wreck's turn is a no-op that still ticks the clock** (`step_count += 1`):
  losing drones never buys the team extra time. The wreck stops moving,
  sensing, and communicating; its maps freeze.
- **Non-blocking.** The crash cell is *not* an obstacle (live drones fly over
  it), the wreck is excluded from collision counting and from
  `Other_Position`. This preserves the reset-time target-reachability
  guarantee.
- **Distress beacon (decentralized fault knowledge).** In `_detect_wrecks(i)`,
  a live drone records a crash site in its **own** `agent_broken[i]` map and
  `agent_known_crashed[i]` set when the wreck is within `comm_range` (beacon
  radio, same metric as live comms) **or** inside its vision box. The knowledge
  then spreads through `_communicate()` fusion like any map knowledge — there
  is **no global fault oracle**.
- **Belief.** `n_alive_belief(i) = n_agents − |agent_known_crashed[i]|` — the
  team size drone `i` *believes* is operational; it feeds the context vector.
- **All dead ⇒ early truncation** (non-terminal): nothing can change anymore.

`info` flags: `broken` (this turn was a wreck's no-op), `just_broke` (the drone
broke at the start of this very turn), `n_alive` (ground-truth live count, for
logging/GUI only — never fed to the network).

---

## 5. Domain randomization (`set_domain_params`)

```python
set_domain_params(vision_radius=None, comm_range=None, n_agents=None,
                  obstacle_density=None, fault_prob=None)
```

- `vision_radius` — recomputes the vision offset grid (`_compute_vision_offsets`,
  a `(2r+1)×(2r+1)` Chebyshev neighborhood) when changed.
- `comm_range` — new fusion distance for `_communicate`.
- `n_agents` — new team size; the next `reset()` reallocates the per-agent map
  arrays `(n_agents, gs, gs)` and spawns that many drones.
- `obstacle_density` — density used by the next `reset()`'s `generate_grid`.
- `fault_prob` — per-step fault probability for the next episode (§4).

Only arguments that are not `None` are applied. The training loop samples these
per episode (`sample_ctx_params`) and calls `set_domain_params` **before**
`reset()`; eval/play/`eval_checkpoints` resolve a fixed or per-seed context the
same way. See [TRAINING.md §3](TRAINING.md).

---

## 6. Communication — union-find map fusion (`_communicate`)

After every map update, **live** drones within `comm_range` of each other merge
their accumulated maps (wrecks neither transmit nor receive — their maps
freeze at fault time):

1. Build a union-find over the live drones (path-halving `find`).
2. For each live pair `(i, j)`, compute distance per `comm_metric`:
   - `"manhattan"`: `|rᵢ−rⱼ| + |cᵢ−cⱼ|`
   - else (euclidean): `√((rᵢ−rⱼ)² + (cᵢ−cⱼ)²)`
   - if `d <= comm_range`, union `i` and `j`.
3. For each connected group of size > 1, merge **Visited / Obstacle / Target /
   Broken** maps with an element-wise `max` (and union the
   `agent_known_crashed` sets) and write the merged state back to every member.

So communication is **transitive within a group** (A↔B, B↔C ⇒ A,B,C all share)
and instantaneous. The **Trajectory** and **Other_Position** channels are **not**
shared — they remain per-drone (trajectory is a private revisit memory). The
default `comm_metric` is `"manhattan"`.

---

## 7. Action mask & corner-case safety fallback (`_get_action_mask`)

Returns a length-4 boolean array marking legal moves:

```python
valid = in_bounds(neighbor)        # for each of the 4 directions
for a in actions:
    if valid[a] and grid[neighbor] == 1:   # obstacle
        valid[a] = False
if not valid.any():
    valid[:] = True                # SAFETY FALLBACK
return valid
```

A move is masked out if it leaves the grid or steps into a known obstacle. The
**safety fallback** matters: if a drone is fully boxed in (every neighbor is a
wall or obstacle), all four entries would be `False`, and downstream
`argmax(q[mask])` over an all-`-inf` vector / `random.choice([])` would crash or
be undefined. Setting the whole mask to `True` guarantees the agent always has a
selectable action (it will simply hit a wall and pay `wall_penalty`, staying put).

---

## 8. Termination vs truncation

`step_agent` returns both flags every call:

| Flag | Condition | Meaning |
|------|-----------|---------|
| `terminated` | `agent_pos[i] == target_pos` | success — the target was reached (absorbing) |
| `truncated` | `step_count >= max_steps · n_agents` **or** all drones broken | time limit / team wiped out |

`step_count` increments **once per `step_agent` call** (i.e. per agent-step,
including wrecks' no-op turns), so the budget `max_steps · n_agents`
corresponds to roughly `max_steps` full rounds regardless of faults.
`self.done` becomes `terminated or truncated`; the loop exits on either, but the
two are **semantically different for learning**: only `terminated` ends an n-step
window and zeroes the bootstrap (truncation must still bootstrap the next state —
see [TRAINING.md §5](TRAINING.md) and Invariant 7 in ARCHITECTURE). A drone's
own **fault** is likewise a truncation for that drone, never a terminal.

`info` carries: `step`, `found` (= `terminated`), `new_cells`,
`agent_positions`, `known_cells`, `action_mask`, `broken`, `just_broke`,
`n_alive`.

---

## 9. Grid generation & spawn (`env/utils.py`, `reset`)

`generate_grid(H, W, obstacle_density, rng)`:

1. Up to **200 attempts**: sample `grid = (rng.random(H,W) < density)`, force
   `grid[0,0] = 0` (start free), BFS from `(0,0)` (`bfs_reachable`, 4-connected).
2. Pick the target uniformly from the reachable cells **excluding** `(0,0)`. This
   **guarantees the target is reachable from `(0,0)`**.
3. Fallback (no valid layout in 200 tries): empty grid, target `(H-1, W-1)`.

Default `obstacle_density = 0.2`; during training it is DR-randomized per
episode from `domain_randomization.obstacle_density` (default `[0.10, 0.30]`),
so the policy sees sparse and dense worlds alike.

`env/utils.py` also hosts the eval-only navigation helpers:
`bfs_shortest_path(known_obstacles, start, goal)` (optimistic BFS on a drone's
*known* map — unknown cells assumed free, replanned every step),
`bfs_next_action(...)` (first move as an action index), and
`auto_nav_action(env, i)` (the deterministic go-to-target override used by
eval/simulate — see ARCHITECTURE Invariant 10).

**Spawn.** All drones start at one corner, shared (the collision penalty teaches
them to disperse). `_SPAWN_CORNERS = [(0,0), (0,-1), (-1,0), (-1,-1)]` (negative
indices wrap to the last row/col). The corner is random per episode unless
`env.spawn_corner` is set (e.g. `--mode simulate` forces index `0`, top-left).
Each spawned drone seeds its trajectory at the start cell and runs an initial
vision update, then `_communicate()` fuses the (currently identical) maps.

> ⚠️ Reachability is only guaranteed from `(0,0)`; a random non-top-left spawn can
> in principle land in a region from which the target is unreachable. See
> [BUGS_AND_IMPROVEMENTS.md](BUGS_AND_IMPROVEMENTS.md) §Known Bugs.

---

## 10. State arrays (per episode, allocated in `reset`)

| Attribute | Shape | Meaning |
|-----------|-------|---------|
| `grid` | `(H, W)` int | `0` free, `1` obstacle |
| `target_pos` | `(row, col)` | hidden target |
| `agent_pos` | list of `n_agents` `(row,col)` | current positions |
| `agent_visited / _obstacle / _target` | `(n_agents, H, W)` | per-drone discovered maps (comm-shared) |
| `agent_trajectory` | `(n_agents, H, W)` | per-drone visit counts (private) |
| `agent_alive` | `(n_agents,)` bool | `False` once a drone broke (ground truth) |
| `crash_pos` | dict `idx → (row,col)` | ground-truth crash sites |
| `agent_broken` | `(n_agents, H, W)` | crash sites **known to each drone** (comm-shared) |
| `agent_known_crashed` | list of sets | crashed teammate ids known to each drone (drives `n_alive_belief`) |
| `_known_mask` | `(H, W)` bool | union of all drones' visited cells (drives exploration bonus) |
| `step_count` | int | agent-steps so far this episode |

`render()` lazily constructs `gui/renderer.DroneRenderer` (human or headless
`rgb_array`); the env never drives the GUI — it is a pure view.

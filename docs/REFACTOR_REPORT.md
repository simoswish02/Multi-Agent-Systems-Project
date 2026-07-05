# Refactor Report

Audit and refactor of the Multi-Drone Search DQN project. The guiding
constraint throughout was the project's own rule: **do not break existing
training runs or checkpoint compatibility, and keep `main.py` /
`eval_checkpoints.py` runnable.** Every change below was made under that
constraint and validated by a smoke test that (a) loads the existing
`checkpoints/best.pt`, (b) runs an env episode with batched action selection,
(c) executes a gradient `update()`, and (d) confirms input validation fires.

## 1. Summary of changes

| File | Change | Reason |
|------|--------|--------|
| `training/train.py` | Store `float(terminated)` (not `float(done)`) as the bootstrap flag | Fix time-limit truncation bootstrapping bug (KNOWN_BUGS #1) |
| `agents/dqn_agent.py` | `eval()`/`train()` toggle around inference forwards in `select_action` and `select_actions_batch` | BatchNorm must use running stats at inference, not noisy per-batch stats (KNOWN_BUGS #2) |
| `agents/dqn_agent.py` | Removed duplicated `nog_t = ...` line in `update()` | Dead code / redundant device copy (KNOWN_BUGS #3) |
| `gui/renderer.py` | Parenthesized `("+" in line or "-" in line)` | Operator-precedence logic bug (KNOWN_BUGS #4) |
| `env/grid_env.py` | Corrected revisit-penalty docstring (linear per-step) | Stale "quadratic" claim (KNOWN_BUGS #5) |
| `agents/dqn_agent.py` | Removed unused `make_ctx_tensor()`; removed its import in `train.py` | Dead code (KNOWN_BUGS #6) |
| `agents/dqn_agent.py` | Removed unused `map_channels` constructor param + `net_kwargs` plumbing | Dead code; channel counts hardcoded in `GlobalMapCNN` |
| `env/grid_env.py` | Removed unused constants `UNKNOWN/FREE_VISITED/OBSTACLE/TARGET/AGENT` and the `known_free_cells` set | Dead code + per-step cost removed |
| `agents/dqn_agent.py` | Added constructor input validation (`grid_size`, `n_actions`, `batch_size ≤ buffer_size`, `gamma ∈ [0,1]`) | Defensive programming (Phase 5) |
| `agents/dqn_agent.py` | Wrapped `save`/`load` in `try/except` with informative messages | Robust checkpoint I/O (Phase 5) |
| `agents/base_agent.py` | **New** `BaseAgent` ABC; `DQNAgent` now implements it | Extensibility: swap algorithms without touching the loop (Phase 7) |
| `docs/KNOWN_BUGS_AND_FIXES.md` | **New** | Phase 2/6 deliverable |
| `docs/ARCHITECTURE.md` | **New** | Phase 6 deliverable |
| `docs/REFACTOR_REPORT.md` | **New** (this file) | Phase 8 deliverable |

No public function signatures used by `main.py` / `eval_checkpoints.py` /
`train.py` changed (other than removing the unused `make_ctx_tensor` import and
the unused `map_channels` kwarg, which had no callers passing it).

## 2. Audit results that required NO change

These were checked against the Phase-2 checklist and found correct; details and
verification steps are in `docs/KNOWN_BUGS_AND_FIXES.md`:

- LR warmup→cosine schedule (verified empirically: base → max → min).
- Double DQN action-selection / value-estimation roles.
- Gradient-clip placement; target hard-update frequency.
- Replay sampling (i.i.d., capacity-bounded, guarded against under-fill).
- Epsilon floor; per-episode decay timing.
- Seeding coverage (Python / NumPy / Torch CPU+CUDA).
- Division-by-zero / NaN paths (none in the hot path).

## 3. Follow-up round — module split + L1/L2/L3

A second pass (checkpoint compatibility no longer required as a hard constraint)
completed the structural split and resolved the three open limitations.

| File | Change | Reason |
|------|--------|--------|
| `agents/networks.py` | **New** — all `nn.Module` definitions + channel/patch/context constants moved here | Phase 4 module split |
| `agents/replay_buffer.py` | **New** — `ReplayBuffer` isolated (with a `capacity > 0` guard) | Phase 4: independently testable |
| `agents/scheduler_utils.py` | **New** — `EpisodeLRScheduler` encapsulates warmup→cosine + `state_dict`/`load_state_dict` | Phase 4: single-responsibility |
| `agents/dqn_agent.py` | Slimmed to the `DQNAgent` class; imports the three modules; delegates the LR schedule | Single responsibility |
| `env/grid_env.py` | `observation_space` → `spaces.Dict({"global","local"})` | **L1** resolved |
| `configs/default.yaml` | Removed `cnn_map_channels` **and** the unused `gui` section | **L2** resolved |
| `agents/dqn_agent.py` | `save(training_state=…)`, `load` returns `training_state`, new `load_weights_only` | **L3** resolved |
| `training/train.py` | `train(resume_path, init_weights_path)`; curriculum-position resume; writes `last.pt`; per-phase `start_ep` / epsilon-reset guard | **L3** resolved |
| `main.py` | `--init-weights` flag, mutually exclusive with `--resume`; updated help | **L3** resolved |

**Backward compatibility kept where free:** `save`/`load` still use the
`warmup_sched` / `cosine_sched` / `sched_episode` checkpoint keys, and the
network module hierarchy is unchanged, so **existing checkpoints (`best.pt`,
`phase*_ep*.pt`) still load** — verified in the smoke test. The public import
surface (`from agents.dqn_agent import DQNAgent`, plus `CnnQNetwork` /
`ReplayBuffer` re-exported through that module's namespace) is preserved.

### Still deliberately scoped out

- **`utils/config_loader.py` schema validation** — worth adding (catch misspelled
  YAML keys early); a minimal allowlist validator over the top-level sections is
  the suggested form.
- **Logging abstraction / Hydra CLI overrides** — larger features with low
  marginal value for the current single-config workflow.

## 3b. Research round — DR coherence, n-step, team reward, GroupNorm

Checkpoint compatibility was intentionally dropped here (a fresh training run is
required anyway).

| File | Change | Reason |
|------|--------|--------|
| `env/grid_env.py` | `set_domain_params(vision_radius, comm_range, n_agents)` applied per episode (incl. `n_agents`, with `reset()` reallocating arrays) | DR was decoupled from the env — see KNOWN_BUGS #7 |
| `training/train.py`, `main.py`, `eval_checkpoints.py` | Apply DR params to the env before every `reset()` (training + eval + play) | Coherent partial observability |
| `agents/nstep.py` | **New** `NStepBuffer` — per-agent n-step returns for the sequential env, folds `gamma`/done into a stored bootstrap discount; `apply_team_terminal` for cooperative credit | Faster propagation of the sparse terminal reward; A/B team reward |
| `agents/dqn_agent.py` | `update()` target = `R + bootstrap_disc · Q(s')`; `push_transition` carries the n-step return + discount | n-step Double-DQN |
| `configs/default.yaml` | `gamma` 0.95 → 0.97, `n_step: 3`, `env.shared_target_reward: false` | Longer horizon; n-step; cooperative-reward toggle |
| `agents/networks.py` | All `BatchNorm2d` → `GroupNorm` (helper `_gn`) | Batch-size-independent norm; removes the small-batch / batch-1 / train-eval-mode hazards |

Verified by unit tests (n-step returns, terminal handling, team reward),
GroupNorm forward/update, FOV/comm/team coherence, and a full tiny end-to-end
`train()` run (DR variation, n-step, team reward, eval, `last.pt` resume).

## 4. Remaining limitations / TODOs

1. **No automated tests.** The smoke checks used across both rounds should be
   committed as `tests/` so regressions in the env/agent contract are caught.
2. **`--resume` resumes at `save_every` granularity** (from `last.pt`): up to
   `save_every` episodes of progress can be lost on an interruption. Lower
   `save_every` or add an end-of-run save hook if finer recovery is needed.

## 5. Suggested next steps (research)

Done in the research round: **γ 0.95→0.97**, **n-step returns** (`n_step=3`),
**GroupNorm**, **cooperative terminal reward** (config toggle).

Still open, roughly by expected impact:

- **Diagnose failure modes first** using the per-DR TensorBoard scalars
  (success vs. `vision_radius` / `n_agents`) before adding more machinery.
- **Prioritized Experience Replay** — focus updates on high-TD-error
  transitions; the sparse `target_reward` makes this attractive. The buffer is
  already isolated in `replay_buffer.py`.
- **A/B the team reward** (`shared_target_reward`) against the individual
  variant to measure the cooperative-credit effect.
- **Larger `batch_size`** (16 → 32) now that GroupNorm removes the small-batch
  norm fragility, if VRAM allows.
- **Reward normalization / scaling** of `target_reward` (200) relative to the
  dense shaping terms, to reduce TD-target variance.
- **Softer DR curriculum for `n_agents`** (e.g. `dr_override` ramping `[1,2] →
  [1,3] → [1,4]`) in early phases.

## GUI — What changed

The pygame renderer was extended from a single grid + thin text panel into an
interactive dashboard. The work is split across three files in `gui/`:

- `gui/renderer.py` — orchestrator (`DroneRenderer`), playback controls, top
  toolbar, main grid, the collapsible sidebar, and the 2×2 drone mini-maps.
- `gui/charts.py` — **new** — pure-pygame `draw_line_chart` / `draw_bar_chart`
  primitives (no matplotlib).
- `gui/terminal.py` — **new** — `TerminalLog`, a colour-coded scrollback widget
  with auto-scroll and mouse-wheel history.
- `gui/__init__.py` — now re-exports `DroneRenderer`.

### Public interface (unchanged)

`DroneRenderer.__init__(env, headless=False)`, `render()`, `get_rgb_array()`,
`close()` keep their old signatures, so `env.render()` and every existing call
site work untouched.

### New public method

- `reset_episode_data()` — clears chart history + per-episode trackers. Called
  automatically when an episode boundary is detected (`env.step_count` rewinds
  to 0), and exposed for callers that want to force a reset.

### Features

1. **Playback controls** in the toolbar: `[> Play]`, `[|| Pause]`, `[>| Step]`
   buttons + a PLAYING/PAUSED status indicator. Keyboard: `Space` toggles
   Play/Pause, `→` single-steps (only while paused). Step is disabled (greyed)
   while playing.
2. **Wider collapsible sidebar** (`SIDEBAR_W = 500`) with three click-to-collapse
   sections (all expanded by default): A mini-maps, B charts, C episode log.
3. **Section A — 2×2 drone mini-maps**: each shows only that drone's own
   `agent_visited` / `agent_obstacle` / `agent_target` arrays, its position
   (coloured circle), a `Di` label, and a coloured border that flashes
   `COLOR_COMM_FLASH` for ~4 frames when the drone communicates. Slots beyond
   `env.n_agents` render as dark `-- inactive --` panels.
4. **Section B — five live charts** (pygame primitives only): per-drone step
   reward (line, one per drone + legend), episode cumulative reward (line),
   coverage % (line, dashed 100% reference), new cells / step (green bars),
   communications / step (yellow bars). All share the `env.step_count` x-base.
5. **Section C — episode log**: colour-coded scrollback (headers white, normal
   grey, comms yellow, target orange, negative-reward lines red-tinted), with
   the episode-start header block, per-drone step lines, comm events, target-
   found and episode-end footer. Auto-scrolls; mouse-wheel scrolls history when
   the cursor is over the panel.

### How playback gates the loop

`render()` does not drive the env. While PAUSED it **blocks** inside `render()`
(processing events + redrawing) until Play or a single Step is requested, then
returns to let the caller advance once. No change to `main.py` was needed.

### Env attributes read

`H`, `W`, `n_agents`, `step_count`, `done`, `agent_pos`, `agent_visited`,
`agent_obstacle`, `agent_target`, `target_pos`, `last_rewards`,
`episode_reward`, `grid` (ground-truth obstacles, for the coverage
denominator), `comm_range`, `comm_metric`, `vision_radius`, `obstacle_density`,
`max_steps`. Optional ones are read via `getattr(..., 'N/A')`. Communication
events are derived in the renderer by re-applying the env's pairwise
comm-range test to `agent_pos` (no new env flag required).

### Assumptions / caveats

- One **Step** = one `render()`-to-`render()` interval. In the eval loop that is
  one full round (all drones move once), since `render()` is called once per
  round — not per individual `step_agent`.
- The eval loop exits right after the terminal `step_agent` **without** a final
  `render()`, so an episode's "target found" / footer lines are emitted on the
  next episode's first frame (best-effort); the very last episode of a run may
  omit its footer. Interactive (paused) stepping and `play` mode are unaffected.
- "New cells" per drone counts cells newly present in that drone's `agent_visited`
  array, which includes cells gained via comm fusion.
- Charts/log update once per render frame (≈ once per round), so the x-axis is a
  per-round sampling of `step_count`.

## GUI additions — comm-range toggle & `simulate` mode (2026-06-17)

- **Comm-range overlay.** `DroneRenderer` gained `show_comm_range` (toggle with
  the toolbar **(C) Comm** button or the `C` key): a translucent per-drone
  overlay of the communication range — a diamond for the Manhattan metric, a
  circle for Euclidean — drawn under the drone markers (`_draw_comm_ranges`).
  Works in `eval`, `play` and `simulate`.
- **`--mode simulate`.** A setup screen (`gui/config_screen.py`: sliders + a
  weights-path field) collects drones / vision / comm / max-steps / episodes /
  weights (defaults to `<checkpoint_dir>/best.pt`, validated before start), then
  runs that many episodes with the live GUI and **reopens the setup screen** when
  they finish; closing the setup window exits the mode (`main.run_simulate`).
  Slider maxima for vision/comm/drones come from `domain_randomization` (coherent
  with the context normalisation). **Spawn is forced to the top-left corner** via
  the new `env.spawn_corner` (config key `env.spawn_corner`; `None` keeps the
  training-time random corner). Uses only pygame + (optional) tkinter for the
  Browse dialog — no new dependencies.

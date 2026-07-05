# Known Bugs and Fixes

Audit of the Multi-Drone Search DQN codebase. Each entry records the bug, its
root cause, the fix applied, and how to verify the fix. A second section lists
items that were **checked and found correct** (so they are not re-investigated),
and a third lists **known limitations left open** with rationale.

---

## Bugs found and fixed

### 1. Time-limit truncation killed the TD bootstrap (RL correctness — high impact)

**Description.** The replay transition stored `done = terminated or truncated`.
The TD target is `r + γ · Q(s') · (1 − done)`, so on a *truncated* episode (the
agent simply ran out of time at `max_steps · n_agents`) the next-state value was
forced to zero.

**Root cause.** Truncation is not termination. A truncated state is non-terminal
— the target could still be found from there — so its value must be
bootstrapped. Conflating the two teaches the agent that running out of time is as
bad as reaching an absorbing failure state (the classic time-limit bootstrapping
bug, Pardo et al. 2018). With `target_reward = 200` and most early-training
episodes ending in truncation, this systematically biases value estimates down.

**Fix.** `training/train.py` now stores `float(terminated)` in the transition.
The episode loop still *exits* on `terminated or truncated`; only the bootstrap
flag changed.

**Verify.** In `DQNAgent.update`, confirm `dn_t` is 1.0 only for transitions
where the target was found. A quick check: run an episode to truncation and
confirm the last stored transition has `done == 0.0` while a target-reaching
transition has `done == 1.0`.

---

### 2. BatchNorm ran in training mode during action selection (RL correctness — medium impact)

**Description.** `q_net` was constructed in (default) `train()` mode and never
switched to `eval()` for inference. `select_action` / `select_actions_batch`
therefore ran the BatchNorm-heavy network using **per-batch** statistics and
**updated the running averages** on every greedy forward pass — including the
batch-size-1 path in `select_action`.

**Root cause.** Missing `eval()`/`train()` discipline. Consequences: (a) greedy
Q-values computed from noisy single-sample batch statistics instead of the
learned running statistics; (b) running mean/var polluted by inference forwards,
which then get copied into `target_net` at every hard sync.

**Fix.** Both selection methods now snapshot the mode, call `self.q_net.eval()`
around the `torch.no_grad()` forward, and restore with
`self.q_net.train(was_training)`. The training forward in `update()` is
unchanged (correctly stays in `train()` mode).

**Verify.** After calling `select_action`, assert `agent.q_net.training is True`
(restored). With identical inputs, greedy actions are now deterministic across
calls regardless of intervening batch sizes. (Covered by the smoke test.)

---

### 3. Duplicated tensor copy in `update()` (dead code)

**Description.** `nog_t = torch.from_numpy(nog).to(self.device)` appeared twice
in a row.

**Root cause.** Copy-paste slip. Harmless but wasteful (an extra host→device
copy each gradient step).

**Fix.** Removed the duplicate line in `agents/dqn_agent.py`.

**Verify.** `update()` still produces a finite loss (smoke test passes).

---

### 4. Operator-precedence logic bug in the renderer (cosmetic)

**Description.** `if line.strip().startswith("D") and ":" in line and "+" in line or "-" in line:`
parses as `(A and B and C) or D`. Because `D = "-" in line` matches almost any
panel line, the per-drone reward-color branch fired on unintended lines.

**Root cause.** Missing parentheses around the `or`.

**Fix.** Grouped the disjunction:
`... and ":" in line and ("+" in line or "-" in line)` in `gui/renderer.py`.

**Verify.** Run `python main.py --mode eval --checkpoint checkpoints/best.pt`;
only the per-drone reward lines are tinted green/red.

---

### 5. Stale "quadratic" revisit-penalty docstring (documentation)

**Description.** The env docstring described the revisit penalty as
`-revisit_penalty · k²` ("scales with visit count squared"), but the code applies
`revisit_penalty · prev_count` — **linear** in the visit count.

**Root cause.** The implementation was changed to a linear per-step penalty (see
the `linearRevisit` curriculum phase) without updating the docstring.

**Fix.** Corrected the `DroneSearchEnv` docstring to state the per-step penalty
is linear in `k`, while the *cumulative* cost of repeated revisits grows
quadratically.

**Verify.** Read `step_agent`: `reward += self.revisit_penalty * prev_count`.

---

### 6. Unused/dead code removed (maintainability)

| Symbol | Location | Note |
|--------|----------|------|
| `make_ctx_tensor()` | `agents/dqn_agent.py` | Defined and imported in `train.py`, never called (`ctx_to_numpy` is used everywhere). Function + import removed. |
| `map_channels` param / `cnn_map_channels` plumbing | `CnnQNetwork.__init__`, `DQNAgent` `net_kwargs` | Accepted but never used (channel counts are hardcoded in `GlobalMapCNN`). Constructor param and plumbing removed. |
| `UNKNOWN, FREE_VISITED, OBSTACLE, TARGET, AGENT` | `env/grid_env.py` | Module constants never referenced (the grid uses literal `1` for obstacles). Removed. |
| `known_free_cells` | `env/grid_env.py` | A `set(zip(*np.where(...)))` rebuilt on every `reset` and every `step_agent` but never read — also a per-step cost. Removed. |

**Verify.** `python main.py` and `python eval_checkpoints.py` import and run; the
smoke test exercises env reset/step and an `update()`.

---

### 7. Domain randomization was decoupled from the environment (RL correctness — high impact)

**Description.** Each episode sampled `vision_radius`, `comm_range`, and
`n_agents` (`sample_ctx_params`) and fed them to the network through the context
vector — but **never applied them to the environment**. The env kept the fixed
`config["env"]` values (`vision_radius = 3`, `comm_range = 5`) and the training
loop iterated a fixed team size. So the network was *told* a random FOV/comm/team
while always *observing* the same fixed configuration.

**Root cause.** `DroneSearchEnv` precomputed its vision offsets once at
construction and was created once per phase; nothing re-applied the per-episode
sample. The context channel for `vision_radius` (and `comm_range`, `n_agents`)
was therefore uncorrelated with the actual observations — effectively noise — so
domain randomization over these parameters was inactive. The partial-observability
assumption ("the drone knows only what is in its vision radius and what it has
explored") still held, but at a *fixed* radius of 3, not the randomized one.

**Fix.**
- Added `DroneSearchEnv.set_domain_params(vision_radius, comm_range, n_agents)`,
  which updates the FOV (recomputing the vision offsets), the comm range, and the
  team size; `reset()` reallocates the per-agent arrays from `self.n_agents`.
- `training/train.py::run_phase` now calls `env.set_domain_params(**ctx_params)`
  before every `reset()` and reads the team size as `env.n_agents` per episode.
- `evaluate()` does the same, sampling the DR config **deterministically** from
  each fixed eval seed (`random.Random(eval_seeds[i])`) so the metric stays
  comparable across epochs while remaining coherent with the env.
- `main.py` (`run_eval`, `run_play`) and `eval_checkpoints.py` apply the resolved
  context (including `--vision-radius` / `--comm-range` / `--n-agents`) to the
  real env, so eval changes both the FOV and the context.

**Verify.** With `set_domain_params(vision_radius=k)`, the cells revealed at spawn
are exactly those within Chebyshev distance `k` (checked for `k = 1, 2, 5`).
`n_agents = m` yields `agent_visited.shape[0] == m`. Two `evaluate()` calls with
the same eval seeds return identical results (deterministic). All covered by the
coherence smoke test.

**Impact on existing checkpoints.** Checkpoints trained before this fix learned
under the incoherent regime (always FOV 3). Re-evaluated coherently they still
solve the task (success ≈ 100%) but with lower reward; a fresh training run is
needed to actually benefit from randomized FOV/comm/team.

---

## Items checked and found CORRECT (no change needed)

- **LR schedule (warmup → cosine).** *Verified empirically.* LR starts at
  `base_lr` (5e-5), warmup reaches `max_lr` (1e-4), and cosine decays to `min_lr`
  (1e-6). The earlier worry that `CosineAnnealingLR` would restart from its
  captured `base_lrs` is unfounded: PyTorch's *chainable* `get_lr` advances
  relative to the **current** param-group LR, so the manual `pg["lr"] = max_lr`
  at the warmup→cosine transition is honored.
- **Double DQN roles.** Online net selects `argmax` for the next state; target
  net evaluates it. Not swapped.
- **Gradient clipping.** `clip_grad_norm_` is applied after `backward()` and
  before `optimizer.step()`, on `q_net` parameters only.
- **Target network update.** Hard copy every `target_update_freq` (500) gradient
  steps; counter logic correct.
- **Replay buffer.** `deque(maxlen=capacity)` with i.i.d. `random.sample`; the
  `len(buffer) < batch_size` guard prevents under-filled sampling; no off-by-one.
- **Epsilon-greedy.** Decays once per episode and is floored at `epsilon_end`
  (`max(epsilon_end, ε·decay)`); never decays below the floor.
- **Seeding.** `set_seeds` covers Python `random`, NumPy, Torch CPU and CUDA.
- **Numerical stability.** Normalization denominators (`ctx_norm`, `grid_size`,
  `_TRAJ_CAP`) are constants `> 0`; Huber (`smooth_l1_loss`) tolerates the large
  `target_reward` magnitude. No `log(0)`/`0÷0` paths.

---

## Previously-open limitations — now RESOLVED

### L1. `observation_space` now matches the returned observation ✅

`DroneSearchEnv.observation_space` was declared as `Tuple(Box, Box)` while
`reset` / `step_agent` return a `dict` `{"global", "local"}`.

**Fix.** Switched the declaration to `spaces.Dict({"global": Box, "local": Box})`
so the space matches what is actually produced. This is the *more correct* of the
two options (the alternative — change the env to return a tuple — would have
forced edits across the agent, training loop, and eval scripts that all index the
obs by the `"global"`/`"local"` keys). Verified:
`env.observation_space.contains(obs[i])` is now `True`.

### L2. Unused YAML keys removed ✅

`cnn_map_channels` (dead) **and** the entire `gui` section (`cell_size`, `fps`,
`enabled` — never read; the renderer uses its own class constants) were removed
from `configs/default.yaml`. A scan confirmed no remaining top-level key is
unreferenced.

### L3. `--resume` vs. `--init-weights` are now two explicit modes ✅

The previously ambiguous `--resume` ("init weights, but also restores
epsilon/schedule") is split into two clear flags (mutually exclusive):

- **`--resume PATH`** — *continue an interrupted run.* `DQNAgent.load` restores
  network weights, optimizer, LR schedule, epsilon, and `train_steps`, and now
  also returns a `training_state` dict (`global_episode`, `phase_idx`,
  `ep_in_phase`, `best_success`, `best_reward`) that `train()` uses to skip
  completed phases and restart the active phase at the right episode (without
  re-applying `epsilon_reset`). The loop writes `checkpoints/last.pt` (carrying
  this metadata) every `save_every` episodes and at each phase end.
- **`--init-weights PATH`** — *warm-start a fresh run.* `DQNAgent.load_weights_only`
  copies **only** the `q_net` weights; optimizer, schedule, epsilon, replay
  buffer, and curriculum all start from scratch.

**Verify.** Round-trip test: `save(..., training_state=…)` then `load(…)` returns
the same dict; `load_weights_only` leaves `epsilon` unchanged; a legacy
checkpoint (no metadata) loads and `load` returns `None`. All covered by the
smoke test.

---

## Limitations still open

- **No automated test suite.** The smoke checks used in this audit should be
  committed as `tests/`. (Tracked in `docs/REFACTOR_REPORT.md`.)

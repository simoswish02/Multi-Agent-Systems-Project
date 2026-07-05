> **Historical document** — inherited from the RL-course project *before* the
> Multi-Agent-Systems fault extension (random faults, Broken channel, CTX_DIM=5,
> obstacle-density/fault DR, eval-only BFS auto-nav). Numbers such as channel
> counts or context dims may be stale here; the current references are
> [ARCHITECTURE.md](ARCHITECTURE.md), [ENV.md](ENV.md), [TRAINING.md](TRAINING.md).

# Bugs & Improvements

Findings from a full read of the `convnext_attn_net` branch. Each item carries a
severity, a `file:line` location, and a concrete fix or recommendation.

**Severity legend:** 🔴 bug (incorrect behavior) · 🟡 performance improvement ·
🔵 style / maintainability / minor inconsistency.

A handful of items were **fixed in the same pass that produced these docs** — they
are marked **✅ FIXED** with a short note. See the historical audit in
[KNOWN_BUGS_AND_FIXES.md](KNOWN_BUGS_AND_FIXES.md) for earlier correctness fixes
(time-limit bootstrap, DR coherence, etc.).

---

## Known Bugs

### 🔴 B1 — Random spawn corner is not guaranteed reachable (or even free)

**Location:** `env/grid_env.py:182-186` (`reset`), `env/utils.py:5-36`
(`generate_grid`).

**Description.** `generate_grid` guarantees only that `(0,0)` is free and that the
target is reachable **from `(0,0)`**. But `reset` spawns all drones at a *random*
corner (`_SPAWN_CORNERS = [(0,0),(0,-1),(-1,0),(-1,-1)]`) whenever
`spawn_corner is None` (the training default). A non-top-left corner may be (a) an
obstacle cell, or (b) in a region disconnected from the target. In case (b) the
episode is **unsolvable** and can only ever truncate, injecting label noise into
training (the agent is penalized for failing an impossible task).

At `obstacle_density = 0.2`, a given corner is an obstacle ~20% of the time, and
disconnection is additionally possible.

**Fix.** Make the spawn coherent with reachability. Minimal version — choose the
target so it is reachable from the *actual* spawn corner, or validate the spawn:

```python
# in generate_grid: accept the spawn corner and guarantee it free + connected
def generate_grid(H, W, obstacle_density, rng, start=(0, 0)):
    sr, sc = start[0] % H, start[1] % W
    for _ in range(max_attempts):
        grid = (rng.random((H, W)) < obstacle_density).astype(np.int32)
        grid[sr, sc] = 0                       # spawn always free
        reachable = bfs_reachable(grid, (sr, sc))
        if len(reachable) < 2:
            continue
        candidates = [p for p in reachable if p != (sr, sc)]
        target_pos = candidates[int(rng.integers(0, len(candidates)))]
        return grid, target_pos
    ...
```

and pass the chosen corner from `reset` into `generate_grid`. (Simulate mode is
unaffected because it forces `spawn_corner = 0`.)

---

### 🔵→🔴 B2 — `step_agent` guards with `assert`, stripped under `python -O`

**Location:** `env/grid_env.py:222` — `assert not self.done, "Episode ended …"`.

**Description.** The "stepping a finished episode" guard is an `assert`. Running
under `python -O` removes all asserts, so a misuse that should fail loudly becomes
a silent corruption of state. This is the same class of issue addressed by
PART-2 FIX 3 in `networks.py`.

**Fix.**

```python
if self.done:
    raise RuntimeError("Episode has ended — call reset() before step_agent().")
```

---

## Performance Improvements

### 🟡 P1 — Vectorize `_extract_local_patch` ✅ FIXED

**Location:** `agents/networks.py` (`CnnQNetwork._extract_local_patch`).

**Was:** a Python `for b in range(B)` loop slicing each sample then `torch.cat`.
On GPU this serializes `B` tiny kernels per forward (and the forward runs for both
current and next states every gradient step).

**Now:** a single advanced-indexing gather. The padded map is indexed with
broadcast index tensors `(B,1,1,1) × (1,C,1,1) × (B,1,ps,1) × (B,1,1,ps)` →
`(B, C, ps, ps)` in one call. Verified **bit-identical** to the loop on random
inputs and corner positions.

---

### 🟡 P2 — Replay buffer stores full flat float32 maps (large RAM footprint)

**Location:** `agents/replay_buffer.py`, `agents/dqn_agent.py:push_transition` /
`update`.

**Description.** Each transition stores the global + local maps for **both** the
current and next state as `float32`: `(5120 + 5120) × 2 ≈ 20.5k` floats ≈ **80 KB**
per transition (the global stream grew from 4 to 5 channels with `Own_Position`).
At `buffer_size = 200000` that is **~16 GB** of host RAM just for observations, plus
a fresh `np.array(..., float32)` copy of the whole minibatch on every `sample`.

**Recommendation.**
- Store observations as `uint8`/`float16` (the channels are mostly `{0,1}`; only
  Trajectory is continuous in `[0,1]` and tolerates `float16`), casting to
  `float32` at sample time. ~2–4× memory reduction.
- Or store each state once and reference it from both the `s` and `s'` slots
  (n-step windows already hold overlapping observations).
- Consider a preallocated ring of NumPy arrays instead of a `deque` of tuples to
  avoid per-sample Python object overhead and the per-batch `np.array` rebuild.

---

### 🟡 P3 — `_communicate` recomputed every agent-step with an O(n²) Python loop

**Location:** `env/grid_env.py:338-370`.

**Description.** `_communicate` runs a Python double loop over agent pairs and a
full-map `max` merge per group, once per `step_agent` call (i.e. `n` times per
round → O(n³) full-map ops per round). Harmless at `n ≤ 4`, but it scales poorly
and runs even when no agent moved into a new comm neighborhood.

**Recommendation.** Cheap to defer for the current team sizes; if `n_agents` max
grows, vectorize the pairwise distance (`scipy`/broadcasted NumPy) and skip the
merge when group membership is unchanged since the last call.

---

## Code Quality

### 🔵 C1 — Replace remaining `assert` guards with explicit raises

Beyond B2, audit for other asserts used as runtime guards (rather than test-only
invariants) and convert to `raise`. PART-2 FIX 3 already did this for
`GlobalMapCNN.__init__` (`assert ch_in == ch` → `raise ValueError`).

---

### 🔵 C2 — Deduplicate the per-episode stepping loop

**Location:** `main.py` (`run_eval`, `run_simulate`), `eval_checkpoints.py`
(`evaluate_checkpoint`), `training/train.py` (`evaluate`).

**Description.** Four near-identical inner loops build `ctx_np` / `masks`, call
`select_action(s)`, then `step_agent` per drone. They have drifted slightly (single
vs batched selection, GUI hooks). (Simpler now that position rides in the obs
channel — no per-step `positions` list is threaded through selection.)

**Recommendation.** Factor a shared helper, e.g.

```python
def run_greedy_episode(env, agent, ctx_params, ctx_norm, grid_size,
                       device, on_render=None) -> tuple[float, dict]:
    """One greedy episode; returns (total_reward, final_info)."""
    ...
```

and have eval/play/simulate/offline-eval call it. Reduces drift and the surface
for bugs like B-class reachability handling.

---

### 🔵 C3 — No automated test suite

**Location:** repo root (no `tests/`).

**Description.** The audit smoke checks were never committed (noted in
[KNOWN_BUGS_AND_FIXES.md](KNOWN_BUGS_AND_FIXES.md) and
[REFACTOR_REPORT.md](REFACTOR_REPORT.md)). The fixes in this pass were verified by
ad-hoc scripts.

**Recommendation.** Add `tests/` covering at least:
- `_extract_local_patch` equivalence (loop reference vs vectorized) — already
  scripted, just commit it.
- env `reset`/`step_agent` return shapes & `observation_space.contains`.
- `NStepBuffer._emit` returns vs `bootstrap_disc` for terminal/truncated windows.
- checkpoint round-trip (`save` → `load` restores `training_state`;
  `load_weights_only` leaves ε untouched).
- `ctx_to_numpy` ↔ `(ctx·gs).long()` position round-trip.
- `select_actions_batch` with all-explore (zero greedy) — guarded by FIX 5.

---

### 🔵 C4 — Stale `make_ctx_tensor` references in env docstrings

**Location:** `env/grid_env.py:54` and `:387`.

**Description.** Both docstrings tell callers to pass position to
`make_ctx_tensor`, but that function was removed (KNOWN_BUGS #6); the live path is
`ctx_to_numpy` + `select_action`/`select_actions_batch`.

**Fix.** Replace `make_ctx_tensor / DQNAgent.select_action` with
`ctx_to_numpy / DQNAgent.select_action` in both docstrings.

---

## Minor Inconsistencies

### 🔵 M1 — Channel naming: "Recency" vs "Trajectory" ✅ PARTIALLY FIXED

**Description.** `env/grid_env.py` calls channel 2 **Trajectory**;
`agents/networks.py` called it **Recency**. PART-2 FIX 4 harmonized the
`GLOBAL_CHANNELS` / `LOCAL_CHANNELS` comment blocks in `networks.py` to
**Trajectory** (Invariant 3 consistency).

**Remaining:** `agents/dqn_agent.py:157-158` (`print_architecture`) still prints
`[Visited, Obstacle, Recency, Target]`. Recommend updating those two print strings
to `Trajectory` for full consistency:

```python
print(f"  global channels  :  {GLOBAL_CHANNELS}  [Visited, Obstacle, Trajectory, Target]")
print(f"  local  channels  :  {LOCAL_CHANNELS}  [Visited, Obstacle, Trajectory, Target, Other_Pos]")
```

---

### 🔵 M2 — "BatchNorm" comments describe a network that has no BatchNorm

**Location:** `agents/dqn_agent.py:205-207` and `:262` (and the historical note in
KNOWN_BUGS_AND_FIXES #2).

**Description.** The `eval()/train()` toggle around inference is correct and still
necessary — but for **`DropPath` (stochastic depth)**, not BatchNorm. The
`convnext_attn_net` network uses only LayerNorm / GRN (no BatchNorm, no running
stats). The comments mislead a reader into thinking running statistics are at
stake.

**Fix.** Reword to reference stochastic depth, e.g. *"Switch to eval() so DropPath
is disabled (identity) during inference; restore the previous mode afterwards."*
The toggle itself should stay.

---

### 🔵 M3 — Position normalization range `[0, (gs-1)/gs]` ✅ OBSOLETED

**Description.** Position used to be normalized into the context as `pos /
grid_size` (range `0..0.969`) and reconstructed for the patch crop with
`(ctx[:, 3:5] * gs).long()` — exact **only because `grid_size = 32` is a power of
two**, fragile for a non-power-of-two grid.

**Resolution.** Position no longer lives in the context. It is the `Own_Position`
global observation channel (a single `1.0`), and `CnnQNetwork.forward` recovers the
crop coordinates with `argmax` over that channel followed by `divmod(idx, gs)`. An
integer index split is exact for **any** `grid_size`, so the power-of-two caveat is
gone and no normalization/denormalization round-trip is involved.

---

### 🔵 M4 — `exploration_bonus` default mismatch (code vs config)

**Location:** `env/grid_env.py:85` (`.get("exploration_bonus", 0.05)`) vs
`configs/default.yaml` (`exploration_bonus: 0.02`).

**Description.** The config value (0.02) wins at runtime, but the in-code fallback
(0.05) disagrees with both the config and the documented value, which is confusing
if the key is ever omitted. Align the fallback to `0.02` (or drop the fallback and
require the key).

---

### 🔵 M5 — `eval_episodes` is silently clamped to the fixed-seed count

**Location:** `training/train.py:142` (`n_episodes = min(n_episodes,
len(eval_seeds))`), `_EVAL_SEEDS = range(100, 120)` (20).

**Description.** Setting `training.eval_episodes > 20` has no effect — `evaluate`
clamps to the 20 fixed seeds. Intentional (comparable maps), but undocumented at
the config site. Recommend a comment in `default.yaml`, or extend `_EVAL_SEEDS`
when a larger eval set is wanted.

---

### 🔵 M6 — README config table is stale (variable grid size)

**Location:** `README.md` ("Parametri configurabili" lists
`grid_min_size`/`grid_max_size`).

**Description.** Those keys no longer exist; the grid is a fixed `32×32`
(`grid_size`). The README still describes the older variable-size design.
[ARCHITECTURE.md](ARCHITECTURE.md) and [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
reflect the current design; the README table should be updated to match.

---

## Summary table

| ID | Sev | Item | Status |
|----|-----|------|--------|
| B1 | 🔴 | Random spawn not guaranteed reachable | open |
| B2 | 🔴 | `assert not self.done` stripped under `-O` | open |
| P1 | 🟡 | Vectorize `_extract_local_patch` | ✅ fixed |
| P2 | 🟡 | Replay buffer RAM footprint | open |
| P3 | 🟡 | `_communicate` O(n²) per step | open (low pri) |
| C1 | 🔵 | assert → raise (GlobalMapCNN) | ✅ fixed (FIX 3) |
| C2 | 🔵 | Deduplicate episode loops | open |
| C3 | 🔵 | No test suite | open |
| C4 | 🔵 | Stale `make_ctx_tensor` docstrings | open |
| M1 | 🔵 | Recency→Trajectory naming | ✅ partial (networks.py) |
| M2 | 🔵 | "BatchNorm" comments (no BatchNorm) | open |
| M3 | 🔵 | Position normalization range | open |
| M4 | 🔵 | `exploration_bonus` default mismatch | open |
| M5 | 🔵 | `eval_episodes` clamp undocumented | open |
| M6 | 🔵 | README stale grid-size table | open |

**Applied in this pass (PART 2):** FIX 1 (rename `enviroment.yaml` →
`environment.yaml`), FIX 2 (P1), FIX 3 (C1), FIX 4 (M1, networks.py), FIX 5
(guard empty `greedy_idx` in `select_actions_batch`).

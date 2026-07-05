> **Historical document** — inherited from the RL-course project *before* the
> Multi-Agent-Systems fault extension (random faults, Broken channel, CTX_DIM=5,
> obstacle-density/fault DR, eval-only BFS auto-nav). Numbers such as channel
> counts or context dims may be stale here; the current references are
> [ARCHITECTURE.md](ARCHITECTURE.md), [ENV.md](ENV.md), [TRAINING.md](TRAINING.md).

# Testing / Evaluation Phase

Methodology and tooling for the **post-training evaluation** of the multi-drone
search policy (Dueling Double DQN, parameter sharing, ConvNeXt+MHSA network).
The policy is trained with per-episode domain randomization over
`vision_radius`, `comm_range`, `n_agents` (fed to the network as a context
vector), so a single set of weights is meant to generalize across team sizes,
sensing and communication budgets. This phase **measures how well it actually
does**, as a function of every parameter, with statistically sound estimates.

> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md), [TRAINING.md](TRAINING.md),
> [ENV.md](ENV.md), [QUICK_REFERENCE.md](QUICK_REFERENCE.md).

The tooling is **self-contained and separate** from the training/eval code — it
only *imports* existing helpers, never edits them:
- `testing/evaluate_policy.py` — the sweep harness (produces CSVs).
- `testing/analysis.ipynb` — paper-style plotting notebook (CSVs → PNGs).
- outputs in `testing_results/csv/` and `testing_results/plots/` (kept apart
  from the training-time `eval_results/`).

---

## 1. Why not a full grid search?

We vary six factors: `n_agents`, `vision_radius`, `comm_range`,
`obstacle_density`, `max_steps`, and the training **epoch**. The full Cartesian
product explodes: e.g. `4 · 6 · 6 · 6 · 5 = 4320` configurations *before*
multiplying by the per-configuration seed count and the number of checkpoints.

We use a **One-Factor-At-A-Time (OFAT) sensitivity design** instead: fix every
axis at a **reference operating point** and sweep one factor at a time. Cost
drops to the *sum* of the axis lengths, `≈ 4+6+6+6+5 = 27` configuration points —
a **~160× reduction** — while still tracing every axis the policy was randomized
over. A small number of targeted **2-D interaction heatmaps** recover the
pairwise effects an OFAT sweep cannot see (e.g. do more drones compensate for a
shorter comm range?).

### Reference operating point
Matches the training operating point (phase 2) with the DR axes at the centre of
their ranges:

| factor | reference |
|--------|-----------|
| `n_agents` | 3 |
| `vision_radius` | 3 |
| `comm_range` | 5 |
| `obstacle_density` | 0.2 |
| `max_steps` | 200 |

### Sweep grids

| Axis | In-distribution | Out-of-distribution (OOD) | Varied via |
|------|-----------------|---------------------------|------------|
| `n_agents` | 1, 2, 3, 4 | 5, 6 | `set_domain_params` + context |
| `vision_radius` | 1, 2, 3, 4, 5, 6 | 7, 8 | `set_domain_params` + context |
| `comm_range` | 2, 4, 6, 8, 10, 12 | 0 (no comm), 14, 16 | `set_domain_params` + context |
| `obstacle_density` | 0.0, 0.1, 0.15, 0.2, 0.25, 0.3 | 0.35, 0.4 | env config |
| `max_steps` | 50, 100, 150, 200, 300 | — | env config |
| **epoch** | every `phase2_*_ep*.pt` at the reference point | — | checkpoint |

The DR training ranges are `vision_radius ∈ [1,6]`, `comm_range ∈ [2,12]`,
`n_agents ∈ [1,4]` (`configs/default.yaml`), and `obstacle_density = 0.2`,
`max_steps = 200` during phase 2 — so the OOD columns genuinely probe
**extrapolation** beyond what the network ever saw. Their context values
normalize to `> 1.0` on purpose (the denominators stay fixed at the DR maxima via
`build_ctx_norm`).

### 2-D interaction heatmaps
Two small grids (not the full product):
- `n_agents × comm_range` (cooperation: team size vs. information sharing).
- `vision_radius × obstacle_density` (sensing vs. clutter).

---

## 2. Paired seed design

A single fixed **seed bank** (default `2000 .. 2000+S−1`) is reused at every point
of a sweep. For `n_agents / vision_radius / comm_range` the **map for a given
seed is identical** — only the agents' capability changes — so the comparison is
properly *paired* and differences are attributable to the parameter, not to map
variance. For `obstacle_density` the map necessarily changes with the seed
(density is an input to `generate_grid`), so that axis is unpaired by
construction — a fundamental, unavoidable property, noted here for honesty.

Evaluation is **greedy** (`epsilon = 0`), matching `eval_checkpoints.py` and
removing exploration noise from the estimates (cf. the `eval_epsilon = 0.05` used
*during* training, which is intentionally non-zero there).

---

## 3. How many environments (seeds)?

Success rate is a binomial proportion `p`; its standard error is
`SE = √(p(1−p)/n)`, maximized at `p = 0.5`. Worst-case 95% CI half-widths:

| n (seeds) | max SE | 95% CI half-width |
|-----------|--------|-------------------|
| 100 | 5.0% | ±9.8% |
| 200 | 3.5% | ±6.9% |
| 300 | 2.9% | ±5.7% |
| 500 | 2.2% | ±4.4% |
| 1000 | 1.6% | ±3.1% |

**Recommendation:** **300** seeds/point is the practical sweet spot for
iteration (CI ≈ ±5–6%); **500–1000** for the final, headline table. **100** is a
noisy floor, only for quick smoke checks. The harness defaults to **500**.

We report the **Wilson score interval**, not the normal approximation, because
success rates sit near 0 and 1 where the normal approximation misbehaves (it can
even leave `[0,1]`). With `n ≥ 300` the two largely agree, but Wilson stays
correct in the tails (e.g. an axis where success ≈ 0% or 100%).

> "1000 vs 100": 100 is the minimum to say anything; 300 the sensible default;
> 1000 for publication-grade tightness. Past ~1000 the CI shrinks slowly
> (`∝ 1/√n`) — diminishing returns.

---

## 4. Metrics

| Metric | Definition | Why it matters |
|--------|------------|----------------|
| **success_rate** | fraction of episodes that reach the target, + **Wilson 95% CI** | primary task outcome |
| **mean_reward** | mean ± std of episode return | dense signal; sanity vs. success |
| **coverage** | `known_cells / free_cells` at episode end | how much of the map was explored |
| **coverage_eff** | `coverage / agent-steps` | exploration *speed* |
| **TTF (steps-to-success)** | mean agent-steps to reach target, **over successful episodes only** | failures cap at the budget, so averaging them in is misleading |
| **SPL** | `mean( S · d / max(d, p) )` | path optimality (the navigation gold standard) |
| **collisions** | mean same-cell overlaps entered / episode | cooperation: do drones avoid stacking? |
| **revisits** | mean per-agent cell re-entries / episode | redundancy / wasted motion |
| **overlap** | fraction of team-visited cells visited by >1 drone | dispersion: are drones covering *different* areas? |
| **generalization gap** | in-distribution vs. OOD aggregates | the headline result for a DR+context policy |

### SPL (Success weighted by Path Length)
Anderson et al. (2018). Per episode `i`: `S_i` is success (0/1), `d_i` the
**optimal** spawn→target distance (4-connected BFS shortest path), `p_i` the
number of moves taken by the drone that *found* the target. `SPL = (1/N) Σ S_i ·
d_i / max(d_i, p_i)` lies in `[0,1]`: 1.0 means every success took the shortest
possible path; 0 means none succeeded. `d_i` is computed by `bfs_distance` in the
harness (a local helper; `env.utils.bfs_reachable` is imported but not modified).

### What is easy to forget (and why it's here)
- **Conditioning TTF on success** — the single most common mistake; otherwise
  the time-to-find metric is dominated by timeouts.
- **Confidence intervals**, not point estimates — a success rate without a CI is
  not interpretable at `n = 100`.
- **SPL / path optimality** — success alone rewards a drone that stumbles onto
  the target after wandering the whole map.
- **Cooperation metrics** — directly tied to the `collision_penalty` /
  `revisit_penalty` reward terms; they show whether the *team* behavior the
  reward was designed to induce actually emerged.
- **OOD / generalization** — the whole rationale for the DR + context-vector
  design; without it the test under-reports what the architecture buys you.

### Documented but intentionally not built (optional extensions)
- **IQM + stratified bootstrap CIs / performance profiles** (Agarwal et al.,
  2021, *Deep RL at the Edge of the Statistical Precipice*) — more robust than
  means across heterogeneous instances; add if a reviewer wants distributional
  robustness. The per-episode `detail_*.csv` already contains everything needed
  to compute them.
- **Explicit failure-mode breakdown** (% timeout vs. % structurally
  unreachable). Reachability is only guaranteed from `(0,0)`; a random non-top-
  left spawn can in principle be walled off from the target (see
  [BUGS_AND_IMPROVEMENTS.md](BUGS_AND_IMPROVEMENTS.md)). Computable offline from
  `detail_*.csv` + a BFS check if needed.

---

## 5. Running it

```bash
# All five OFAT axes (incl. OOD), 500 seeds, default checkpoint:
python testing/evaluate_policy.py --axis all --seeds 500

# A single axis at headline precision:
python testing/evaluate_policy.py --axis vision --seeds 1000

# Learning curve across training (one checkpoint glob):
python testing/evaluate_policy.py --axis epoch  --seeds 300

# 2-D interaction heatmaps:
python testing/evaluate_policy.py --axis heatmaps --seeds 300

# In-distribution only / pick a checkpoint / force device:
python testing/evaluate_policy.py --axis density --no-ood
python testing/evaluate_policy.py --axis all --checkpoint checkpoints/Phase3/best.pt --device cuda
```

| Flag | Meaning (default) |
|------|-------------------|
| `--axis` | `all` \| `n_agents` \| `vision` \| `comm` \| `density` \| `max_steps` \| `epoch` \| `heatmaps` (`all`) |
| `--seeds` | episodes per configuration point (`500`) |
| `--seed-start` | first seed in the bank (`2000`) |
| `--ood` / `--no-ood` | include OOD points (`on`) |
| `--checkpoint` | weights to test (`checkpoints/Phase3/best.pt`) |
| `--ckpt-glob` | checkpoint glob for the epoch axis (`checkpoints/Phase3/phase2_ConvNeXT_10k_dr_ep*.pt`) |
| `--device` | `cuda` \| `cpu` (auto) |
| `--out-dir` | output root (`testing_results`) |

### Outputs
- `testing_results/csv/detail_<axis>.csv` — one row per (config-point, seed).
- `testing_results/csv/agg_<axis>.csv` — one row per config-point (all metrics +
  Wilson CI).
- `testing_results/csv/heatmap_<pair>.csv` — 2-D interaction grids.
- `testing_results/csv/_run_meta.txt` — checkpoint, seed bank, reference config.

### Plotting
Open **`testing/analysis.ipynb`** and Run-All. It reads only the CSVs and writes
publication-ready PNGs to `testing_results/plots/`:
- `fig_<axis>.png` — 6-panel sensitivity figure per axis (success+CI, return,
  coverage, SPL, TTF, cooperation), with the OOD region shaded and the training
  range marked.
- `fig_epoch.png` — learning curve at the reference point.
- `fig_heatmap_<pair>.png` — annotated interaction heatmaps.
- plus `summary_table.csv` (in-dist vs. OOD aggregates — the generalization gap).

The notebook locates the repo root automatically, so it runs whether launched
from the project root or from inside `testing/`.

---

## 6. Suggested reporting order (for a paper / report)

1. **Headline table** — reference-point metrics at `n = 1000` (success + CI,
   reward, coverage, SPL).
2. **Learning curve** (`fig_epoch.png`) — convergence on held-out maps.
3. **Per-axis sensitivity** (`fig_*.png`) — one figure per factor; discuss the
   monotonic trends (e.g. success ↑ with vision) and where they saturate.
4. **Generalization** — the OOD columns / `summary_table.csv`: how gracefully the
   single policy degrades beyond its training ranges.
5. **Interactions** (`fig_heatmap_*.png`) — e.g. whether extra drones substitute
   for comm range.
6. **Cooperation analysis** — collisions / revisits / overlap vs. team size.

---

## 7. References

- P. Anderson et al., *On Evaluation of Embodied Navigation Agents*, 2018 — SPL.
- R. Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*,
  NeurIPS 2021 — IQM, stratified bootstrap CIs, performance profiles.
- P. Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018 —
  fixed eval sets, report CIs, reproducibility.
- R. Kirk et al., *A Survey of Generalisation in Deep RL*, 2023; K. Cobbe et al.,
  *Leveraging Procedural Generation… (Procgen)*, ICML 2020 — train/test
  generalization gap.
- F. Pardo et al., *Time Limits in RL*, ICML 2018 — truncation ≠ termination
  (already respected in training; see Invariant 7).

# Quick Reference

One-page cheat sheet for the Multi-Drone Search DQN **with random faults**
(Multi-Agent Systems project). Deeper detail in
[ARCHITECTURE.md](ARCHITECTURE.md), [TRAINING.md](TRAINING.md),
[ENV.md](ENV.md), [BUGS_AND_IMPROVEMENTS.md](BUGS_AND_IMPROVEMENTS.md).

---

## Setup

```bash
conda env create -f environment.yaml && conda activate rl-drone   # PyTorch + CUDA 11.8
# or:  pip install -r requirements.txt
```

---

## Train / resume / warm-start

```bash
python main.py --mode train                                   # fresh run
python main.py --mode train --resume checkpoints/last.pt      # continue interrupted run
python main.py --mode train --init-weights checkpoints/best.pt  # warm-start fresh run (weights only)
python main.py --mode train --config configs/default.yaml     # explicit config
```

| Flag | Restores |
|------|----------|
| `--resume PATH` | weights + optimizer + LR schedule + ε + `train_steps` + **curriculum position** (use `last.pt`) |
| `--init-weights PATH` | **only** `q_net` weights; everything else fresh |

`--resume` and `--init-weights` are mutually exclusive. Curriculum is staged by
editing `n_episodes` per phase in the config (`0` = skip).

---

## Eval / play / simulate

```bash
# Greedy eval with GUI (needs a checkpoint)
python main.py --mode eval --checkpoint checkpoints/best.pt
python main.py --mode eval --checkpoint checkpoints/best.pt --phase 1 \
               --vision-radius 4 --comm-range 6 --n-agents 3 \
               --fault-prob 0.003 --auto-nav

python main.py --mode play       # arrow keys drive drone 0, others random
python main.py --mode simulate   # GUI setup screen (fault slider + auto-nav toggle)
```

| Flag (eval/play) | Meaning |
|------|---------|
| `--checkpoint PATH` | checkpoint to load (required for eval) |
| `--phase N` | curriculum phase whose `n_agents`/`density`/`max_steps` apply (default: last) |
| `--vision-radius / --comm-range / --n-agents` | override the eval context |
| `--fault-prob P` | per-step drone fault probability (default: config `env.fault_prob`) |
| `--auto-nav` | drones that know the target follow the BFS shortest path (eval-only) |

## Rank checkpoints offline

```bash
python eval_checkpoints.py                                   # batch, default pattern
python eval_checkpoints.py --pattern "mas_*_ep*.pt" --episodes 200 --last_n 10
python eval_checkpoints.py --no_dr --device cuda --fault_prob 0.002 --auto_nav
python eval_checkpoints.py --watch --pattern "mas_*_ep*.pt"   # live during training
```

## Fault-robustness sweep (the MAS experiment)

```bash
python testing/evaluate_policy.py --axis fault --seeds 500              # policy only
python testing/evaluate_policy.py --axis fault --seeds 500 --auto-nav   # hybrid
```

Outputs to `eval_results/`: `detail_ep*.csv`, `ranking_<ts>.csv`, and (watch mode)
`watch_summary.csv`. Auto-spawned by the trainer when
`training.eval_during_training: true`.

---

## Config structure (`configs/default.yaml`)

| Section | Key knobs (defaults) |
|---------|----------------------|
| `env` | `grid_size: 32` (fixed), `obstacle_density: 0.2`, `n_agents: 3`, `vision_radius: 3`, `comm_range: 5`, `comm_metric: manhattan`, `max_steps: 300`, reward terms, `shared_target_reward: true`, `fault_prob: 0.0`, `seed: 42` |
| `domain_randomization` | `vision_radius: [1,6]`, `comm_range: [2,12]`, `n_agents: [1,4]` (max = context normalizer), `obstacle_density: [0.10,0.30]`, `fault_prob: [0.0,0.003]` |
| `curriculum` | list of phases: `name`, `n_episodes` (`0`=skip), `n_agents`, `obstacle_density`, `max_steps`, `epsilon_reset`, `dr_override` |
| `agent` | `lr: 1e-4`, `lr_scheduler{max_lr:5e-4, min_lr:1e-6, warmup_pct:0.1}`, `gamma: 0.97`, `n_step: 3`, `epsilon_start/end/decay: 1.0/0.05/0.999`, `batch_size: 16`, `buffer_size: 200000`, `target_update_freq: 500`, `fc_hidden: 1536`, `grad_clip: 1`, `weight_decay: 1e-6` |
| `training` | `update_every: 4`, `eval_every: 100`, `eval_episodes: 20`, `eval_epsilon: 0.05`, `save_every: 200`, `log_dir`, `checkpoint_dir`, `eval_during_training: true`, `eval_during_training_device`, `eval_during_training_episodes` |

### Reward terms (`env`)

| Key | Default | Applied |
|-----|---------|---------|
| `step_penalty` | `-0.3` | every move (base) |
| `wall_penalty` | `-0.02` | invalid move (no move happens) |
| `revisit_penalty` | `-0.05` | × prior visit count (linear; cumulative O(k²)) |
| `collision_penalty` | `-0.5` | × other **live** drones on the entered cell |
| `exploration_bonus` | `0.02` | × newly globally-discovered cells |
| `target_reward` | `200.0` | reaching the target (terminates) |

---

## Channel layout

Constants (`agents/networks.py` **and** `env/grid_env.py` — keep in sync):
`GLOBAL_CHANNELS=6`, `LOCAL_CHANNELS=5`, `LOCAL_PATCH_SIZE=13`, `CTX_DIM=5`,
`OWN_POSITION_CHANNEL=4`, `BROKEN_CHANNEL=5`, `_TRAJ_CAP=5.0`, `grid_size=32`.

**Global stream** — flat shape `6 · 32² = 6144`:

| # | Channel | Encoding |
|---|---------|----------|
| 0 | Visited | `{0,1}` free cells seen (comm-shared) |
| 1 | Obstacle | `{0,1}` discovered obstacles |
| 2 | Trajectory | `clip(visit_count / 5.0, 0, 1)` |
| 3 | Target | `{0,1}` target once seen |
| 4 | Own_Position | `{0,1}` this drone's own cell (argmax → patch crop) |
| 5 | Broken | `{0,1}` crash sites known to this drone (beacon + comm fusion) |

**Local stream** — flat shape `5 · 32² = 5120` (cropped to `13×13` around the agent
inside the network):

| # | Channel | Encoding |
|---|---------|----------|
| 0–3 | Visited / Obstacle / Trajectory / Target | same as global |
| 4 | Other_Position | `{0,1}` other **live** drones within vision |

---

## Context vector (`CTX_DIM = 5`)

`ctx_to_numpy(ctx_params, ctx_norm, agent_id, n_alive)`:

| Index | Component | Value |
|-------|-----------|-------|
| 0 | `vision_radius` | `vr / 6` (DR max) |
| 1 | `comm_range` | `cr / 12` (DR max) |
| 2 | `n_agents` | `na / 4` (DR max) |
| 3 | `agent_id` | `i / 4` (breaks parameter-sharing symmetry) |
| 4 | `n_alive_belief` | `env.n_alive_belief(i) / 4` (this drone's own belief) |

Position is **not** in the context — it rides in the `Own_Position` global
channel, from which `forward()` recovers the crop coordinates via argmax.
`n_alive_belief` can change mid-episode, so the context is rebuilt **every
round per drone**. Normalizers come from `build_ctx_norm` (= max of each DR
range); `agent_id`/`n_alive` reuse the `n_agents` one.

---

## Fault model in one breath

Per step, a live drone breaks with `fault_prob` → it stops forever (no move /
sense / comm), its turns still tick the clock, the wreck doesn't block or
collide. Crash knowledge is decentralized: a distress beacon within
`comm_range` (or the vision box) writes the crash into the *observer's* Broken
map, which then spreads via comm fusion. Fault = truncation for that drone
(flush n-step with bootstrap); all-dead = early truncation; dead drones get no
`shared_target_reward`.

---

## Checkpoint `.pt` dict (`DQNAgent.save`)

| Key | Content |
|-----|---------|
| `q_net` | online network `state_dict` |
| `optimizer` | Adam state |
| `warmup_sched` / `cosine_sched` | LR sub-scheduler states (or `None`) |
| `sched_episode` | LR-schedule step count |
| `epsilon` | current ε |
| `train_steps` | gradient-step counter |
| `training_state` | *(optional)* `{global_episode, phase_idx, ep_in_phase, best_success, best_reward}` |

| File | When | Has `training_state` |
|------|------|----------------------|
| `best.pt` | new eval best | no |
| `{name}_ep{N}.pt` | every `save_every` | no |
| `last.pt` | every `save_every` + phase end | **yes** (used by `--resume`) |

Replay buffer is **not** saved.

---

## Network at a glance

```
global (B,6,32,32) → GlobalMapCNN [ConvNeXt 96→192→384, FiLM(ctx)/stage,
                                    MHSA over 8x8 tokens +CLS +ctx] → 1024
                     own pos recovered from Own_Position channel (argmax)
local  (B,5,32,32) → crop 13x13 → LocalCNN [stride-1 ConvNeXt, GRN,
                                    FiLM(global)] → 384
cat[global(1024), local(384), ctx(5)] = 1413 → LN → FC 1536→768→384
→ Dueling head: Q = V + A − mean(A)        # n_actions = 4
```

Actions: `0 up · 1 down · 2 left · 3 right`. Termination: `terminated` = target
reached; `truncated` = `step_count ≥ max_steps · n_agents` **or** all drones
broken.

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

Per-phase: `{name}/episode_reward|epsilon|episode_length|found_target|lr|
dr_vision_radius|dr_comm_range|dr_n_agents|dr_obstacle_density|dr_fault_prob|
n_broken` and (on eval) `{name}/eval_success_rate|eval_mean_reward`. Global
mirrors: `train/episode_reward|epsilon|lr`, `eval/success_rate|mean_reward`.

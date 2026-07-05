# Training Pipeline

Everything that happens between `python main.py --mode train` and a saved
checkpoint: the curriculum, domain randomization, n-step returns, the ε and LR
schedules, the checkpoint/resume system, the evaluation protocol, and the
TensorBoard scalars. Code lives in `training/train.py` (loop),
`agents/dqn_agent.py` (gradient step), `agents/nstep.py` (n-step windows),
`agents/scheduler_utils.py` (LR), and `eval_checkpoints.py` (offline ranking).

> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) (network, invariants),
> [ENV.md](ENV.md) (rewards, observations), [QUICK_REFERENCE.md](QUICK_REFERENCE.md).

---

## 1. Top-level flow (`train()`)

```
load config → set_seeds → pick device (cuda if available)
make runs/ + checkpoint_dir, open TensorBoard SummaryWriter
(optional) spawn parallel evaluator  [training.eval_during_training]
grid_size = config.env.grid_size ; ctx_norm = build_ctx_norm(config)
total_episodes = sum(phase.n_episodes)        # sizes the LR schedule
build DQNAgent(grid_size, n_actions=4, config, device, total_episodes)
agent.print_architecture()
resolve checkpoint mode: --init-weights (warm start) | --resume (continue) | fresh
for phase_idx, phase in enumerate(curriculum, 1):
    skip if n_episodes == 0  or  (resuming) already-completed / already-finished
    run_phase(...)            # the actual episodes
finally: signal + wait for parallel evaluator, print final ranking
```

`n_actions` is fixed to 4 (up/down/left/right). `max_agents` (the largest
`n_agents` across phases) is computed but the network is agent-count-agnostic —
team size only enters through the context vector and the per-episode loop range.

---

## 2. Curriculum phases

`config["curriculum"]` is an **ordered list** of phase dicts. A run is staged by
editing these, not by changing code.

| Key | Effect |
|-----|--------|
| `name` | Phase name; used in TensorBoard tags and checkpoint filenames. |
| `n_episodes` | Episodes to run. **`0` ⇒ skip this phase.** |
| `n_agents` | Team size for the phase (also overridden per episode by DR). |
| `obstacle_density` | Phase obstacle density (optional; falls back to `env`). |
| `max_steps` | Phase per-drone step budget (optional; falls back to `env`). |
| `epsilon_reset` | ε is reset to this at phase **start** (optional). |
| `dr_override` | Per-phase DR ranges; else the global `domain_randomization`. |

`make_phase_config` deep-copies the base config and applies the phase's
`n_agents` / `obstacle_density` / `max_steps` to the `env` section. `grid_size`
is **never** set per-phase. `get_dr_cfg` returns `phase.dr_override` if present,
else the global DR config.

**Resume interaction.** When resuming, phases with `phase_idx < resume_phase` are
skipped (already done); the active phase restarts at `ep_in_phase + 1` with
`apply_eps_reset=False` (the checkpoint's ε is kept, not reset to
`epsilon_reset`); fully-finished phases are skipped.

---

## 3. Domain randomization (DR)

Per episode the loop samples a context and applies it to the env **before**
`reset()`:

```python
ctx_params = sample_ctx_params(dr_cfg)        # uniform randint in [min, max]
env.set_domain_params(**ctx_params)           # FOV / comm / team now match
obs, info = env.reset()
n_agents  = env.n_agents                       # read back the sampled team size
```

`sample_ctx_params(dr_cfg, rng=None)` draws `vision_radius`, `comm_range`,
`n_agents` with **inclusive** `randint`. An optional `random.Random` instance
makes sampling deterministic (used by evaluation, see §7). Default ranges
(`configs/default.yaml`):

| Param | Range | Context normalizer (= max) |
|-------|-------|----------------------------|
| `vision_radius` | `[1, 6]` | 6 |
| `comm_range` | `[2, 12]` | 12 |
| `n_agents` | `[1, 4]` | 4 |

`set_domain_params` recomputes the env's vision offsets when the radius changes,
updates `comm_range`, and updates `n_agents` (the next `reset()` reallocates the
per-agent map arrays). Applying DR to the env is what makes the context vector
*coherent* with what the drone actually observes (see Invariant 5 in ARCHITECTURE).

---

## 4. `ctx_to_numpy` and the context vector

```python
ctx_to_numpy(ctx_params, ctx_norm) -> (3,) float32
  = [ vision_radius / ctx_norm["vision_radius"],     # /6
      comm_range    / ctx_norm["comm_range"],        # /12
      n_agents      / ctx_norm["n_agents"] ]         # /4
```

`ctx_norm` is passed in explicitly (no module-level constants) so every value
traces back to the config. The drone position is **not** in the context — it
rides in the `Own_Position` global observation channel, from which `forward()`
recovers the integer crop coordinates via argmax. None of the three components
changes within an episode, so the context is **constant per episode** (one vector
shared by every agent, stored as-is in each transition). No separate position
field is needed in the replay buffer: position is already inside `obs_global`.

---

## 5. n-step returns (`agents/nstep.py`)

Because drones act round-robin, an agent's "next n steps" are n **rounds** away.
`NStepBuffer` keeps a per-agent sliding window of single-step records and emits
n-step transitions matching `ReplayBuffer.push` / `DQNAgent.update`:

```
(obs_g, obs_l, ctx, action, R_n, next_obs_g, next_obs_l, next_ctx, bootstrap_disc)
R_n            = Σ_{k=0}^{m-1} γ^k r_{t+k}        (truncated at a terminal)
bootstrap_disc = γ^m   if the window did NOT hit a terminal, else 0.0
```

So the agent's TD target is simply `R_n + bootstrap_disc · Q(next)`; γ and the
done-mask are folded into the stored values. With `n_step = 1` this reduces
exactly to `r + γ · Q(s') · (1 − terminated)`.

- `push(...)` appends a step and emits one transition once the window reaches
  `n_step`.
- `flush(agent_i)` drains the remaining short windows at episode end (so the tail
  of each trajectory is not lost).
- **`terminated`, not `truncated`, ends a window.** Time-limit truncation is
  non-terminal and must still bootstrap (Pardo et al. 2018).

### Cooperative terminal reward (`shared_target_reward`)

When `true` and a drone finds the target, `apply_team_terminal(finder, bonus)`:
adds `target_reward` to every *other* agent's most-recent pending step and marks
each agent's last step terminal (so its window stops bootstrapping). This shares
credit for the team success. It **requires `n_step >= 2`** — with `n_step == 1`
every step is emitted immediately and nothing remains in the windows to credit
(the loop prints a warning if misconfigured). Default config: `n_step = 3`,
`shared_target_reward: true`.

---

## 6. Schedules

### ε-greedy (`DQNAgent.decay_epsilon`)

Per-episode multiplicative decay, floored:
`ε ← max(epsilon_end, ε · epsilon_decay)`. Defaults: `epsilon_start = 1.0`,
`epsilon_end = 0.05`, `epsilon_decay = 0.999`. A phase may reset ε to
`epsilon_reset` at its start (only on a fresh, non-resumed phase). During
`select_actions_batch`, agents independently roll ε; if **all** explore, no
network forward runs (guarded early return).

### Learning rate — warmup → cosine (`EpisodeLRScheduler`)

Stepped **once per episode** (not per gradient step), sized by `total_episodes`
across all phases:

```
episodes 0 .. warmup_eps        linear warmup base_lr → max_lr   (LambdaLR)
episodes warmup_eps .. end      cosine max_lr → min_lr           (CosineAnnealingLR)
warmup_eps = max(1, total_episodes · warmup_pct)
```

Defaults: `lr (base) = 1e-4`, `max_lr = 5e-4`, `min_lr = 1e-6`, `warmup_pct =
0.1`. `SequentialLR` is deliberately avoided (it pre-steps sub-schedulers at
construction and skips the first LR); instead the two schedulers are switched
manually, and the LR is set to exactly `max_lr` at the transition so cosine
starts cleanly. The scheduler is created only when `use_episode_scheduler` is
true **and** `total_episodes > 0` (so eval/play, which pass `total_episodes=None`,
have no scheduler).

---

## 7. Evaluation protocol

### In-training `evaluate()`

Runs every `eval_every` episodes inside `run_phase`. Uses a **fixed seed set**
`_EVAL_SEEDS = list(range(100, 120))` (20 maps), so every call tests the agent on
the exact same instances → metrics comparable across epochs. For each eval
episode:

1. Sample DR **deterministically** from the eval seed: `sample_ctx_params(dr_cfg,
   rng=random.Random(eval_seeds[i]))`, then `env.set_domain_params(...)`. So the
   `{map, vision, comm, n_agents}` instance is fixed per seed but varied across
   the 20 seeds.
2. `env.reset(seed=eval_seeds[i])` → fixed map layout.
3. Run greedy-ish (`eval_epsilon = 0.05`) to termination/truncation.

Returns `(mean_reward, success_rate)` where success = fraction of episodes with
`info["found"]`. ε is saved/restored around the eval.

### "New best" rule (`_is_new_best`)

A checkpoint is a new best if its **success rate is higher**, or — when both old
and new success rates are exactly `1.0` — its **mean reward is higher**. So
success dominates; reward is the tie-breaker only at 100% success. New bests are
saved to `best.pt` (weights only, no `training_state`).

### Offline ranking (`eval_checkpoints.py`)

Ranks many checkpoints on identical instances. Two modes:

- **batch** (default): glob `--pattern` in `--ckpt_dir`, evaluate each over
  `--episodes` greedy episodes on seeds `0 .. episodes-1`, write per-checkpoint
  `detail_ep*.csv` and a sorted `ranking_<ts>.csv` (sorted by `sum_reward`).
  `--last_n` evaluates only the most recent N; `--no_dr` disables DR (fixed
  context). `--phase` selects the curriculum phase whose `n_agents` /
  `obstacle_density` / `max_steps` are applied.
- **watch** (`--watch`): poll `--ckpt_dir`, evaluate each new checkpoint as the
  trainer writes it, append a row to `watch_summary.csv` incrementally
  (crash-safe / resumable), and when the trainer writes `--done_file`, do a final
  sweep and print/save the ranking.

The trainer auto-spawns watch mode when `training.eval_during_training: true`
(see §9).

---

## 8. Checkpoint system

`DQNAgent.save(path, training_state=None)` writes a `torch.save` dict:

| Key | Content |
|-----|---------|
| `q_net` | online `CnnQNetwork.state_dict()` |
| `optimizer` | Adam state |
| `warmup_sched` | LambdaLR state (or `None`) |
| `cosine_sched` | CosineAnnealingLR state (or `None`) |
| `sched_episode` | LR-schedule step count |
| `epsilon` | current ε |
| `train_steps` | gradient-step counter (drives target sync) |
| `training_state` | *(optional)* `{global_episode, phase_idx, ep_in_phase, best_success, best_reward}` |

**Files written by `run_phase`:**

| File | When | Has `training_state`? |
|------|------|------------------------|
| `best.pt` | on a new eval best | no (weights only) |
| `{name}_ep{N}.pt` | every `save_every` episodes | no |
| `last.pt` | every `save_every` episodes **and** at phase end | **yes** (resume metadata) |

### Loading modes

- **`load(path)`** — full restore: `q_net`, `optimizer`, LR schedule, ε,
  `train_steps`; copies `q_net` into `target_net`; **returns** the stored
  `training_state` (or `None`). Used by `--resume` from `last.pt`. Legacy
  checkpoints with no `training_state` load fine — the curriculum just restarts.
- **`load_weights_only(path)`** — copies **only** `q_net` into both networks;
  optimizer, schedule, ε, replay buffer, curriculum all start fresh. Used by
  `--init-weights` to warm-start a brand-new run.

The **replay buffer is never serialized** (too large, by design). Loading is
device-aware via `map_location`. `--resume` and `--init-weights` are mutually
exclusive.

```bash
python main.py --mode train                                  # fresh run
python main.py --mode train --resume checkpoints/last.pt     # continue interrupted run
python main.py --mode train --init-weights checkpoints/best.pt  # warm-start fresh run
```

---

## 9. Parallel evaluation during training

With `training.eval_during_training: true`, `_spawn_parallel_evaluator` launches
`eval_checkpoints.py --watch` as a **separate process** before training starts.
It watches the active phase's `{name}_ep*.pt` pattern, evaluates each periodic
checkpoint as it lands (with DR, deterministic per seed), and writes:

- `eval_results/watch_summary.csv` — one row per checkpoint (incremental).
- `eval_results/parallel_eval.log` — the evaluator's stdout/stderr.
- `eval_results/ranking_<ts>.csv` — final ranking when training finishes.

If multiple phases are active, only the **last** one is watched.
`eval_during_training_episodes` (default 200) sets episodes per checkpoint;
`eval_during_training_device` (`cpu`/`cuda`) sets its device (cuda is faster but
shares VRAM with training). On training end/Ctrl-C/exception,
`_finalize_parallel_evaluator` writes the sentinel `.training_done`, waits for the
evaluator, and echoes the final ranking.

---

## 10. TensorBoard scalars

Logged from `run_phase` (`{name}` is the phase name). Open with
`tensorboard --logdir runs/`.

**Per phase, every episode:**

| Tag | Value |
|-----|-------|
| `{name}/episode_reward` | summed reward over the episode |
| `{name}/epsilon` | current ε |
| `{name}/episode_length` | `info["step"]` (total agent-steps) |
| `{name}/found_target` | `1.0` if the target was found |
| `{name}/lr` | current learning rate |
| `{name}/dr_vision_radius`, `{name}/dr_comm_range`, `{name}/dr_n_agents` | sampled DR values |

**Per phase, every `eval_every` episodes:**

| Tag | Value |
|-----|-------|
| `{name}/eval_success_rate` | success rate on the fixed eval seeds |
| `{name}/eval_mean_reward` | mean reward on the fixed eval seeds |

**Global (cross-phase) mirrors:**

| Tag | Value |
|-----|-------|
| `train/episode_reward`, `train/epsilon`, `train/lr` | per episode |
| `eval/success_rate`, `eval/mean_reward` | per eval |

The tqdm progress bar additionally shows rolling 50-episode reward/success
(`r50`, `suc50`), ε, LR, best success rate, and the current DR vision radius.

---

## 11. Gradient step recap (`DQNAgent.update`)

Runs only when `len(buffer) >= batch_size`. One Double-DQN step:

```
sample batch  (position rides inside obs_global; recovered via argmax in forward)
q      = Q_online(s, ctx).gather(action)
a*     = argmax_a Q_online(s', ctx')                    # online selects
next_q = Q_target(s', ctx').gather(a*)                  # target evaluates
target = R_n + bootstrap_disc · next_q                  # γ & done folded in
loss   = SmoothL1(q, target)  → backward → clip_grad_norm_(grad_clip) → step
every target_update_freq steps: target_net ← q_net      # hard sync
```

Defaults: `gamma = 0.97`, `n_step = 3`, `batch_size = 16`, `buffer_size =
200000`, `target_update_freq = 500`, `update_every = 4`, `grad_clip = 1`,
`weight_decay = 1e-6`, optimizer Adam.

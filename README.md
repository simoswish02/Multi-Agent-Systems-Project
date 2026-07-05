# Multi-Drone Exploration with Random Faults — Multi-Agent Systems Project

Final project for the **Multi-Agent Systems** course (University of Bologna,
A.Y. 2025/26). See [`MAS_proposal.pdf`](MAS_proposal.pdf) for the initial
report this project implements.

A team of `N` drones cooperatively searches a 32×32 grid world with random
obstacles for one hidden target, under **partial observability**, **limited
proximity communication**, and a fixed step budget — while **any drone can
randomly break down mid-episode** and stay inactive for the rest of the run.
The remaining drones must adapt to the unexpected loss of teammates.

A single **Dueling Double DQN with parameter sharing** drives every drone; one
policy generalizes across team sizes, vision/communication ranges, obstacle
densities, and fault regimes thanks to per-episode **domain randomization** fed
to the network as a context vector.

## Key features

- **Random drone faults** — a configurable per-step probability
  (`env.fault_prob`, DR-randomized in training) breaks a drone at the start of
  its turn: no more movement, sensing, or communication. Its turns still
  consume the time budget.
- **Decentralized fault knowledge** — no global oracle: a wreck emits a
  distress beacon; drones passing within comm range (or seeing the crash)
  record it in their own `Broken` map, and the knowledge spreads through the
  usual proximity map-fusion. Each drone acts on its own *believed* alive
  count.
- **Fault-aware policy inputs** — the observation gains a `Broken` channel
  (known crash sites) and the context vector gains `agent_id` (breaks the
  parameter-sharing symmetry so drones can split up from step 0) and
  `n_alive_belief`.
- **Deterministic target navigation (eval-only)** — with `--auto-nav`, as soon
  as the target enters a drone's known map (seen or received via
  communication) the drone abandons the network policy and follows the BFS
  shortest path on its own known map, replanned every step. Never used in
  training.
- **Extended domain randomization** — `obstacle_density` and `fault_prob` are
  sampled per episode alongside vision radius, comm range, and team size.
- **Live GUI** — pygame renderer with per-drone minimaps, charts, and episode
  log; wrecks are drawn grayed-out with a red X, the toolbar shows the alive
  count, and auto-nav paths are overlaid in green. The simulate mode's setup
  screen exposes a fault-probability slider and an auto-nav toggle.

## Installation

```bash
# Option A — conda (recommended: includes PyTorch + CUDA 11.8)
conda env create -f environment.yaml
conda activate rl-drone

# Option B — pip
pip install -r requirements.txt
```

## Usage

All entry points read `configs/default.yaml` (override with `--config`).

```bash
# Train (curriculum-staged, faults + density randomized via DR)
python main.py --mode train
python main.py --mode train --resume checkpoints/last.pt

# Evaluate with GUI (requires a checkpoint)
python main.py --mode eval --checkpoint checkpoints/best.pt \
               [--fault-prob 0.003] [--auto-nav] \
               [--vision-radius K --comm-range K --n-agents K]

# Interactive
python main.py --mode play        # arrow keys drive drone 0
python main.py --mode simulate    # GUI setup screen (fault slider, auto-nav toggle)

# Rank checkpoints on identical fixed-seed instances
python eval_checkpoints.py --episodes 200 [--fault_prob P] [--auto_nav]

# Fault-robustness experiment (success rate vs fault probability)
python testing/evaluate_policy.py --axis fault --seeds 500 [--auto-nav]

# TensorBoard
tensorboard --logdir runs/
```

## Project structure

```
├── env/                  # Gymnasium environment
│   ├── grid_env.py       # DroneSearchEnv (fault model, comm fusion, rewards)
│   └── utils.py          # grid generation, BFS path utilities, auto-nav
├── agents/               # DQN agent
│   ├── networks.py       # GlobalMapCNN (ConvNeXt+MHSA) + LocalCNN + dueling head
│   ├── dqn_agent.py      # action selection, replay, Double-DQN update
│   └── nstep.py          # per-agent n-step return windows
├── training/train.py     # curriculum training loop + in-training evaluation
├── testing/              # OFAT sensitivity sweeps (incl. the fault axis)
├── gui/                  # pygame renderer + simulation setup screen
├── configs/default.yaml  # all parameters (env, DR, curriculum, agent, training)
├── main.py               # entry point (train / eval / play / simulate)
└── eval_checkpoints.py   # offline checkpoint ranking (batch / --watch)
```

## Documentation

The `docs/` folder is the deep-dive reference:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — modules, data flow, fault
  model, network, context vector, invariants
- [`docs/ENV.md`](docs/ENV.md) — observations, rewards, faults, communication,
  termination
- [`docs/TRAINING.md`](docs/TRAINING.md) — curriculum, domain randomization,
  n-step returns, checkpoints, TensorBoard
- [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md) — one-page cheat sheet

## Origin

The codebase extends the author's Reinforcement Learning course project
(multi-drone search, `bignet-pos-channel` variant) with the fault model,
decentralized fault knowledge, fault-aware policy inputs, extended domain
randomization, and the deterministic navigation mode described above.

# Multi-Drone Cooperative Search with Random Faults

Final project for the **Multi-Agent Systems** course, MSc in Artificial
Intelligence, University of Bologna, A.Y. 2025/26.
Simone Rimondi — [simone.rimondi5@studio.unibo.it](mailto:simone.rimondi5@studio.unibo.it)

A team of `N` drones cooperatively searches a 32×32 grid world with random
obstacles for one hidden target, under **partial observability**, **limited
proximity communication**, and a fixed step budget — while **any drone can
randomly break down mid-episode** and stay inactive for the rest of the run,
with no central authority to announce the loss. The survivors have to notice and
absorb it on their own.

A single **Dueling Double DQN with parameter sharing** drives every drone; one
policy generalizes across team sizes, vision/communication ranges, obstacle
densities and fault regimes thanks to per-episode **domain randomization** fed
to the network as a context vector.

## Deliverables

| | |
|---|---|
| 📄 **Report** | [`report/main.pdf`](report/main.pdf) — the full write-up (49 pp.), also the deep-dive reference for this code |
| 🎬 **Video walkthrough** | **<https://youtu.be/az1LtAcZYZo>** (~4 min, generated end to end from code) |
| 🧠 **Trained weights** | [`simoswish/MAS_MultiDroneExploration`](https://huggingface.co/simoswish/MAS_MultiDroneExploration) on Hugging Face |
| 📊 **Results** | [`testing_results/`](testing_results/) — every CSV and figure behind Section 7 |
| 🔬 **Analysis** | [`testing/analysis.ipynb`](testing/analysis.ipynb) — executed, the single source of the report's figures |

Repositories: [GitLab](https://dvcs.apice.unibo.it/pika-lab/courses/ai-ethics/projects/rimondi2526-mas)
(course repository) · [GitHub](https://github.com/simoswish02/Multi-Agent-Systems-Project) (mirror).

## Key features

- **Random drone faults** — a configurable per-step probability
  (`env.fault_prob`, DR-randomized in training) breaks a drone at the start of
  its turn: no more movement, sensing, or communication. Its turns still
  consume the time budget.
- **Decentralized fault knowledge** — no global oracle: a wreck emits a
  distress beacon; drones passing within comm range (or seeing the crash)
  record it in their own `Broken` map, and the knowledge spreads through the
  usual proximity map-fusion. Each drone acts on its own *believed* alive
  count, which may be sound but stale.
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
conda env create -f environment.yaml     # or: make install
conda activate rl-drone

# Option B — pip
pip install -r requirements.txt          # or: make install-pip
```

## Trained weights

The policy behind every number in the report is
`mas_50k_dr_faults_ep50000.pt`, the final checkpoint of the 50,000-episode run.
It is ~270 MB, too large to track here, and is published on Hugging Face:

**<https://huggingface.co/simoswish/MAS_MultiDroneExploration>**

```bash
huggingface-cli download simoswish/MAS_MultiDroneExploration \
    mas_50k_dr_faults_ep50000.pt --local-dir checkpoints/
```

With the file in `checkpoints/`, the evaluation commands below run as written.
Training from scratch needs no checkpoint.

## Usage

All entry points read `configs/default.yaml` (override with `--config`).
The `Makefile` wraps the common ones (`make help` lists them).

```bash
# Train (curriculum-staged, faults + density randomized via DR)
python main.py --mode train
python main.py --mode train --resume checkpoints/last.pt

# Evaluate with GUI (requires a checkpoint)
python main.py --mode eval --checkpoint checkpoints/mas_50k_dr_faults_ep50000.pt \
               [--fault-prob 0.003] [--auto-nav] \
               [--vision-radius K --comm-range K --n-agents K]

# Interactive
python main.py --mode play        # arrow keys drive the first drone
python main.py --mode simulate    # GUI setup screen (fault slider, auto-nav toggle)

# Rank checkpoints on identical fixed-seed instances
python eval_checkpoints.py --episodes 200 [--fault_prob P] [--auto_nav]

# The Section-7 experiments (or: make experiments)
python testing/evaluate_policy.py --axis all      --seeds 500
python testing/evaluate_policy.py --axis fault    --seeds 500 --auto-nav
python testing/evaluate_policy.py --axis heatmaps --seeds 300
python testing/evaluate_policy.py --axis epoch    --seeds 300

# Regenerate every figure in the report (or: make figures)
jupyter nbconvert --to notebook --execute --inplace testing/analysis.ipynb
python testing/make_training_figures.py

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
├── testing/              # OFAT sensitivity sweeps + the figure notebook
├── testing_results/      # the CSVs and plots those sweeps produced
├── gui/                  # pygame renderer + simulation setup screen
├── demo/                 # code that renders the video walkthrough
├── hpc/                  # SLURM batch scripts used to produce the results
├── report/               # LaTeX sources, figures, and the compiled main.pdf
├── configs/default.yaml  # all parameters (env, DR, curriculum, agent, training)
├── main.py               # entry point (train / eval / play / simulate)
└── eval_checkpoints.py   # offline checkpoint ranking (batch / --watch)
```

## Documentation

[`report/main.pdf`](report/main.pdf) is the reference for this codebase, not
just an account of the results. Section 3 formalizes the problem, Section 4 the
environment, observation channels and network, Section 5 the training pipeline,
Section 6 reads the system through the multi-agent paradigm, Section 7 reports
the experiments, and the appendices give every hyperparameter and a guide to
the repository.

## Origin

The codebase extends the author's Reinforcement Learning course project
(multi-drone search, `bignet-pos-channel` variant) with the fault model,
decentralized fault knowledge, fault-aware policy inputs, extended domain
randomization, and the deterministic navigation mode described above.

## License

MIT — see [`LICENSE`](LICENSE).

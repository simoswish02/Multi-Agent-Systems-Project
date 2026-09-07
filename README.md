# Multi-Drone Cooperative Search with Random Faults

Final project for the Multi-Agent Systems course, MSc in Artificial
Intelligence, University of Bologna, A.Y. 2025/26.
Simone Rimondi, 0001189110, <simone.rimondi5@studio.unibo.it>

A team of `N` drones searches a 32x32 grid world with random obstacles for one
hidden target under partial observability, range-limited communication and a
fixed step budget. Any drone can break permanently at any turn and stay inactive
for the rest of the episode; there is no central authority to announce the loss,
so the survivors must detect and absorb it themselves.

A single Dueling Double DQN with parameter sharing drives every drone. One
policy covers all team sizes, vision and communication ranges, obstacle
densities and fault rates, via per-episode domain randomization supplied to the
network as a context vector.

## Deliverables

| Artefact | Location |
|---|---|
| Report (49 pp.) | [`report/Rimondi-MultiDroneSearch-MAS.pdf`](report/Rimondi-MultiDroneSearch-MAS.pdf) |
| Video walkthrough (approx. 4 min) | <https://youtu.be/8dnRG6ZneP8> |
| Trained weights | [`simoswish/MAS_MultiDroneExploration`](https://huggingface.co/simoswish/MAS_MultiDroneExploration) (Hugging Face) |
| Experimental results | [`testing_results/`](testing_results/) (CSVs and figures for Section 7) |
| Analysis notebook | [`testing/analysis.ipynb`](testing/analysis.ipynb) (executed; sole source of the report's figures) |

Repositories:
[GitLab](https://dvcs.apice.unibo.it/pika-lab/courses/ai-ethics/projects/rimondi2526-mas)
(course repository),
[GitHub](https://github.com/simoswish02/Multi-Agent-Systems-Project) (mirror).

## Features

**Random drone faults.** A configurable per-turn probability (`env.fault_prob`,
randomized during training) breaks a drone at the start of its turn: it stops
moving, sensing and communicating. Its turns still consume the time budget.

**Decentralized fault knowledge.** There is no global oracle. A wreck emits a
passive distress beacon; drones within communication range, or which see the
crash site directly, record it in their own `Broken` map, and the information
then spreads through ordinary proximity map fusion. Each drone acts on its own
believed alive count, which is sound but may be stale.

**Fault-aware policy inputs.** The observation carries a `Broken` channel of
known crash sites. The context vector carries `agent_id`, which breaks the
parameter-sharing symmetry so that drones can split up from the first step, and
`n_alive_belief`.

**Deterministic target navigation (evaluation only).** Under `--auto-nav`, once
the target enters a drone's known map, whether seen directly or received over
communication, that drone stops querying the network and follows the BFS
shortest path computed on its own known obstacles, replanned every step. This is
never used during training.

**Extended domain randomization.** `obstacle_density` and `fault_prob` are
sampled per episode alongside vision radius, communication range and team size.

**Live GUI.** A pygame renderer with per-drone minimaps, live charts and an
episode log. Wrecks are drawn greyed out with a red cross, the toolbar reports
the alive count, and auto-nav paths are overlaid in green. The setup screen of
simulate mode exposes a fault-probability slider and an auto-nav toggle.

## Installation

Conda, which pins PyTorch and CUDA 11.8:

```bash
conda env create -f environment.yaml     # or: make install
conda activate rl-drone
```

Or pip:

```bash
pip install -r requirements.txt          # or: make install-pip
```

## Trained weights

Every number in the report comes from `mas_50k_dr_faults_ep50000.pt`, the final
checkpoint of the 50,000-episode run. At roughly 270 MB it is not tracked in
this repository; it is published on Hugging Face together with the 20 periodic
checkpoints behind the learning curve:

```bash
hf download simoswish/MAS_MultiDroneExploration \
    mas_50k_dr_faults_ep50000.pt --local-dir checkpoints/
```

With that file in `checkpoints/`, the evaluation commands below run as written.
Training from scratch requires no checkpoint.

## Usage

All entry points read `configs/default.yaml`; pass `--config` to override it.
The `Makefile` wraps the common invocations (`make help` lists them).

```bash
# Training (curriculum-staged; faults and density randomized per episode)
python main.py --mode train
python main.py --mode train --resume checkpoints/last.pt

# Evaluation with the GUI (requires a checkpoint)
python main.py --mode eval --checkpoint checkpoints/mas_50k_dr_faults_ep50000.pt \
               [--fault-prob 0.003] [--auto-nav] \
               [--vision-radius K --comm-range K --n-agents K]

# Interactive modes
python main.py --mode play        # arrow keys drive the first drone
python main.py --mode simulate    # setup screen: fault slider, auto-nav toggle

# Offline checkpoint ranking on identical fixed-seed instances
python eval_checkpoints.py --episodes 200 [--fault_prob P] [--auto_nav]

# The Section 7 experiments (or: make experiments)
python testing/evaluate_policy.py --axis all      --seeds 500
python testing/evaluate_policy.py --axis fault    --seeds 500 --auto-nav
python testing/evaluate_policy.py --axis heatmaps --seeds 300
python testing/evaluate_policy.py --axis epoch    --seeds 300

# Every figure in the report (or: make figures)
jupyter nbconvert --to notebook --execute --inplace testing/analysis.ipynb
python testing/make_training_figures.py

# Training telemetry
tensorboard --logdir runs/
```

The SLURM batch scripts in [`hpc/`](hpc/) are the exact jobs used to produce
`testing_results/`; their `--chdir` and `TMPDIR` settings are cluster-specific.

## Repository layout

```
env/                  Gymnasium environment
  grid_env.py         DroneSearchEnv: fault model, comm fusion, rewards
  utils.py            grid generation, BFS path utilities, auto-nav
agents/               DQN agent
  networks.py         GlobalMapCNN (ConvNeXt + MHSA), LocalCNN, dueling head
  dqn_agent.py        action selection, replay, Double-DQN update
  nstep.py            per-agent n-step return windows
training/train.py     curriculum training loop and in-training evaluation
testing/              OFAT sensitivity sweeps and the figure notebook
testing_results/      CSVs and plots produced by those sweeps
gui/                  pygame renderer and simulation setup screen
demo/                 code that renders the video walkthrough
hpc/                  SLURM batch scripts used to produce the results
report/               LaTeX sources, figures and the compiled PDF
configs/default.yaml  all parameters (env, DR, curriculum, agent, training)
main.py               entry point: train / eval / play / simulate
eval_checkpoints.py   offline checkpoint ranking (batch or --watch)
```

## Documentation

[`report/Rimondi-MultiDroneSearch-MAS.pdf`](report/Rimondi-MultiDroneSearch-MAS.pdf)
documents this codebase as well as the
results. Section 3 formalizes the problem, Section 4 covers the environment,
the observation channels and the network, Section 5 the training pipeline,
Section 6 reads the system through the multi-agent paradigm and Section 7
reports the experiments. The appendices list every hyperparameter and give a
guide to the repository.

## Origin

The codebase extends the author's Reinforcement Learning course project
(multi-drone search, `bignet-pos-channel` variant) with the fault model,
decentralized fault knowledge, fault-aware policy inputs, extended domain
randomization and the deterministic navigation mode described above.

## License

MIT. See [`LICENSE`](LICENSE).

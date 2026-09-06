# `hpc/` — SLURM batch scripts

The scripts that produced the results in `testing_results/`, as submitted on the
university's SLURM cluster. They are kept for the record: the paths inside them
(`--chdir`, `TMPDIR`, `PY=venv/bin/python3`) are specific to that machine and to
one user's scratch space, so on any other system either edit those three lines
or run the underlying commands directly — every script is a thin wrapper around
one `testing/evaluate_policy.py` invocation, and `make experiments` runs the
same four sweeps locally.

| Script | What it runs | Report section |
|---|---|---|
| `script_ofat.sbatch` | OFAT sweep over all six axes, 500 seeds | §7.3–§7.4 |
| `script_fault_autonav.sbatch` | Fault axis with the BFS auto-nav override, 500 seeds | §7.2, §7.4 |
| `script_heatmaps.sbatch` | The two 2-D interaction grids, 300 seeds/cell | §7.6 |
| `script_epoch.sbatch` | Learning curve over the periodic checkpoints, 300 seeds | §5, §7 |
| `script.sbatch` | All four of the above in one job | — |

Every script evaluates `checkpoints/mas_50k_dr_faults_ep50000.pt`, the final
checkpoint of the 50,000-episode run; see the README for where to download it.

Training itself was run as a chain of jobs of the same shape, each resuming from
`checkpoints/last.pt`, because the cluster's wall-clock limit is shorter than a
full run. `python main.py --mode train --resume checkpoints/last.pt` is the only
command that differs.

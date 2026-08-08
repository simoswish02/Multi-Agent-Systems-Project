#!/bin/bash
#SBATCH --job-name=rl-drone-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simone.rimondi5@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rtx2080
#SBATCH --output=logs/output_%j.log
#SBATCH --chdir=/scratch.hpc/simone.rimondi5/Multi-Agent-Systems-Project
#SBATCH --gres=gpu:1

# Keep temp files off the tiny node-local /tmp (~2G free); use scratch.
export TMPDIR=/scratch.hpc/simone.rimondi5/tmp
mkdir -p "$TMPDIR"

PY=venv/bin/python3
CKPT=checkpoints/mas_50k_dr_faults_ep50000.pt
CKPT_GLOB="checkpoints/mas_50k_dr_faults_ep*.pt"

# 1) OFAT sweep (n_agents, vision, comm, density, max_steps, fault) - report §7.3/§7.4
$PY testing/evaluate_policy.py --axis all --seeds 500 --checkpoint "$CKPT" --out-dir testing_results

# 2) Fault axis with BFS auto-nav - policy-vs-hybrid comparison, report §7.2/§7.4
$PY testing/evaluate_policy.py --axis fault --seeds 500 --auto-nav --checkpoint "$CKPT" --out-dir testing_results/auto_nav

# 3) 2-D interaction heatmaps - report §7.6
$PY testing/evaluate_policy.py --axis heatmaps --seeds 300 --checkpoint "$CKPT" --out-dir testing_results

# 4) Learning curve across periodic checkpoints - report §5/§7
$PY testing/evaluate_policy.py --axis epoch --seeds 300 --ckpt-glob "$CKPT_GLOB" --out-dir testing_results
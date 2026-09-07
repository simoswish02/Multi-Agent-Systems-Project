#!/bin/bash
#SBATCH --job-name=rl-drone-test-fault-autonav
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simone.rimondi5@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rtx2080
#SBATCH --output=logs/output_fault_autonav_%j.log
#SBATCH --chdir=/scratch.hpc/simone.rimondi5/Multi-Agent-Systems-Project
#SBATCH --gres=gpu:1

# Keep temp files off the tiny node-local /tmp (~2G free); use scratch.
# Stream stdout as it is produced: SLURM redirects it to a file, and
# Python block-buffers a redirected stdout, so progress lines otherwise
# only appear when the buffer fills or the job ends.
export PYTHONUNBUFFERED=1

export TMPDIR=/scratch.hpc/simone.rimondi5/tmp
mkdir -p "$TMPDIR"

PY=venv/bin/python3
CKPT=checkpoints/mas_50k_dr_faults_ep50000.pt

# Fault axis with BFS auto-nav - policy-vs-hybrid comparison, report §7.2/§7.4
$PY testing/evaluate_policy.py --axis fault --seeds 500 --auto-nav --checkpoint "$CKPT" --out-dir testing_results/auto_nav

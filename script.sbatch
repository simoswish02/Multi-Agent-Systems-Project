#!/bin/bash
#SBATCH --job-name=rl-drone-train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simone.rimondi5@studio.unibo.it
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rtx2080
#SBATCH --output=logs/output_%j.log
#SBATCH --chdir=/scratch.hpc/simone.rimondi5/Reinforcment-Learning-Project-2526
#SBATCH --gres=gpu:1

venv/bin/python3 testing/evaluate_policy.py --axis heatmaps --seeds 500 --checkpoint checkpoints/Phase3/phase2_ConvNeXT_10k_dr_ep10000.pt
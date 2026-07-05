#!/bin/bash
#SBATCH --job-name=rl-drone-train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simone.rimondi5@studio.unibo.it
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rtx2080
#SBATCH --output=logs/output_%j.log
#SBATCH --chdir=/scratch.hpc/simone.rimondi5/Multi-Agent-Systems-Project
#SBATCH --gres=gpu:1

venv/bin/python3 main.py
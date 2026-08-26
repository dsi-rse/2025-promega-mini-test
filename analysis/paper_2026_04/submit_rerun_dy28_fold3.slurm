#!/bin/bash
#SBATCH --job-name=dy28-fold3-rerun
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/dy28_fold3_rerun_%j.out
#SBATCH --error=logs/dy28_fold3_rerun_%j.err

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.rerun_dy28_fold3

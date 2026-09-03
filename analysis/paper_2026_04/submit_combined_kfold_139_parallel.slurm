#!/bin/bash
#SBATCH --job-name=comb-139
#SBATCH --partition=general
#SBATCH --array=0-10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/combined_kfold_139_%A_%a.out
#SBATCH --error=logs/combined_kfold_139_%A_%a.err

# 11 parallel jobs (one per day), 10×4-fold combined CV, 139 organoids, 3 met variants.
# Merge with: python3 -m analysis.paper_2026_04.merge_combined_results_139

DAYS=(Dy03 Dy06 Dy08 Dy10 Dy13 Dy15 Dy17 Dy20_5 Dy24 Dy28 Dy30)
DAY=${DAYS[$SLURM_ARRAY_TASK_ID]}

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

echo "Array task ${SLURM_ARRAY_TASK_ID}: day=${DAY}"

conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.combined_kfold \
    --n-folds 4 \
    --n-repeats 10 \
    --include-stitched \
    --days ${DAY}

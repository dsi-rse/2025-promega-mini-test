#!/bin/bash
#SBATCH --job-name=combined-kfold-139
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/combined_kfold_139_%j.out
#SBATCH --error=logs/combined_kfold_139_%j.err

# 4-fold combined (met + morph + image) CV, 139 organoids (stitched included).
# Output: analysis_output/images/combined_results_kfold_series_idor_139.json
#         figures/combined_kfold_balanced_accuracy_series_idor_139.png

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.combined_kfold \
    --n-folds 4 \
    --include-stitched

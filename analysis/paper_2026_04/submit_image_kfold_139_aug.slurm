#!/bin/bash
#SBATCH --job-name=img-kfold-139-aug
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/img_kfold_139_aug_%j.out
#SBATCH --error=logs/img_kfold_139_aug_%j.err

# 4-fold image kfold CV with augmentation, 139 organoids (stitched included).
# Output: analysis_output/images/perday_results_kfold_series_idor_139.json
#         figures/perday_image_kfold_balanced_accuracy_series_idor_139.png

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.perday_image_kfold \
    --n-folds 4 \
    --include-stitched

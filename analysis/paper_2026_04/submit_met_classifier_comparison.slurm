#!/bin/bash
#SBATCH --job-name=met-clf-cmp
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/met_classifier_comparison_%j.out
#SBATCH --error=logs/met_classifier_comparison_%j.err

# Met-only 10×4-fold repeated CV: lgbm / logreg / svm / mlp × nan / raw / no_malate
# No GPU required.  16 CPUs feed n_jobs=-1 in GridSearchCV.
# After completion, run:
#   python3 -m analysis.paper_2026_04.plot_met_classifier_comparison

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.met_classifier_comparison \
    --n-repeats 10 \
    --n-folds 4 \
    --include-stitched

#!/bin/bash
#SBATCH --job-name=gradcam-demo
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/gradcam_demo_%j.out
#SBATCH --error=logs/gradcam_demo_%j.err

# Generates GradCAM panels for two days (Dy20.5 fold2, Dy30 fold2).
# Each run trains one fold then renders GradCAM + augmented views.
#
# Output:
#   figures/gradcam_Dy20_5_fold2.png
#   figures/gradcam_Dy30_fold2.png

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

echo "=== Dy20.5 fold 2 ==="
conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.gradcam_demo --day Dy20_5 --fold 2 --n-per-group 2

echo "=== Dy30 fold 2 ==="
conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.gradcam_demo --day Dy30 --fold 2 --n-per-group 2

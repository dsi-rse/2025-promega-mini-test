#!/bin/bash
#SBATCH --job-name=perday-image
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/perday_image_%j.out
#SBATCH --error=logs/perday_image_%j.err

# Per-day EfficientNet-B0 image classifier (fixed train/val/test split).
#
# Augmentation: RandomHorizontalFlip + RandomAffine(rot±180°, translate±10%)
#               + ColorJitter(brightness/contrast 0.3, saturation 0.2, hue 0.05)
#               Translation disabled for Dy28/Dy30 (boundary organoids).
#
# Configurable via env vars:
#   FILTER_MODE    base | series_idor   (default: series_idor)
#   INPUT_MODE     cm_source_image | overlay | img  (default: cm_source_image)
#   DAYS           space-separated days e.g. "Dy30 Dy28" (default: all 11)
#
# Usage:
#   sbatch submit_perday_image.slurm
#   FILTER_MODE=base sbatch submit_perday_image.slurm
#   DAYS="Dy30" sbatch submit_perday_image.slurm   # smoke test

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-/net/projects/CLS/lding/gitcode/2025-promega-mini-test}"
FILTER_MODE="${FILTER_MODE:-series_idor}"
INPUT_MODE="${INPUT_MODE:-cm_source_image}"
DAYS="${DAYS:-}"

mkdir -p "${PROJ_ROOT}/logs"

source /opt/conda/etc/profile.d/conda.sh
conda activate core_env

echo "================================================"
echo "Per-Day EfficientNet Image Classifier"
echo "================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Filter mode: ${FILTER_MODE}"
echo "Input mode:  ${INPUT_MODE}"
echo "Days:        ${DAYS:-all}"
echo "Start:       $(date)"
echo ""
nvidia-smi || true
echo ""

DAYS_ARG=""
if [ -n "${DAYS}" ]; then
    DAYS_ARG="--days ${DAYS}"
fi

cd "${PROJ_ROOT}"
python3 -u -m analysis.paper_2026_04.perday_image_study \
    --filter-mode "${FILTER_MODE}" \
    --input-mode  "${INPUT_MODE}" \
    ${DAYS_ARG}

EXIT_CODE=$?
echo ""
echo "================================================"
[ $EXIT_CODE -eq 0 ] && echo "DONE — $(date)" || echo "FAILED (exit $EXIT_CODE) — $(date)"
echo "================================================"
exit $EXIT_CODE

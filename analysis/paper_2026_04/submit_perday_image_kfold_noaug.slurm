#!/bin/bash
#SBATCH --job-name=perday-kfold-noaug
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/perday_image_kfold_noaug_%j.out
#SBATCH --error=logs/perday_image_kfold_noaug_%j.err

# Per-day EfficientNet-B0 — NO augmentation, 5-fold CV (series_idor).
# Identical to the augmentation run except training transforms are
# resize + normalize only (no flip / affine / colour jitter).
#
# Usage:
#   sbatch submit_perday_image_kfold_noaug.slurm
#   DAYS="Dy30" sbatch submit_perday_image_kfold_noaug.slurm   # smoke test

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-/net/projects/CLS/lding/gitcode/2025-promega-mini-test}"
FILTER_MODE="${FILTER_MODE:-series_idor}"
INPUT_MODE="${INPUT_MODE:-cm_image}"
DAYS="${DAYS:-}"
N_FOLDS="${N_FOLDS:-5}"

mkdir -p "${PROJ_ROOT}/logs"

source /opt/conda/etc/profile.d/conda.sh
conda activate core_env

echo "================================================"
echo "Per-Day EfficientNet 5-Fold CV — NO AUGMENTATION"
echo "================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Filter mode: ${FILTER_MODE}"
echo "Input mode:  ${INPUT_MODE}"
echo "N folds:     ${N_FOLDS}"
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
python3 -u -m analysis.paper_2026_04.perday_image_kfold \
    --filter-mode "${FILTER_MODE}" \
    --input-mode  "${INPUT_MODE}" \
    --n-folds     "${N_FOLDS}" \
    --no-augmentation \
    ${DAYS_ARG}

EXIT_CODE=$?
echo ""
echo "================================================"
[ $EXIT_CODE -eq 0 ] && echo "DONE — $(date)" || echo "FAILED (exit $EXIT_CODE) — $(date)"
echo "================================================"
exit $EXIT_CODE

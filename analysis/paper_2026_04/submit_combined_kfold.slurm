#!/bin/bash
#SBATCH --job-name=combined-kfold
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/combined_kfold_%j.out
#SBATCH --error=logs/combined_kfold_%j.err

# Combined model: metabolite + morphology + image, 5-fold CV.
# Trains LightGBM (met+morph) and EfficientNet (image) per fold per day.
# Runtime estimate: ~4h for all 11 days on a single GPU.
#
# Configurable via env vars:
#   DAYS      space-separated days e.g. "Dy30" (default: all 11)
#   N_FOLDS   integer (default: 5)
#
# Usage:
#   sbatch submit_combined_kfold.slurm
#   DAYS="Dy30" sbatch submit_combined_kfold.slurm   # smoke test

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-/net/projects/CLS/lding/gitcode/2025-promega-mini-test}"
DAYS="${DAYS:-}"
N_FOLDS="${N_FOLDS:-5}"

mkdir -p "${PROJ_ROOT}/logs"

source /opt/conda/etc/profile.d/conda.sh
conda activate core_env

echo "================================================"
echo "Combined Model: Metabolite + Morphology + Image"
echo "================================================"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "N folds:  ${N_FOLDS}"
echo "Days:     ${DAYS:-all}"
echo "Start:    $(date)"
echo ""
nvidia-smi || true
echo ""

DAYS_ARG=""
if [ -n "${DAYS}" ]; then
    DAYS_ARG="--days ${DAYS}"
fi

cd "${PROJ_ROOT}"
python3 -u -m analysis.paper_2026_04.combined_kfold \
    --n-folds "${N_FOLDS}" \
    ${DAYS_ARG}

EXIT_CODE=$?
echo ""
echo "================================================"
[ $EXIT_CODE -eq 0 ] && echo "DONE — $(date)" || echo "FAILED (exit $EXIT_CODE) — $(date)"
echo "================================================"
exit $EXIT_CODE

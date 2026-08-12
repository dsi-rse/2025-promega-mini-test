#!/bin/bash
# Run from project root with conda env active:
# conda activate /net/projects2/promega
#
# Usage:
#   bash run_all_lstm.sh                        # writes to lstm_runs/ (legacy)
#   bash run_all_lstm.sh <cohort_label>         # writes to lstm_runs/<label>/
#                                                 and emits a montage PNG
#
# The cohort_label tags the run so different cohorts (e.g. idor,
# idor_minvotes3, expanded, expanded_minvotes3) don't overwrite each other
# and can be compared via make_run_montage.py --compare ... later.

set -e

LABEL="${1:-}"
RUNS_ROOT=/net/projects2/promega/project_data/model_tests/lstm_runs
PLOTS_DIR=/net/projects2/promega/project_data/amanda_test/model_plots

if [ -n "$LABEL" ]; then
    OUT="$RUNS_ROOT/$LABEL"
else
    OUT="$RUNS_ROOT"
fi
mkdir -p "$OUT"

echo "[run_all_lstm] cohort label: ${LABEL:-<none>}"
echo "[run_all_lstm] output dir:   $OUT"

echo "========================================"
echo "1/3  BASE MODEL"
echo "========================================"
python analysis/images/cnn_lstm/train_base_model.py \
    --output-dir "$OUT/base_effnet" \
    --image-type clipped

echo "========================================"
echo "2/3  TEMPORAL ABLATION (ATTENTION)"
echo "========================================"
python analysis/images/cnn_lstm/train_temporal_ablation_attn.py \
    --output-dir "$OUT/temporal_ablation_attn" \
    --image-type clipped

echo "========================================"
echo "3/3  TEMPORAL ABLATION (LSTM)"
echo "========================================"
python analysis/images/cnn_lstm/train_temporal_ablation_lstm.py \
    --output-dir "$OUT/temporal_ablation_lstm" \
    --image-type clipped

if [ -n "$LABEL" ]; then
    echo "========================================"
    echo "MONTAGE"
    echo "========================================"
    python analysis/images/cnn_lstm/make_run_montage.py \
        --run-dir    "$RUNS_ROOT" \
        --label      "$LABEL" \
        --output-dir "$PLOTS_DIR"
fi

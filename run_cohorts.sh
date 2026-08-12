#!/bin/bash
# run_cohorts.sh — train the full LSTM suite across multiple cohorts back-to-back.
#
# For each cohort label passed in:
#   1. Copy data/cohorts/<label>/series/{train,val,test}.json into data_splits/
#      with the filenames the trainers expect.
#   2. Run run_all_lstm.sh <label> (which writes to lstm_runs/<label>/ and
#      builds montage_<label>.png).
# After all cohorts finish, builds a cross-cohort comparison PNG and an
# updated cohort_summary.{txt,csv,png}.
#
# Usage:
#   conda activate /net/projects2/promega
#   bash run_cohorts.sh idor idor_minvotes3 expanded expanded_minvotes3
#
# Requirements:
#   - The named cohorts already exist under data/cohorts/<label>/series/.
#     If a cohort doesn't exist, the script aborts before touching anything.

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cohort_label> [<cohort_label> ...]"
    exit 1
fi

LABELS=("$@")

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
COHORTS_DIR="$PROJECT_ROOT/data/cohorts"
SPLITS_DIR="$PROJECT_ROOT/data_splits"
RUNS_ROOT=/net/projects2/promega/project_data/model_tests/lstm_runs
PLOTS_DIR=/net/projects2/promega/project_data/amanda_test/model_plots

mkdir -p "$SPLITS_DIR" "$PLOTS_DIR"

# --- Preflight: confirm every cohort exists before kicking off any training ---
echo "[preflight] checking cohorts..."
for label in "${LABELS[@]}"; do
    src="$COHORTS_DIR/$label/series"
    if [ ! -f "$src/train.json" ] || [ ! -f "$src/val.json" ] || [ ! -f "$src/test.json" ]; then
        echo "[error] cohort '$label' missing series files in $src"
        echo "        run make_splits.py to generate it first."
        exit 1
    fi
done
echo "[preflight] OK — all ${#LABELS[@]} cohorts present"

# --- Train each cohort sequentially ---
for label in "${LABELS[@]}"; do
    echo
    echo "############################################################"
    echo "##  cohort: $label"
    echo "############################################################"

    src="$COHORTS_DIR/$label/series"
    cp "$src/train.json" "$SPLITS_DIR/train_idor_series.json"
    cp "$src/val.json"   "$SPLITS_DIR/val_idor_series.json"
    cp "$src/test.json"  "$SPLITS_DIR/test_idor_series.json"
    echo "[copy] $src/{train,val,test}.json -> $SPLITS_DIR/{train,val,test}_idor_series.json"

    bash "$PROJECT_ROOT/run_all_lstm.sh" "$label"
done

# --- Cross-cohort comparison + summary table ---
echo
echo "############################################################"
echo "##  cross-cohort comparison + cohort summary"
echo "############################################################"

python "$PROJECT_ROOT/analysis/images/cnn_lstm/make_run_montage.py" \
    --run-dir    "$RUNS_ROOT" \
    --compare    "${LABELS[@]}" \
    --output-dir "$PLOTS_DIR"

python "$PROJECT_ROOT/scripts/splits/cohort_summary.py" \
    --cohorts    "${LABELS[@]/#/$COHORTS_DIR/}" \
    --view       series \
    --output-dir "$PLOTS_DIR"

echo
echo "[done] outputs in: $PLOTS_DIR"

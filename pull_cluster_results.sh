#!/bin/bash
# Pull cnn_lstm results + figures from the Promega cluster to this local repo.
#
# Requirements:
#   - Your Mac can `ssh` to the cluster login node (VPN on if needed).
#   - rsync is installed (it is, by default, on macOS).
#
# Usage:
#   ./pull_cluster_results.sh user@login-host                 # all cohorts, no checkpoints
#   ./pull_cluster_results.sh user@login-host idor            # one cohort only
#   ./pull_cluster_results.sh user@login-host idor --with-ckpts   # also pull .pt weights (big)
#
# Example host might look like: amandaj@fe01.rcc.uchicago.edu  (fill in your real one)

set -euo pipefail

# ---- fill in / pass as args ---------------------------------------------
SSH_TARGET="${1:?Pass your ssh target, e.g. user@login-host}"
COHORT="${2:-ALL}"          # ALL, or one of: idor idor_minvotes3 expanded expanded_minvotes3
WITH_CKPTS="${3:-}"         # pass --with-ckpts to also copy model weights

# ---- cluster paths (from the code defaults) -----------------------------
REMOTE_RUNS="/net/projects2/promega/project_data/model_tests/lstm_runs"
REMOTE_PLOTS="/net/projects2/promega/project_data/amanda_test/model_plots"

# ---- local destination (in the PARENT folder, next to the repo) ---------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$HERE")"                 # .../Promega
DEST="$PARENT/promega_cluster_results"      # .../Promega/promega_cluster_results
mkdir -p "$DEST/lstm_runs" "$DEST/model_plots"

# ---- what to include ----------------------------------------------------
# Always: results/metrics JSONs, CSVs, PNGs. Checkpoints only if asked.
INCLUDES=(--include="*/" --include="*.json" --include="*.csv" --include="*.png" --include="*.txt")
if [ "$WITH_CKPTS" = "--with-ckpts" ]; then
    INCLUDES+=(--include="*.pt" --include="*.pth")
fi
INCLUDES+=(--exclude="*")

RSYNC_OPTS=(-avh --prune-empty-dirs --progress)

echo "==> Pulling from $SSH_TARGET"
echo "    runs  : $REMOTE_RUNS"
echo "    plots : $REMOTE_PLOTS"
echo "    cohort: $COHORT   ckpts: ${WITH_CKPTS:-no}"
echo

# ---- runs (result JSONs live here) --------------------------------------
if [ "$COHORT" = "ALL" ]; then
    rsync "${RSYNC_OPTS[@]}" "${INCLUDES[@]}" \
        "$SSH_TARGET:$REMOTE_RUNS/" "$DEST/lstm_runs/"
else
    mkdir -p "$DEST/lstm_runs/$COHORT"
    rsync "${RSYNC_OPTS[@]}" "${INCLUDES[@]}" \
        "$SSH_TARGET:$REMOTE_RUNS/$COHORT/" "$DEST/lstm_runs/$COHORT/"
fi

# ---- generated figures / CSVs -------------------------------------------
rsync "${RSYNC_OPTS[@]}" "${INCLUDES[@]}" \
    "$SSH_TARGET:$REMOTE_PLOTS/" "$DEST/model_plots/"

echo
echo "==> Done. Pulled into: $DEST"
echo "    Result JSONs:  cluster_pull/lstm_runs/<cohort>/{base_effnet,temporal_ablation_attn,temporal_ablation_lstm}/*.json"
echo "    Figures/CSVs:  cluster_pull/model_plots/"
find "$DEST" -maxdepth 4 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.png" \) | sort

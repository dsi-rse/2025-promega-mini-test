#!/usr/bin/env python3
"""
make_temporal_attention.py — visualize temporal attention weights from the
OrganoidCNN_TAtt model.

The model in train_temporal_ablation_attn.py already returns per-timepoint
attention weights (a softmax-normalized vector of length T, summing to 1) as
its second output. This script loads a trained checkpoint, runs it over a
selected set of test-organoid sequences, and renders the attention weights
as a heatmap so you can see WHICH DAY each organoid's prediction relied on.

Per organoid, the attention vector tells you:
    - day with weight ~1.0  -> model's whole decision came from that one day
    - flat distribution     -> model averaged across timepoints (no clear focus)
    - early-day peak        -> model thinks early morphology was decisive
    - late-day peak         -> model relied on day-30 quality (most common)

Usage
-----
    python analysis/images/cnn_lstm/make_temporal_attention.py \\
        --label idor_minvotes3 \\
        --max-day 30 \\
        --top-n 10 \\
        --selection-mode missed

Outputs to --plots-dir:
    tattn_<label>_max<N>_<mode>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use the trainer's model definition (the canonical one for this checkpoint).
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_temporal_ablation_attn import OrganoidCNN_TAtt  # noqa: E402
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, load_split_from_json,
)


# ============================================================
# Helpers — select organoids based on miss patterns
# ============================================================

def select_organoids(label: str, max_day: float, run_dir: Path,
                     cohorts_dir: Path, mode: str, top_n: int,
                     filter_label: str = "none") -> list[str]:
    """
    Select organoids from the cohort test split, ranked by how often they were
    misclassified in the attn results for this max_day. Modes:
        missed  -> most-missed first
        lowest  -> least-missed first (model gets these right consistently)
        perfect -> miss_count == 0
    """
    # Load the temporal_attn results to get FP/FN lists per variant
    rpath = run_dir / label / "temporal_ablation_attn" / "temporal_ablation_results.json"
    if not rpath.exists():
        raise FileNotFoundError(f"missing results JSON: {rpath}")
    with open(rpath) as f:
        results = json.load(f)

    # Aggregate misses across ALL variants (not just max_day), as in analyze_misses
    misses = defaultdict(int)
    for variant in results:
        for oid in (variant.get("test_false_positives") or []):
            misses[oid] += 1
        for oid in (variant.get("test_false_negatives") or []):
            misses[oid] += 1

    # Load test split for ground-truth labels + organoid IDs
    test_p = cohorts_dir / label / "series" / "test.json"
    with open(test_p) as f:
        test_d = json.load(f)

    # Optional filter by true label
    candidates = list(test_d.keys())
    if filter_label != "none":
        candidates = [o for o in candidates if test_d[o].get("label") == filter_label]

    if mode == "missed":
        ranked = sorted(candidates, key=lambda o: (-misses.get(o, 0), o))
        ranked = [o for o in ranked if misses.get(o, 0) > 0][:top_n]
    elif mode == "lowest":
        ranked = sorted(candidates, key=lambda o: (misses.get(o, 0), o))[:top_n]
    elif mode == "perfect":
        ranked = [o for o in candidates if misses.get(o, 0) == 0][:top_n]
    else:
        raise ValueError(f"unknown mode: {mode}")

    return ranked, test_d, misses


# ============================================================
# Inference — run model + collect attention vectors
# ============================================================

def run_inference(ckpt_path: Path, oids: list[str], series_meta: dict,
                  max_day: float, device: str = "cuda") -> dict:
    """
    For each organoid, run the trained model and collect:
        - attention weights vector (one per included day)
        - days included
        - logit + predicted label
    """
    dataset = OrganoidTimeSeriesDataset(
        oids, series_meta, max_day=max_day, image_type="clipped",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OrganoidCNN_TAtt().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    out: dict[str, dict] = {}
    with torch.no_grad():
        for batch_idx, (seqs, days_norm, label, weight, oid) in enumerate(loader):
            seqs = seqs.to(device)
            days_norm = days_norm.to(device).float()
            logit, attn = model(seqs, days_norm)
            prob = torch.sigmoid(logit).item()
            pred = int(prob >= 0.5)

            # Recover the actual day values for this sequence
            actual_days = [
                tp["mdl_day"] for tp in series_meta[oid[0]]["timepoints"]
                if tp["mdl_day"] <= max_day
            ]
            out[oid[0]] = {
                "attn":   attn.squeeze(0).cpu().numpy(),
                "days":   actual_days,
                "logit":  float(logit.item()),
                "prob":   float(prob),
                "pred":   pred,
                "true":   int(label.item()),
            }
    return out


# ============================================================
# Render
# ============================================================

def render_heatmap(att_data: dict, test_d: dict, misses: dict,
                   label: str, max_day: float, mode: str,
                   out_path: Path) -> None:
    """Render rows = organoids, columns = days, color = attention weight."""
    oids = list(att_data.keys())
    if not oids:
        print("[warn] nothing to render"); return

    all_days = sorted({d for r in att_data.values() for d in r["days"]})
    matrix = np.zeros((len(oids), len(all_days)))
    for i, oid in enumerate(oids):
        for d, w in zip(att_data[oid]["days"], att_data[oid]["attn"]):
            j = all_days.index(d)
            matrix[i, j] = w

    fig_h = max(3.0, 0.4 * len(oids) + 1.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=matrix.max())

    ax.set_xticks(range(len(all_days)))
    ax.set_xticklabels([f"Dy {d:g}" for d in all_days], fontsize=10)
    ax.set_yticks(range(len(oids)))

    # Y-axis labels: id + true label + votes + pred + miss_count
    yticklabels = []
    for oid in oids:
        entry = test_d.get(oid, {})
        tl = "Acc" if entry.get("label") == "Acceptable" else "Bad"
        g, t = entry.get("n_votes_good"), entry.get("n_votes_total")
        votes = f"{g}/{t}" if t else "?"
        pred = att_data[oid]["pred"]
        pred_lab = "Acc" if pred == 1 else "Bad"
        ok = "✓" if pred == att_data[oid]["true"] else "✗"
        miss_n = misses.get(oid, 0)
        yticklabels.append(f"{oid}  [{tl} {votes}]  pred={pred_lab} {ok}  miss={miss_n}")
    ax.set_yticklabels(yticklabels, fontsize=8)

    # Annotate each cell with the weight
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v > 0:
                color = "white" if v > matrix.max() * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("attention weight (softmax over days)", fontsize=10)

    title = (f"Temporal attention weights — {label}, max_day = {max_day:g}, "
             f"selection = {mode}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Day in sequence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# ============================================================
# Entrypoint
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="idor_minvotes3",
                   help="Cohort label (idor / idor_minvotes3 / expanded / expanded_minvotes3).")
    p.add_argument("--max-day", type=float, default=30,
                   help="Which trained variant to visualize. Default 30 (full sequence).")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--selection-mode", default="missed",
                   choices=["missed", "lowest", "perfect"])
    p.add_argument("--filter-label", default="none",
                   choices=["none", "Acceptable", "Not Acceptable"])
    p.add_argument("--run-dir", type=Path,
                   default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    p.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    p.add_argument("--plots-dir", type=Path,
                   default=Path("/net/projects2/promega/project_data/amanda_test/model_plots"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Checkpoint path matches the trainer's filename convention
    ckpt_dir = args.run_dir / args.label / "temporal_ablation_attn" / f"days_3-{args.max_day:g}"
    ckpt_path = ckpt_dir / f"model_days_3-{args.max_day:g}.pth"
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        return 1

    print(f"[load]  checkpoint {ckpt_path}")
    oids, test_d, misses = select_organoids(
        args.label, args.max_day, args.run_dir, args.cohorts_dir,
        args.selection_mode, args.top_n, args.filter_label,
    )
    if not oids:
        print("[warn] no organoids selected — try --selection-mode lowest")
        return 1
    print(f"[select] {len(oids)} organoids ({args.selection_mode}, filter={args.filter_label})")

    series_meta = test_d  # series JSON has per-organoid timepoints
    att = run_inference(ckpt_path, oids, series_meta, args.max_day, device=args.device)

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.filter_label == "none" else f"_{args.filter_label.replace(' ', '')}"
    out_path = args.plots_dir / (
        f"tattn_{args.label}_max{args.max_day:g}_{args.selection_mode}{suffix}.png"
    )
    render_heatmap(att, test_d, misses, args.label, args.max_day,
                   args.selection_mode, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

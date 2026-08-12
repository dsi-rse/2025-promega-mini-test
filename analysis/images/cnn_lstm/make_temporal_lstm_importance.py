#!/usr/bin/env python3
"""
make_temporal_lstm_importance.py — visualize per-day importance for the
OrganoidCNN_LSTM model using TEMPORAL OCCLUSION SENSITIVITY.

The LSTM doesn't expose attention weights the way the attn model does. To get
a comparable "which day mattered" signal, we mask out each day's image one at
a time and measure how much the model's probability changes. Days that change
the prediction a lot were doing real work; days that don't matter much when
removed were being ignored.

Workflow per organoid:
    1. Forward the full sequence -> baseline probability p0.
    2. For each timepoint t in the included window:
         - Replace x[t] with zeros (== ImageNet-mean grey in normalized space)
         - Re-forward -> probability p_t
         - importance[t] = |p0 - p_t|
    3. Normalize so importance sums to 1.0 -> directly comparable to the
       attention model's softmax weights.

The output PNG is a heatmap identical in layout to make_temporal_attention.py
so the two can be displayed side by side ("attention vs occlusion importance").

Usage
-----
    python analysis/images/cnn_lstm/make_temporal_lstm_importance.py \\
        --label idor_minvotes3 \\
        --max-day 30 \\
        --top-n 10 \\
        --selection-mode missed
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_temporal_ablation_lstm import OrganoidCNN_LSTM  # noqa: E402
from analysis.images.cnn_lstm.organoid_dataset import OrganoidTimeSeriesDataset  # noqa: E402


# ============================================================
# Helpers — same selection logic as make_temporal_attention.py
# ============================================================

def select_organoids(label: str, run_dir: Path, cohorts_dir: Path,
                     mode: str, top_n: int, filter_label: str = "none"):
    """Rank test-set organoids by aggregated miss count across LSTM variants."""
    rpath = run_dir / label / "temporal_ablation_lstm" / "temporal_ablation_results_lstm.json"
    if not rpath.exists():
        raise FileNotFoundError(f"missing results JSON: {rpath}")
    with open(rpath) as f:
        results = json.load(f)

    misses = defaultdict(int)
    for variant in results:
        for oid in (variant.get("test_false_positives") or []):
            misses[oid] += 1
        for oid in (variant.get("test_false_negatives") or []):
            misses[oid] += 1

    test_p = cohorts_dir / label / "series" / "test.json"
    with open(test_p) as f:
        test_d = json.load(f)

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
# Occlusion-based importance
# ============================================================

def run_occlusion(ckpt_path: Path, oids: list[str], series_meta: dict,
                  max_day: float, device: str = "cuda") -> dict:
    dataset = OrganoidTimeSeriesDataset(
        oids, series_meta, max_day=max_day, image_type="clipped",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OrganoidCNN_LSTM().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    out: dict[str, dict] = {}
    with torch.no_grad():
        for seqs, days_norm, label, weight, oid in loader:
            seqs = seqs.to(device)             # (1, T, C, H, W)
            days_norm = days_norm.to(device).float()
            T = seqs.shape[1]

            # Baseline probability with full sequence
            logit = model(seqs, days_norm)
            p0 = torch.sigmoid(logit).item()
            pred = int(p0 >= 0.5)

            # Per-timepoint occlusion
            importance = np.zeros(T)
            for t in range(T):
                masked = seqs.clone()
                masked[:, t] = 0.0  # zero in normalized space == ImageNet-mean grey
                logit_t = model(masked, days_norm)
                p_t = torch.sigmoid(logit_t).item()
                importance[t] = abs(p0 - p_t)

            # Normalize so each row sums to 1, like the attn softmax weights
            total = importance.sum()
            if total > 0:
                importance = importance / total

            actual_days = [
                tp["mdl_day"] for tp in series_meta[oid[0]]["timepoints"]
                if tp["mdl_day"] <= max_day
            ]
            out[oid[0]] = {
                "importance": importance,
                "days":       actual_days,
                "p_full":     p0,
                "pred":       pred,
                "true":       int(label.item()),
            }
    return out


# ============================================================
# Render — same layout as make_temporal_attention.py for direct comparison
# ============================================================

def render_heatmap(att_data: dict, test_d: dict, misses: dict,
                   label: str, max_day: float, mode: str,
                   out_path: Path) -> None:
    oids = list(att_data.keys())
    if not oids:
        print("[warn] nothing to render"); return

    all_days = sorted({d for r in att_data.values() for d in r["days"]})
    matrix = np.zeros((len(oids), len(all_days)))
    for i, oid in enumerate(oids):
        for d, w in zip(att_data[oid]["days"], att_data[oid]["importance"]):
            j = all_days.index(d)
            matrix[i, j] = w

    fig_h = max(3.0, 0.4 * len(oids) + 1.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=matrix.max() if matrix.max() > 0 else 1)

    ax.set_xticks(range(len(all_days)))
    ax.set_xticklabels([f"Dy {d:g}" for d in all_days], fontsize=10)
    ax.set_yticks(range(len(oids)))

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

    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v > 0:
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("occlusion importance (|Δprob|, normalized)", fontsize=10)

    title = (f"LSTM temporal importance (occlusion) — {label}, "
             f"max_day = {max_day:g}, selection = {mode}")
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
    p.add_argument("--label", default="idor_minvotes3")
    p.add_argument("--max-day", type=float, default=30)
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

    ckpt_dir = args.run_dir / args.label / "temporal_ablation_lstm" / f"days_3-{args.max_day:g}"
    ckpt_path = ckpt_dir / f"model_days_3-{args.max_day:g}.pth"
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        return 1

    print(f"[load]  checkpoint {ckpt_path}")
    oids, test_d, misses = select_organoids(
        args.label, args.run_dir, args.cohorts_dir,
        args.selection_mode, args.top_n, args.filter_label,
    )
    if not oids:
        print("[warn] no organoids selected — try --selection-mode lowest"); return 1
    print(f"[select] {len(oids)} organoids ({args.selection_mode}, filter={args.filter_label})")
    print(f"[occlude] running {args.max_day:g}-day occlusion on each organoid "
          f"(this is num_organoids × num_days forward passes)")

    att = run_occlusion(ckpt_path, oids, test_d, args.max_day, device=args.device)

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.filter_label == "none" else f"_{args.filter_label.replace(' ', '')}"
    out_path = args.plots_dir / (
        f"tlstm_{args.label}_max{args.max_day:g}_{args.selection_mode}{suffix}.png"
    )
    render_heatmap(att, test_d, misses, args.label, args.max_day,
                   args.selection_mode, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

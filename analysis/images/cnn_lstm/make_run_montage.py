#!/usr/bin/env python3
"""
make_run_montage.py — assemble one PNG montage per cohort training run, and
build cross-cohort comparison plots.

Two modes
---------

Per-cohort montage (one PNG summarizing all 4 models for one cohort run):
    python analysis/images/cnn_lstm/make_run_montage.py \
        --run-dir /net/projects2/promega/project_data/model_tests/lstm_runs \
        --label   expanded_minvotes3 \
        --output-dir /net/projects2/promega/project_data/amanda_test/model_plots

Cross-cohort comparison (one PNG of test_acc + test_f1 across cohorts):
    python analysis/images/cnn_lstm/make_run_montage.py \
        --run-dir /net/projects2/promega/project_data/model_tests/lstm_runs \
        --compare idor idor_minvotes3 expanded expanded_minvotes3 \
        --output-dir /net/projects2/promega/project_data/amanda_test/model_plots

Layout (per-cohort montage)
---------------------------
3 rows (one per model) x 3 columns:
    col 1: test acc/F1 vs day (ablation curve)
    col 2: confusion matrix (best-F1 day for ablation models)
    col 3: text block with headline test metrics

All three models (base_effnet, temporal_ablation_attn, temporal_ablation_lstm)
emit lists of per-day results. We pick the entry with the highest test_f1 as the
"headline" view and embed its already-saved confusion-matrix PNG.

Expected directory layout after `bash run_all_lstm.sh <label>`:

    <run-dir>/<label>/
        base_effnet/
            baseline_results.json
            day_<N>/confusion_matrix_day_<N>.png
        temporal_ablation_attn/
            temporal_ablation_results.json
            days_3-<N>/confusion_matrix_days_3-<N>.png
        temporal_ablation_lstm/
            temporal_ablation_results_lstm.json
            days_3-<N>/confusion_matrix_days_3-<N>.png

If a model's outputs are missing, that row of the montage is rendered as a
placeholder rather than crashing — useful when you've only run a subset of
the three trainers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


# ---- Default plot styling ----
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 100


# ============================================================
# Model registry — names, expected files, how to find CMs
# ============================================================

# Each model spec describes:
#   subdir          – folder name under <run-dir>/<label>/
#   results_file    – filename of the JSON inside that folder
#   kind            – "single" (one result dict) or "ablation" (list of dicts)
#   ablation_key    – which field varies across the list (e.g. "target_day")
#   cm_subdir_fmt   – format string for the per-result subdir (ablation only)
#   cm_file_fmt     – format string for the CM PNG inside that subdir
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "label":         "base_effnet",
        "title":         "EfficientNet (per-day baseline)",
        "subdir":        "base_effnet",
        "results_file":  "baseline_results.json",
        "kind":          "ablation",
        "ablation_key":  "target_day",
        "ablation_xlabel": "Day",
        "cm_subdir_fmt": "day_{val}",
        "cm_file_fmt":   "confusion_matrix_day_{val}.png",
    },
    {
        "label":         "temporal_ablation_attn",
        "title":         "Temporal Ablation (Attention)",
        "subdir":        "temporal_ablation_attn",
        "results_file":  "temporal_ablation_results.json",
        "kind":          "ablation",
        "ablation_key":  "max_day",
        "ablation_xlabel": "Max day (cumulative)",
        "cm_subdir_fmt": "days_3-{val}",
        "cm_file_fmt":   "confusion_matrix_days_3-{val}.png",
    },
    {
        "label":         "temporal_ablation_lstm",
        "title":         "Temporal Ablation (LSTM)",
        "subdir":        "temporal_ablation_lstm",
        "results_file":  "temporal_ablation_results_lstm.json",
        "kind":          "ablation",
        "ablation_key":  "max_day",
        "ablation_xlabel": "Max day (cumulative)",
        "cm_subdir_fmt": "days_3-{val}",
        "cm_file_fmt":   "confusion_matrix_days_3-{val}.png",
    },
]


# ============================================================
# Loaders
# ============================================================

def load_test_split_balance(cohorts_dir: Path, label: str) -> tuple[int, int] | None:
    """
    Read the cohort's series test split and return (n_pos, n_neg) where pos =
    'Acceptable'. Used to reconstruct balanced accuracy from each result's
    test_false_positives / test_false_negatives counts.
    """
    p = cohorts_dir / label / "series" / "test.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    n_pos = sum(1 for v in d.values() if v.get("label") == "Acceptable")
    n_neg = len(d) - n_pos
    return n_pos, n_neg


def balanced_acc_for_result(result: dict, n_pos: int, n_neg: int) -> float | None:
    """
    Prefer a 'test_balanced_acc' field if the trainer saved it directly.
    Otherwise reconstruct from test_false_positives / test_false_negatives
    plus the test split size.

        balanced_acc = (recall + specificity) / 2
                     = (TP / (TP+FN) + TN / (TN+FP)) / 2

    FP = positives predicted by mistake on actual negatives  -> 'Bad' called 'Good'
    FN = negatives predicted by mistake on actual positives  -> 'Good' called 'Bad'
    """
    if "test_balanced_acc" in result and result["test_balanced_acc"] is not None:
        return float(result["test_balanced_acc"])

    fp_list = result.get("test_false_positives")
    fn_list = result.get("test_false_negatives")
    if fp_list is None or fn_list is None or n_pos is None or n_neg is None:
        return None

    fp = len(fp_list)
    fn = len(fn_list)
    tp = n_pos - fn
    tn = n_neg - fp
    if (tp + fn) == 0 or (tn + fp) == 0:
        return None
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (recall + specificity) / 2


def load_results(run_dir: Path, label: str, spec: dict) -> dict | list | None:
    p = run_dir / label / spec["subdir"] / spec["results_file"]
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] failed to load {p}: {e}")
        return None


def pick_best_ablation_entry(results: list, metric: str = "test_f1") -> dict | None:
    if not results:
        return None
    return max(results, key=lambda r: r.get(metric, float("-inf")))


def _fmt_day_val(v) -> str:
    """Match the trainer's naming: integer days have no decimal, 20.5 stays as 20.5."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# ============================================================
# Drawing helpers (per cell of the montage)
# ============================================================

def draw_training_curves(ax, results: dict, title: str) -> None:
    """For cnn_lstm-style single results with a train_history list of dicts."""
    history = results.get("train_history") or []
    if not history:
        ax.text(0.5, 0.5, "no train_history", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    epochs    = [h.get("epoch", i) for i, h in enumerate(history)]
    train_acc = [h.get("train_acc") for h in history]
    val_acc   = [h.get("val_acc")   for h in history]
    train_loss = [h.get("train_loss") for h in history]
    val_loss   = [h.get("val_loss")   for h in history]

    if any(a is not None for a in train_acc):
        ax.plot(epochs, train_acc, label="train acc", color="tab:blue")
    if any(a is not None for a in val_acc):
        ax.plot(epochs, val_acc,   label="val acc",   color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    # Loss on a twin axis
    ax2 = ax.twinx()
    if any(l is not None for l in train_loss):
        ax2.plot(epochs, train_loss, label="train loss", color="tab:blue",
                 linestyle="--", alpha=0.6)
    if any(l is not None for l in val_loss):
        ax2.plot(epochs, val_loss, label="val loss", color="tab:orange",
                 linestyle="--", alpha=0.6)
    ax2.set_ylabel("Loss")

    # Combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax.set_title(title)


def draw_ablation_curve(ax, results: list, x_key: str, xlabel: str, title: str,
                        n_pos: int = None, n_neg: int = None) -> None:
    """
    For ablation-style list results: test_acc / test_bal_acc / test_f1 vs the
    varying day-window key, plus best_val_acc as a faint reference line.

    If n_pos and n_neg are provided, missing test_balanced_acc values are
    reconstructed from FP/FN lists so the line draws even for older runs that
    didn't save the field natively.
    """
    if not results:
        ax.text(0.5, 0.5, "no results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    rs = sorted(results, key=lambda r: r.get(x_key, 0))
    xs       = [r.get(x_key) for r in rs]
    test_acc = [r.get("test_acc") for r in rs]
    test_bal = [balanced_acc_for_result(r, n_pos, n_neg) for r in rs]
    test_f1  = [r.get("test_f1")  for r in rs]
    val_acc  = [r.get("best_val_acc") for r in rs]

    ax.plot(xs, test_acc, "o-", label="test acc", color="tab:green")
    if any(v is not None for v in test_bal):
        ax.plot(xs, test_bal, "D-", label="test bal acc", color="tab:purple")
    ax.plot(xs, test_f1,  "s-", label="test F1",  color="tab:red")
    if any(v is not None for v in val_acc):
        ax.plot(xs, val_acc, "x--", label="best val acc",
                color="tab:gray", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)


def draw_confusion_from_array(ax, cm: list, title: str) -> None:
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Accept", "Accept"])
    ax.set_yticklabels(["Not Accept", "Accept"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Annotate cells
    vmax = cm_arr.max() if cm_arr.size else 1
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            v = int(cm_arr[i, j])
            color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")


def draw_confusion_from_png(ax, png_path: Path, title: str) -> None:
    if not png_path.exists():
        ax.text(0.5, 0.5, f"missing CM:\n{png_path.name}", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title)
        ax.axis("off")
        return
    img = mpimg.imread(str(png_path))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def draw_metrics_text(ax, lines: list[str], title: str = "") -> None:
    ax.axis("off")
    txt = "\n".join(lines)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            ha="left", va="top", fontfamily="monospace", fontsize=10)
    if title:
        ax.set_title(title)


def fmt_metric(v) -> str:
    return "N/A" if v is None else f"{v:.3f}"


# ============================================================
# Per-cohort montage
# ============================================================

def render_per_cohort(run_dir: Path, label: str, output_dir: Path,
                      cohorts_dir: Path) -> Path:
    cohort_dir = run_dir / label
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Cohort run dir not found: {cohort_dir}")

    # Load test split balance once — used to reconstruct balanced_acc for any
    # result that didn't save it directly.
    split_balance = load_test_split_balance(cohorts_dir, label)
    n_pos, n_neg = split_balance if split_balance else (None, None)

    n_rows = len(MODEL_SPECS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row, spec in enumerate(MODEL_SPECS):
        results = load_results(run_dir, label, spec)
        ax_left, ax_mid, ax_right = axes[row, 0], axes[row, 1], axes[row, 2]

        if results is None:
            for a in (ax_left, ax_mid, ax_right):
                a.axis("off")
            ax_left.text(0.5, 0.5, f"[{spec['title']}]\nno results file",
                         ha="center", va="center", transform=ax_left.transAxes,
                         fontsize=11)
            continue

        if spec["kind"] == "single":
            # Training curves
            draw_training_curves(ax_left, results, f"{spec['title']} — training")
            # Confusion matrix
            cm = results.get("confusion_matrix")
            if cm is not None:
                draw_confusion_from_array(ax_mid, cm, "Test confusion matrix")
            else:
                cm_png = cohort_dir / spec["subdir"] / spec.get("cm_file", "")
                draw_confusion_from_png(ax_mid, cm_png, "Test confusion matrix")
            # Metrics text
            bal = balanced_acc_for_result(results, n_pos, n_neg) if isinstance(results, dict) else None
            lines = [
                f"best_val_acc   {fmt_metric(results.get('best_val_acc'))}",
                f"best_val_loss  {fmt_metric(results.get('best_val_loss'))}",
                f"test_acc       {fmt_metric(results.get('test_acc'))}",
                f"test_bal_acc   {fmt_metric(bal)}",
                f"test_f1        {fmt_metric(results.get('test_f1'))}",
                f"test_precision {fmt_metric(results.get('test_precision'))}",
                f"test_recall    {fmt_metric(results.get('test_recall'))}",
            ]
            draw_metrics_text(ax_right, lines, title="Final metrics")

        else:  # ablation
            x_key = spec["ablation_key"]
            xlabel = spec["ablation_xlabel"]
            draw_ablation_curve(ax_left, results, x_key, xlabel,
                                f"{spec['title']} — test metrics vs {xlabel}",
                                n_pos=n_pos, n_neg=n_neg)

            best = pick_best_ablation_entry(results, metric="test_f1")
            if best is None:
                ax_mid.axis("off")
                draw_metrics_text(ax_right, ["no results"])
                continue

            val_str = _fmt_day_val(best.get(x_key))
            cm_subdir = spec["cm_subdir_fmt"].format(val=val_str)
            cm_file   = spec["cm_file_fmt"].format(val=val_str)
            cm_png    = cohort_dir / spec["subdir"] / cm_subdir / cm_file
            draw_confusion_from_png(
                ax_mid, cm_png,
                f"Confusion matrix — best ({xlabel.split()[0].lower()}={val_str})",
            )

            best_bal = balanced_acc_for_result(best, n_pos, n_neg)
            lines = [
                f"BEST {x_key}      {val_str}",
                f"  test_acc       {fmt_metric(best.get('test_acc'))}",
                f"  test_bal_acc   {fmt_metric(best_bal)}",
                f"  test_f1        {fmt_metric(best.get('test_f1'))}",
                f"  test_precision {fmt_metric(best.get('test_precision'))}",
                f"  test_recall    {fmt_metric(best.get('test_recall'))}",
                f"  best_val_acc   {fmt_metric(best.get('best_val_acc'))}",
                "",
                f"({len(results)} variants total; "
                f"best by test_f1)",
            ]
            draw_metrics_text(ax_right, lines, title="Headline (best variant)")

    fig.suptitle(f"Cohort: {label}", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"montage_{label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# Cross-cohort comparison
# ============================================================

def _headline_metric(results, kind: str, metric: str):
    """Return the single 'headline' metric for a given model's results."""
    if results is None:
        return None
    if kind == "single":
        return results.get(metric)
    best = pick_best_ablation_entry(results, metric="test_f1")
    return None if best is None else best.get(metric)


def _headline_balanced_acc(results, kind: str, n_pos: int, n_neg: int):
    """Headline balanced_accuracy for the model's best variant (or single result)."""
    if results is None:
        return None
    if kind == "single":
        return balanced_acc_for_result(results, n_pos, n_neg)
    best = pick_best_ablation_entry(results, metric="test_f1")
    return None if best is None else balanced_acc_for_result(best, n_pos, n_neg)


def render_comparison(run_dir: Path, labels: list[str], output_dir: Path,
                      cohorts_dir: Path) -> Path:
    n_models  = len(MODEL_SPECS)
    n_cohorts = len(labels)

    # Pre-load test split balance per cohort (for post-hoc balanced_acc)
    splits = {}
    for lab in labels:
        splits[lab] = load_test_split_balance(cohorts_dir, lab) or (None, None)

    # Collect headline metrics
    matrix_acc = np.full((n_models, n_cohorts), np.nan)
    matrix_bal = np.full((n_models, n_cohorts), np.nan)
    matrix_f1  = np.full((n_models, n_cohorts), np.nan)
    for c_idx, lab in enumerate(labels):
        n_pos, n_neg = splits[lab]
        for m_idx, spec in enumerate(MODEL_SPECS):
            r = load_results(run_dir, lab, spec)
            matrix_acc[m_idx, c_idx] = _headline_metric(r, spec["kind"], "test_acc") or np.nan
            matrix_bal[m_idx, c_idx] = _headline_balanced_acc(r, spec["kind"], n_pos, n_neg) or np.nan
            matrix_f1 [m_idx, c_idx] = _headline_metric(r, spec["kind"], "test_f1")  or np.nan

    fig, axes = plt.subplots(n_models, 3, figsize=(16, 3.5 * n_models))
    if n_models == 1:
        axes = np.array([axes])
    x = np.arange(n_cohorts)
    bar_w = 0.6

    for m_idx, spec in enumerate(MODEL_SPECS):
        for col, (metric_name, mat) in enumerate(
            (("test_acc", matrix_acc),
             ("test_bal_acc", matrix_bal),
             ("test_f1",  matrix_f1))
        ):
            ax = axes[m_idx, col]
            vals = mat[m_idx]
            ax.bar(x, vals, bar_w, color="tab:blue", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_title(f"{spec['title']} — {metric_name}", fontsize=10)
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.02, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

    fig.suptitle("Cohort comparison — headline test metrics",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_vs_".join(labels)
    out_path = output_dir / f"compare_{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# Entrypoint
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Top-level lstm_runs directory containing per-cohort subdirs.")
    p.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"),
                   help=("Root containing data/cohorts/<label>/series/test.json. "
                         "Used to reconstruct balanced accuracy post-hoc when the "
                         "trainer didn't save it. Default: data/cohorts/"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/net/projects2/promega/project_data/amanda_test/model_plots"),
                   help="Where montage PNGs are written.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--label", type=str,
                   help="Cohort label (subdir under --run-dir) to summarize.")
    g.add_argument("--compare", nargs="+",
                   help="Two or more cohort labels to compare side-by-side.")
    args = p.parse_args()

    if args.label:
        out = render_per_cohort(args.run_dir, args.label, args.output_dir,
                                args.cohorts_dir)
        print(f"[done]  wrote {out}")
    else:
        if len(args.compare) < 2:
            p.error("--compare requires at least 2 labels")
        out = render_comparison(args.run_dir, args.compare, args.output_dir,
                                args.cohorts_dir)
        print(f"[done]  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

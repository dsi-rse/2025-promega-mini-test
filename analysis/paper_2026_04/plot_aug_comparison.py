#!/usr/bin/env python3
"""Augmentation vs no-augmentation comparison for EfficientNet-B0 image classifier.

Reads:
  - analysis_output/images/perday_results_kfold_series_idor.json       (with aug)
  - analysis_output/images/perday_results_kfold_series_idor_noaug.json (no aug)

Output:
  - figures/aug_comparison_series_idor.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.plot_aug_comparison"
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER

_AUG_PATH   = ANALYSIS_OUTPUT_DIR / "images" / "perday_results_kfold_series_idor_139.json"
_NOAUG_PATH = ANALYSIS_OUTPUT_DIR / "images" / "perday_results_kfold_series_idor_139_noaug.json"
_FIGURE_DIR = ANALYSIS_OUTPUT_DIR / "figures"
_REPO_DIR   = Path("figures")
_OUT_NAME   = "aug_comparison_series_idor_139.png"


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    aug   = _load(_AUG_PATH)
    noaug = _load(_NOAUG_PATH)

    days = [d for d in DAY_ORDER if d in aug or d in noaug]
    xs   = list(range(len(days)))

    aug_mean, aug_std     = [], []
    noaug_mean, noaug_std = [], []
    deltas                = []

    for d in days:
        a  = aug.get(d, {})
        n  = noaug.get(d, {})
        am = a.get("balanced_accuracy_mean", a.get("balanced_accuracy", np.nan))
        asd = a.get("balanced_accuracy_std", np.nan)
        nm = n.get("balanced_accuracy_mean", n.get("balanced_accuracy", np.nan))
        nsd = n.get("balanced_accuracy_std", np.nan)
        aug_mean.append(am);   aug_std.append(asd)
        noaug_mean.append(nm); noaug_std.append(nsd)
        if not np.isnan(am) and not np.isnan(nm):
            deltas.append(am - nm)
        else:
            deltas.append(np.nan)

    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    # ── Main panel ────────────────────────────────────────────────────────────
    def _plot_series(ax, ys, stds, color, marker, label):
        valid = [(i, y, s) for i, (y, s) in enumerate(zip(ys, stds))
                 if not np.isnan(y)]
        if not valid:
            return
        xi, yi, si = zip(*valid)
        line, = ax.plot(xi, yi, marker=marker, color=color, linewidth=2,
                        markersize=7, label=label)
        c = line.get_color()
        lo = [y - s if not np.isnan(s) else y for y, s in zip(yi, si)]
        hi = [y + s if not np.isnan(s) else y for y, s in zip(yi, si)]
        ax.fill_between(xi, lo, hi, color=c, alpha=0.15, linewidth=0)

    _plot_series(ax_main, aug_mean,   aug_std,   "#1f77b4", "o", "With augmentation")
    _plot_series(ax_main, noaug_mean, noaug_std, "#d62728", "s", "No augmentation")

    ax_main.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6,
                    label="Chance (0.50)")
    ax_main.set_ylabel("Balanced Accuracy (5-fold CV mean ± 1 SD)", fontsize=11)
    ax_main.set_ylim(0.35, 1.05)
    ax_main.grid(True, alpha=0.25)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.legend(fontsize=10, loc="upper left")
    ax_main.set_title(
        "EfficientNet-B0: Augmentation vs No Augmentation\n"
        "(4-fold CV, series_idor, n = 139 incl. stitched, cm_image input)",
        fontsize=12, fontweight="bold",
    )

    # ── Delta panel ───────────────────────────────────────────────────────────
    colors = ["#1f77b4" if d >= 0 else "#d62728"
              for d in [v if not np.isnan(v) else 0 for v in deltas]]
    ax_delta.bar(xs, [v if not np.isnan(v) else 0 for v in deltas],
                 color=colors, alpha=0.7, width=0.6)
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_ylabel("Δ BA\n(aug − noaug)", fontsize=9)
    ax_delta.set_ylim(-0.25, 0.25)
    ax_delta.grid(True, alpha=0.25, axis="y")
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.set_xticks(xs)
    ax_delta.set_xticklabels(days, rotation=45, fontsize=9)
    ax_delta.set_xlabel("Day", fontsize=11)

    # Annotate delta values
    for i, d in enumerate(deltas):
        if np.isnan(d):
            continue
        va = "bottom" if d >= 0 else "top"
        offset = 0.01 if d >= 0 else -0.01
        ax_delta.text(i, d + offset, f"{d:+.3f}", ha="center", va=va,
                      fontsize=7.5, fontweight="bold",
                      color="#1f77b4" if d >= 0 else "#d62728")

    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _REPO_DIR.mkdir(exist_ok=True)
    out = _FIGURE_DIR / _OUT_NAME
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(out, _REPO_DIR / _OUT_NAME)
    print(f"Saved {_REPO_DIR / _OUT_NAME}")

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'Day':<8}  {'Aug mean+/-std':>16}  {'NoAug mean+/-std':>18}  {'Delta':>7}")
    print("-" * 58)
    for i, d in enumerate(days):
        am  = aug_mean[i];   asd = aug_std[i]
        nm  = noaug_mean[i]; nsd = noaug_std[i]
        aug_str   = f"{am:.3f}+/-{asd:.3f}" if not np.isnan(am)  else "—"
        noaug_str = f"{nm:.3f}+/-{nsd:.3f}" if not np.isnan(nm)  else "—"
        delta_str = f"{deltas[i]:+.3f}"     if not np.isnan(deltas[i]) else "—"
        print(f"{d:<8}  {aug_str:>16}  {noaug_str:>18}  {delta_str:>7}")


if __name__ == "__main__":
    main()

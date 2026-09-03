#!/usr/bin/env python3
"""Two-panel figure: classifier comparison across malate variants.

Left panel:  4 classifiers on met_nan — which model performs best?
Right panel: Per-classifier malate sensitivity — does nan ≈ raw ≈ no_malate?
             Shows mean BA difference (raw − nan, no_malate − nan) per day.

Also saves a table PNG and prints a text summary.

Reads:
  analysis_output/images/met_classifier_comparison.json

Outputs:
  figures/met_classifier_comparison.png
  figures/met_classifier_table.png

Usage:
    python3 -m analysis.paper_2026_04.plot_met_classifier_comparison
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER

_INPUT_PATH = ANALYSIS_OUTPUT_DIR / "images" / "met_classifier_comparison.json"
_FIGURE_DIR = ANALYSIS_OUTPUT_DIR / "figures"
_REPO_FIG   = Path("figures")

CLF_DISPLAY = {
    "lgbm":   ("LightGBM",           "#1f77b4", "o", "-"),
    "logreg": ("Logistic Regression", "#ff7f0e", "s", "-"),
    "svm":    ("SVM (RBF)",           "#2ca02c", "^", "-"),
    "mlp":    ("MLP",                 "#d62728", "D", "-"),
}

MAL_DISPLAY = {
    "met_nan":       ("Floor→NaN",    "#1f77b4", "o", "-"),
    "met_raw":       ("Raw values",   "#ff7f0e", "s", "--"),
    "met_no_malate": ("Drop MalateGlo","#2ca02c", "^", ":"),
}


def _load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        return json.load(f)


def _get(r, day, key):
    if day not in r:
        return None, None
    v = r[day].get(key)
    if v is None:
        return None, None
    return v["balanced_accuracy_mean"], v.get("balanced_accuracy_std")


def _plot_left(ax, r, days):
    """4 classifiers using met_nan."""
    for clf, (label, color, marker, ls) in CLF_DISPLAY.items():
        xs, ys, lo, hi = [], [], [], []
        for i, day in enumerate(days):
            mean, std = _get(r, day, f"{clf}_met_nan")
            if mean is None:
                continue
            xs.append(i)
            ys.append(mean)
            lo.append(mean - (std or 0))
            hi.append(mean + (std or 0))
        if not xs:
            continue
        line, = ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                        linewidth=2, markersize=6, label=label)
        if any(l != h for l, h in zip(lo, hi)):
            ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15, linewidth=0)

    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=45, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylabel("Balanced Accuracy (mean ± 1 SD)", fontsize=10)
    ax.set_xlabel("Day", fontsize=10)
    ax.set_title("Classifier comparison (met_nan)\n10×4-fold CV, series_idor, n=139",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")


def _plot_right(ax, r, days):
    """Per-classifier: raw−nan and no_malate−nan differences."""
    x = np.arange(len(days))
    width = 0.2
    offsets = {"lgbm": -1.5, "logreg": -0.5, "svm": 0.5, "mlp": 1.5}

    for clf, (label, color, _, _) in CLF_DISPLAY.items():
        diffs_raw, diffs_drop = [], []
        for day in days:
            m_nan, _ = _get(r, day, f"{clf}_met_nan")
            m_raw, _ = _get(r, day, f"{clf}_met_raw")
            m_drop, _ = _get(r, day, f"{clf}_met_no_malate")
            if m_nan is None or m_raw is None or m_drop is None:
                diffs_raw.append(0.0)
                diffs_drop.append(0.0)
            else:
                diffs_raw.append(m_raw - m_nan)
                diffs_drop.append(m_drop - m_nan)

        xpos = x + offsets[clf] * width
        bars = ax.bar(xpos, diffs_raw, width=width * 0.85, color=color, alpha=0.7,
                      label=f"{label} (raw−nan)")
        ax.bar(xpos, diffs_drop, width=width * 0.85, color=color, alpha=0.35,
               hatch="//", label=f"{label} (drop−nan)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, fontsize=9)
    ax.set_ylabel("BA difference vs met_nan", fontsize=10)
    ax.set_xlabel("Day", fontsize=10)
    ax.set_title("Malate correction sensitivity\n(raw−nan, drop−nan per classifier)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Compact legend with two entries per classifier (solid + hatch)
    handles, labels = ax.get_legend_handles_labels()
    # Show only raw−nan entries for a cleaner legend
    h = [h for h, l in zip(handles, labels) if "raw−nan" in l]
    la = [l.replace(" (raw−nan)", "") for l in labels if "raw−nan" in l]
    ax.legend(h, la, fontsize=8, loc="upper left", title="Classifier (solid=raw−nan, hatch=drop−nan)")


def _save_table(r, days):
    """Render mean±std table: classifiers × days, one section per malate variant."""
    rows, row_labels = [], []
    for clf, (clf_label, _, _, _) in CLF_DISPLAY.items():
        for mk, (mal_label, _, _, _) in MAL_DISPLAY.items():
            row = []
            for day in days:
                mean, std = _get(r, day, f"{clf}_{mk}")
                if mean is None:
                    row.append("—")
                elif std is not None:
                    row.append(f"{mean:.3f}\n±{std:.3f}")
                else:
                    row.append(f"{mean:.3f}")
            rows.append(row)
            row_labels.append(f"{clf_label}\n({mal_label})")

    n_rows, n_cols = len(rows), len(days)
    fig_w = 2.5 + n_cols * 1.1
    fig_h = 0.6 + n_rows * 0.65

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, rowLabels=row_labels, colLabels=days,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 2.1)

    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    clf_colors = {"lgbm": "#D9E1F2", "logreg": "#E2EFDA", "svm": "#FCE4D6", "mlp": "#FFF2CC"}
    clf_list = list(CLF_DISPLAY.keys())
    for i in range(n_rows):
        clf = clf_list[i // len(MAL_DISPLAY)]
        tbl[(i + 1, -1)].set_facecolor(clf_colors[clf])
        tbl[(i + 1, -1)].set_text_props(fontweight="bold")
        for j in range(n_cols):
            if i % 2 == 1:
                tbl[(i + 1, j)].set_facecolor("#F5F5F5")

    fig.suptitle("Met-classifier comparison: BA mean ± std (10×4-fold CV)",
                 fontsize=10, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    name = "met_classifier_table.png"
    out  = _FIGURE_DIR / name
    repo = _REPO_FIG / name
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _REPO_FIG.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(out, repo)
    print(f"Saved table → {repo}")


def _print_table(r, days):
    col_w = 12
    print("\n" + "=" * 80)
    print("Met-classifier comparison: mean BA (10×4-fold repeated CV, n=139)")
    print("=" * 80)
    for mk, (mal_label, _, _, _) in MAL_DISPLAY.items():
        print(f"\n  Variant: {mal_label}")
        hdr = f"  {'Day':<8}" + "".join(f"{c:>{col_w}}" for c in CLF_DISPLAY)
        print(hdr)
        print("  " + "-" * (8 + 4 * col_w))
        for day in days:
            row = f"  {day:<8}"
            for clf in CLF_DISPLAY:
                mean, std = _get(r, day, f"{clf}_{mk}")
                if mean is None:
                    cell = "—"
                elif std is not None:
                    cell = f"{mean:.3f}±{std:.3f}"
                else:
                    cell = f"{mean:.3f}"
                row += f"{cell:>{col_w}}"
            print(row)
    print("=" * 80)


def main():
    r = _load(_INPUT_PATH)
    days = [d for d in DAY_ORDER if d in r]
    if not days:
        raise RuntimeError(f"No days found in {_INPUT_PATH}")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"wspace": 0.35})
    _plot_left(ax_l, r, days)
    _plot_right(ax_r, r, days)

    fig.suptitle("Metabolite Classifier Comparison — Organoid Quality",
                 fontsize=13, fontweight="bold", y=1.02)

    name = "met_classifier_comparison.png"
    out  = _FIGURE_DIR / name
    repo = _REPO_FIG / name
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _REPO_FIG.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(out, repo)
    print(f"Saved figure → {repo}")

    _print_table(r, days)
    _save_table(r, days)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two-panel balanced-accuracy comparison: single modalities vs fusion strategies.

Left panel:  Metabolite (nan/raw/no-malate) / Morphology / Image — all from the
             shared-split combined_kfold run (139 organoids, 10×4-fold CV).
Right panel: Fusion combinations from the same run.

Also prints a plain-text table of mean +/- std for every strategy at every day.

Reads:
  - analysis_output/images/combined_results_kfold_series_idor_139.json

Output:
  - figures/combined_kfold_two_panel_series_idor_139.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.plot_combined_comparison"
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER

# ── Paths ────────────────────────────────────────────────────────────────────
_COMBINED_PATH = ANALYSIS_OUTPUT_DIR / "images" / "combined_results_kfold_series_idor_139.json"
_FIGURE_DIR    = ANALYSIS_OUTPUT_DIR / "figures"
_REPO_FIG_DIR  = Path("figures")
_OUT_NAME      = "combined_kfold_two_panel_series_idor_139.png"


def _load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p) as f:
        return json.load(f)


def _get_ba(d: dict):
    """Return (mean, std) from a result dict, preferring fold-level mean."""
    mean = d.get("balanced_accuracy_mean", d.get("balanced_accuracy"))
    std  = d.get("balanced_accuracy_std")
    return mean, std


def _build_series(results_by_day, strategy_key=None):
    """Build {day: result_dict} for one strategy.

    strategy_key=None  → results_by_day is {day: result_dict}
    strategy_key given → results_by_day is {day: {strategy_key: result_dict}}
    """
    out = {}
    for day in DAY_ORDER:
        if day not in results_by_day:
            continue
        entry = results_by_day[day]
        if strategy_key is not None:
            entry = entry.get(strategy_key)
        if entry is not None:
            out[day] = entry
    return out


def _plot_panel(ax, series_cfg, days):
    """Draw one panel. series_cfg: list of (label, day_results, color, marker, ls)."""
    for label, day_results, color, marker, ls in series_cfg:
        xs, ys, lo, hi = [], [], [], []
        for i, day in enumerate(days):
            r = day_results.get(day)
            if r is None:
                continue
            mean, std = _get_ba(r)
            if mean is None:
                continue
            xs.append(i)
            ys.append(mean)
            lo.append(mean - std if std is not None else mean)
            hi.append(mean + std if std is not None else mean)

        if not xs:
            continue
        line, = ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                        linewidth=2, markersize=6, label=label)
        c = line.get_color()
        if any(l != h for l, h in zip(lo, hi)):
            ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)

    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=45, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylabel("Balanced Accuracy (CV mean ± 1 SD)", fontsize=10)
    ax.set_xlabel("Day", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")


def _print_table(all_series, days):
    """Print mean +/- std table to stdout."""
    labels = [label for label, _, _, _, _ in all_series]
    col_w = 14  # wide enough for "0.818+/-0.062"

    header = f"{'Day':<8}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    sep = "=" * len(header)
    print()
    print(sep)
    print("Balanced Accuracy: mean +/- std (10×4-fold repeated CV, n=139)")
    print(sep)
    print(header)
    print("-" * len(header))

    for day in days:
        row = f"{day:<8}"
        for label, day_results, *_ in all_series:
            r = day_results.get(day)
            if r is None:
                cell = "—"
            else:
                mean, std = _get_ba(r)
                if mean is None:
                    cell = "—"
                elif std is not None:
                    cell = f"{mean:.3f}+/-{std:.3f}"
                else:
                    cell = f"{mean:.3f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)


def _save_table_figure(all_series, days):
    """Render mean±std table as a PNG — strategies as rows, days as columns."""
    labels = [label for label, _, _, _, _ in all_series]

    # Build cell strings: one row per strategy, one column per day
    cell_data = []
    for label, day_results, *_ in all_series:
        row = []
        for day in days:
            r = day_results.get(day)
            if r is None:
                row.append("—")
            else:
                mean, std = _get_ba(r)
                if mean is None:
                    row.append("—")
                elif std is not None:
                    row.append(f"{mean:.3f}\n±{std:.3f}")
                else:
                    row.append(f"{mean:.3f}")
        cell_data.append(row)

    n_rows = len(labels)   # strategies
    n_cols = len(days)     # days
    col_w  = 1.05
    row_h  = 0.55
    fig_w  = 2.5 + n_cols * col_w
    fig_h  = 0.6 + n_rows * row_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=labels,
        colLabels=days,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.9)

    # Header row (days) styling
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    # Row label (strategy) styling
    for i in range(n_rows):
        cell = tbl[(i + 1, -1)]
        cell.set_facecolor("#D9E1F2")
        cell.set_text_props(fontweight="bold")
    # Alternating row shading
    for i in range(n_rows):
        for j in range(n_cols):
            if i % 2 == 1:
                tbl[(i + 1, j)].set_facecolor("#EEF2FA")

    fig.suptitle(
        "Balanced Accuracy: mean ± std (10×4-fold CV) — All Strategies by Day",
        fontsize=10, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    tbl_name = "combined_kfold_table_series_idor_139.png"
    out_path  = _FIGURE_DIR / tbl_name
    repo_path = _REPO_FIG_DIR / tbl_name
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _REPO_FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(out_path, repo_path)
    print(f"Saved table figure → {repo_path}")


def main():
    combined_raw = _load(_COMBINED_PATH)

    days = [d for d in DAY_ORDER if d in combined_raw]

    # ── series config: (label, {day: result_dict}, color, marker, linestyle) ─
    left_series = [
        ("Metabolite",  _build_series(combined_raw, "met_nan"), "#2ca02c", "o", "-"),
        ("Morphology",  _build_series(combined_raw, "morph"),   "#9467bd", "s", "-"),
        ("Image",       _build_series(combined_raw, "img"),     "#1f77b4", "^", "-"),
    ]

    right_series = [
        ("Met + Morph",      _build_series(combined_raw, "met_nan+morph_mean_prob"),          "#e377c2", "o",  "-"),
        ("Met + Img",        _build_series(combined_raw, "met_nan+img_mean_prob"),             "#8c564b", "s",  "-"),
        ("Morph + Img",      _build_series(combined_raw, "morph+img_mean_prob"),              "#17becf", "^",  "-"),
        ("All Three (mean)", _build_series(combined_raw, "met_nan+morph+img_mean_prob"),      "#d62728", "D",  "-"),
        ("All Three (vote)", _build_series(combined_raw, "met_nan+morph+img_majority_vote"),  "#ff7f0e", "P",  "--"),
    ]

    all_series = left_series + right_series

    # Full set for table
    table_series = [
        ("Metabolite",       _build_series(combined_raw, "met_nan"),                          "#2ca02c", "o", "-"),
        ("Morphology",       _build_series(combined_raw, "morph"),                            "#9467bd", "s", "-"),
        ("Image",            _build_series(combined_raw, "img"),                              "#1f77b4", "^", "-"),
        ("Met+Morph",        _build_series(combined_raw, "met_nan+morph_mean_prob"),          "#e377c2", "o", "-"),
        ("Met+Img",          _build_series(combined_raw, "met_nan+img_mean_prob"),             "#8c564b", "s", "-"),
        ("Morph+Img",        _build_series(combined_raw, "morph+img_mean_prob"),              "#17becf", "^", "-"),
        ("All3 (mean)",      _build_series(combined_raw, "met_nan+morph+img_mean_prob"),      "#d62728", "D", "-"),
        ("All3 (vote)",      _build_series(combined_raw, "met_nan+morph+img_majority_vote"),  "#ff7f0e", "P", "--"),
    ]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"wspace": 0.30},
    )

    _plot_panel(ax_left,  left_series,  days)
    _plot_panel(ax_right, right_series, days)

    ax_left.set_title(
        "Single Modality — 3 Met Variants\n(10×4-fold repeated CV, series_idor, n=139)",
        fontsize=11, fontweight="bold",
    )
    ax_right.set_title(
        "Late-Fusion Strategies\n(shared splits, series_idor, n=139)",
        fontsize=11, fontweight="bold",
    )

    fig.suptitle(
        "Balanced Accuracy by Day — Organoid Quality Classification",
        fontsize=13, fontweight="bold", y=1.02,
    )

    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _FIGURE_DIR / _OUT_NAME
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")

    repo_path = _REPO_FIG_DIR / _OUT_NAME
    _REPO_FIG_DIR.mkdir(exist_ok=True)
    shutil.copy(out_path, repo_path)
    print(f"Copied to {repo_path}")

    # ── Table ─────────────────────────────────────────────────────────────────
    _print_table(table_series, days)
    _save_table_figure(table_series, days)


if __name__ == "__main__":
    main()

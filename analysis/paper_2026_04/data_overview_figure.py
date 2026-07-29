#!/usr/bin/env python3
"""Study design, longitudinal data, and prediction framework.

Panels:
  A  Representative Acceptable / Not Acceptable organoid images at 6 key timepoints.
  B  Day-30 expert voting scheme (left) and cohort definition (right).
  C  Three-modality prediction framework: independent predictions + combined model.

Outputs:
  figures/data_overview.png

Usage:
  make analysis-data-overview
  make analysis-data-overview ARGS="--acc-id 'BA1 96_1 A1'"
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from pipeline.data_loader import (
    FIGURE_DIR,
    OrganoidDataset,
    _load_idor_organoid_ids,
    iter_organoid_records,
    filters_for_mode,
    HIGH_QUALITY_BATCHES,
    MIN_VOTES,
    LABEL_DAY,
)
from pipeline.splits import Splits

warnings.filterwarnings("ignore")

ALL_DATA_PATH = "data/all_data.json"

# Six representative development timepoints shown in Panel A
KEY_DAYS       = ["Dy03", "Dy08", "Dy15", "Dy20_5", "Dy24", "Dy30"]
KEY_DAY_LABELS = ["Day 3", "Day 8", "Day 15", "Day 20.5", "Day 24", "Day 30"]

ACCEPTABLE_COLOR     = "#2196F3"
NOT_ACCEPTABLE_COLOR = "#F44336"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _raw_image_path(record: dict):
    ar = (record.get("images") or {}).get("aspect_ratio") or {}
    tif = ar.get("ar_raw_tif")
    if tif and Path(tif).exists():
        return Path(tif)
    img = (record.get("images") or {}).get("img_path")
    if img and Path(img).exists():
        return Path(img)
    return None


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.dtype == np.uint16 or str(img.mode).startswith("I"):
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)
        arr = (arr * 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _um_per_px(record: dict):
    ar = (record.get("images") or {}).get("aspect_ratio") or {}
    return ar.get("ar_orig_um_per_px")


def _add_scale_bar(ax, img_h: int, img_w: int, um_per_px: float, scale_um: int = 500):
    bar_px = scale_um / um_per_px
    margin_x = img_w * 0.04
    margin_y = img_h * 0.07
    x1 = img_w - margin_x - bar_px
    x2 = img_w - margin_x
    ax.plot([x1, x2], [img_h - margin_y] * 2,
            color="white", linewidth=2.0, solid_capstyle="butt", zorder=5)


def _stick_figure(ax, cx, cy, scale=0.30, vote_color=None):
    c = "#444"
    r = scale * 0.22
    ax.add_patch(mpatches.Circle((cx, cy + scale * 0.75), r, color=c, zorder=4))
    ax.plot([cx, cx], [cy + scale * 0.53, cy - scale * 0.10],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    ax.plot([cx - scale * 0.45, cx, cx + scale * 0.30],
            [cy + scale * 0.14, cy + scale * 0.28, cy + scale * 0.04],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    ax.plot([cx, cx - scale * 0.28], [cy - scale * 0.10, cy - scale * 0.60],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    ax.plot([cx, cx + scale * 0.28], [cy - scale * 0.10, cy - scale * 0.60],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    if vote_color is not None:
        ax.add_patch(mpatches.Circle((cx, cy + scale * 1.22), r * 0.80,
                                     color=vote_color, zorder=5))


def _find_example_organoids(ds, acc_id=None, nacc_id=None):
    acc = nacc = None
    for org_id, info in ds.iter_organoids():
        if not all(d in info["records"] for d in KEY_DAYS):
            continue
        if not all(_raw_image_path(info["records"][d]) is not None for d in KEY_DAYS):
            continue
        if info["label"] == "Acceptable" and acc is None:
            if acc_id is None or org_id == acc_id:
                acc = (org_id, info)
        elif info["label"] == "Not Acceptable" and nacc is None:
            if nacc_id is None or org_id == nacc_id:
                nacc = (org_id, info)
        if acc and nacc:
            break
    if acc is None or nacc is None:
        raise RuntimeError("Could not find example organoids with complete key-day image series.")
    return acc, nacc


def _get_cohort_counts(all_data_path: str) -> dict:
    """Return full-consensus and strong-consensus organoid counts from Day-30 survey."""
    _col1, col2_pairs = _load_idor_organoid_ids()
    col2 = {oid for oid, _ in col2_pairs}
    orgs = {
        oid: recs
        for oid, recs, _batch in iter_organoid_records(all_data_path, batches=HIGH_QUALITY_BATCHES)
    }
    full_acc = full_nacc = no_consensus = 0
    strong_acc = strong_nacc = 0
    for oid in col2:
        rec = (orgs.get(oid) or {}).get(LABEL_DAY)
        if rec is None:
            continue
        reg = (rec.get("label") or {}).get("regular_votes", {})
        n_a, n_n = reg.get("Acceptable", 0), reg.get("Not Acceptable", 0)
        if n_a + n_n == 0:
            continue
        if n_a >= MIN_VOTES:
            full_acc += 1
            if n_n == 0:
                strong_acc += 1
        elif n_n >= MIN_VOTES:
            full_nacc += 1
            if n_a == 0:
                strong_nacc += 1
        else:
            no_consensus += 1
    return {
        "total":  full_acc + full_nacc + no_consensus,
        "full":   (full_acc, full_nacc),
        "strong": (strong_acc, strong_nacc),
        "no_consensus": no_consensus,
    }


# ---------------------------------------------------------------------------
# Panel A — image strips
# ---------------------------------------------------------------------------

def _draw_image_strips(fig, outer_gs, acc_info, nacc_info, acc_id, nacc_id):
    """Two rows × six key timepoints; white 500 µm scale bar; day labels top row."""
    inner = gridspec.GridSpecFromSubplotSpec(
        2, len(KEY_DAYS), subplot_spec=outer_gs, wspace=0.03, hspace=0.05
    )
    row_configs = [
        (0, acc_info,  acc_id,  ACCEPTABLE_COLOR,     "Acceptable"),
        (1, nacc_info, nacc_id, NOT_ACCEPTABLE_COLOR, "Not Acceptable"),
    ]
    for row_i, info, org_id, color, label_str in row_configs:
        for col_i, (day, day_label) in enumerate(zip(KEY_DAYS, KEY_DAY_LABELS)):
            ax = fig.add_subplot(inner[row_i, col_i])
            img = _load_image(_raw_image_path(info["records"][day]))
            ax.imshow(img)
            um_px = _um_per_px(info["records"][day])
            if um_px:
                _add_scale_bar(ax, img.shape[0], img.shape[1], um_px, scale_um=500)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if col_i == 0:
                ax.set_ylabel(label_str, color=color, fontsize=8,
                              fontweight="bold", rotation=90, labelpad=4)
            if row_i == 0:
                ax.set_title(day_label, fontsize=7, pad=2)


# ---------------------------------------------------------------------------
# Panel B — voting scheme + cohort definition
# ---------------------------------------------------------------------------

def _draw_voting_scheme(ax, dy30_rec):
    """Left sub-panel of B: expert voting schematic (generic, one-row)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    mw, mh, mcx, mcy = 1.25, 1.00, 0.82, 1.55

    # Monitor
    ax.add_patch(FancyBboxPatch(
        (mcx - mw / 2, mcy - mh / 2), mw, mh,
        boxstyle="round,pad=0.04", lw=2.0, edgecolor="#444", facecolor="#111", zorder=2,
    ))
    path = _raw_image_path(dy30_rec)
    if path:
        mg = 0.05
        ax.imshow(_load_image(path),
                  extent=[mcx - mw/2 + mg, mcx + mw/2 - mg,
                          mcy - mh/2 + mg, mcy + mh/2 - mg],
                  aspect="auto", zorder=3)
    ax.plot([mcx, mcx], [mcy - mh/2, mcy - mh/2 - 0.14], color="#555", lw=2.5, zorder=2)
    ax.plot([mcx - 0.17, mcx + 0.17], [mcy - mh/2 - 0.14] * 2, color="#555", lw=2.5, zorder=2)
    ax.text(mcx, mcy - mh/2 - 0.30, "Day 30\nimage only",
            ha="center", va="top", fontsize=6, color="#555")

    # Gaze lines and 5 stick figures (4 Acc + 1 NotAcc as illustration)
    fig_xs = [2.18 + i * 0.82 for i in range(5)]
    eye_dy = 0.30 * 0.75
    vote_colors = [ACCEPTABLE_COLOR] * 4 + [NOT_ACCEPTABLE_COLOR]
    for fx in fig_xs:
        ax.plot([mcx + mw/2 + 0.03, fx - 0.06], [mcy + eye_dy] * 2,
                color="#ddd", lw=0.7, linestyle="--", zorder=1)
    for fx, vc in zip(fig_xs, vote_colors):
        _stick_figure(ax, fx, mcy, scale=0.30, vote_color=vc)

    # Arrow + consensus box
    ax.annotate("", xy=(6.75, mcy), xytext=(6.22, mcy),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#777"))
    ax.add_patch(FancyBboxPatch(
        (6.80, mcy - 0.33), 2.95, 0.62,
        boxstyle="round,pad=0.08", lw=1.5,
        edgecolor=ACCEPTABLE_COLOR, facecolor=ACCEPTABLE_COLOR + "22", zorder=2,
    ))
    ax.text(8.28, mcy, "Acceptable", ha="center", va="center",
            fontsize=8, fontweight="bold", color=ACCEPTABLE_COLOR)

    # Legend + rule
    ax.add_patch(mpatches.Circle((0.22, 0.42), 0.09, color=ACCEPTABLE_COLOR, zorder=4))
    ax.text(0.37, 0.42, "Acc vote", va="center", fontsize=6, color=ACCEPTABLE_COLOR)
    ax.add_patch(mpatches.Circle((1.80, 0.42), 0.09, color=NOT_ACCEPTABLE_COLOR, zorder=4))
    ax.text(1.95, 0.42, "Not Acc vote", va="center", fontsize=6, color=NOT_ACCEPTABLE_COLOR)
    ax.text(5.8, 0.42, "Consensus: ≥ 4 / 5 votes",
            ha="center", va="center", fontsize=6.5, color="#555", style="italic")

    ax.text(5.0, 2.90,
            "5 expert evaluators — Day 30 image only (no metabolite data)",
            ha="center", va="top", fontsize=7.5, color="#333", fontweight="bold")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)


def _draw_cohort_def(ax, cohort_counts: dict):
    """Right sub-panel of B: cohort sizes for full and strong-consensus cohorts."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    total = cohort_counts["total"]
    full_acc, full_nacc = cohort_counts["full"]
    strong_acc, strong_nacc = cohort_counts["strong"]
    nc = cohort_counts["no_consensus"]

    def _cohort_box(cx, cy, w, h, title, n_acc, n_nacc):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.10", lw=1.4, edgecolor="#aaa", facecolor="#F8F8F8", zorder=2,
        ))
        ax.text(cx, cy + h * 0.27, title,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333")
        ax.text(cx - 1.3, cy - h * 0.12, f"Acceptable: {n_acc}",
                ha="center", va="center", fontsize=7.5, color=ACCEPTABLE_COLOR, fontweight="bold")
        ax.text(cx + 1.5, cy - h * 0.12, f"Not Acc: {n_nacc}",
                ha="center", va="center", fontsize=7.5, color=NOT_ACCEPTABLE_COLOR, fontweight="bold")
        ax.text(cx, cy - h * 0.40, f"N = {n_acc + n_nacc}",
                ha="center", va="center", fontsize=7, color="#666")

    def _arrow(y1, y2):
        ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#888"))

    # Top: total surveyed
    ax.text(5.0, 2.88, f"All Day-30 surveyed organoids:  N = {total}",
            ha="center", va="top", fontsize=8, fontweight="bold", color="#333")

    _arrow(2.60, 2.30)
    ax.text(5.5, 2.46, f"≥ 4 / 5 votes", ha="left", va="center",
            fontsize=6.5, color="#777", style="italic")
    ax.text(5.5, 2.34, f"({nc} organoids excluded, 3–2 split)",
            ha="left", va="center", fontsize=6, color="#999")

    # Full-consensus cohort box
    _cohort_box(5.0, 1.82, 9.2, 0.82,
                f"Full cohort  (≥ 4 / 5 votes)", full_acc, full_nacc)

    _arrow(1.41, 1.12)
    ax.text(5.5, 1.28, "5 / 5 votes agree", ha="left", va="center",
            fontsize=6.5, color="#777", style="italic")

    # Strong-consensus cohort box
    _cohort_box(5.0, 0.68, 9.2, 0.74,
                "Strong-consensus cohort  (5 / 5 votes)", strong_acc, strong_nacc)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)


# ---------------------------------------------------------------------------
# Panel C — simplified prediction framework
# ---------------------------------------------------------------------------

def _draw_framework(ax):
    """Panel C: three-modality framework with independent + combined predictions."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("C  Prediction Framework", fontsize=9, fontweight="bold", loc="left")

    def _box(x, y, w, h, label, color, fontsize=8):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08", lw=1.5,
            edgecolor=color, facecolor=color + "22",
        ))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)

    def _arrow(x1, y1, x2, y2, **kw):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#555", **kw))

    xs     = [1.5, 5.0, 8.5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # --- Input modality boxes ---
    _box(xs[0], 2.58, 2.0, 0.66, "Images",      colors[0])
    _box(xs[1], 2.58, 2.2, 0.66, "Metabolites", colors[1])
    _box(xs[2], 2.58, 2.0, 0.66, "Morphology",  colors[2])

    # --- Independent prediction path ---
    for x, c in zip(xs, colors):
        _arrow(x, 2.25, x, 1.77)
        ax.add_patch(FancyBboxPatch(
            (x - 0.92, 1.45), 1.84, 0.30,
            boxstyle="round,pad=0.05", lw=1.2, edgecolor="#555", facecolor="#EEE",
        ))
        ax.text(x, 1.60, "Prediction", ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="#333")

    # --- Divider ---
    ax.plot([0.45, 9.55], [1.32, 1.32], color="#ccc", lw=0.8, linestyle="--")
    ax.text(0.20, 2.10, "Indep.", ha="center", va="center",
            fontsize=5.5, color="#bbb", style="italic")
    ax.text(0.20, 0.75, "Combined", ha="center", va="center",
            fontsize=5.5, color="#bbb", style="italic")

    # --- Curved arrows: inputs → combined box ---
    ax.annotate("", xy=(4.42, 1.07), xytext=(xs[0], 2.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888",
                               connectionstyle="arc3,rad=0.18"))
    ax.annotate("", xy=(5.00, 1.07), xytext=(xs[1], 2.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888"))
    ax.annotate("", xy=(5.58, 1.07), xytext=(xs[2], 2.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888",
                               connectionstyle="arc3,rad=-0.18"))

    # --- Combined model box ---
    ax.add_patch(FancyBboxPatch(
        (2.55, 0.67), 4.90, 0.38,
        boxstyle="round,pad=0.08", lw=1.5,
        edgecolor="#555555", facecolor="#55555522",
    ))
    ax.text(5.0, 0.86, "Combined Model", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#555")

    # --- Arrow to shared output ---
    _arrow(5.0, 0.67, 5.0, 0.42)
    ax.text(5.0, 0.22, "Acceptable  /  Not Acceptable",
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#EEEEEE",
                      edgecolor="#444444", lw=1.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc-id",  default=None, help="Acceptable example organoid ID")
    parser.add_argument("--nacc-id", default=None, help="Not Acceptable example organoid ID")
    args = parser.parse_args()

    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode("base"))
    (acc_id, acc_info), (nacc_id, nacc_info) = _find_example_organoids(
        ds, acc_id=args.acc_id, nacc_id=args.nacc_id
    )
    print(f"Acceptable example:     {acc_id}")
    print(f"Not Acceptable example: {nacc_id}")

    cohort_counts = _get_cohort_counts(ALL_DATA_PATH)
    print(f"Cohort counts: {cohort_counts}")

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        "Study design, longitudinal data, and prediction framework",
        fontsize=13, fontweight="bold", y=0.99,
    )

    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.42,
                              height_ratios=[2.6, 1.8, 1.6])

    # --- Panel A ---
    ax_a_label = fig.add_subplot(outer[0])
    ax_a_label.axis("off")
    ax_a_label.text(0.01, 0.98, "A  Organoid images across development",
                    transform=ax_a_label.transAxes,
                    fontsize=9, fontweight="bold", va="top")
    ax_a_label.text(0.99, 0.01, "Scale bar: 500 µm",
                    transform=ax_a_label.transAxes,
                    fontsize=7, va="bottom", ha="right", color="#555")
    _draw_image_strips(fig, outer[0], acc_info, nacc_info, acc_id, nacc_id)

    # --- Panel B ---
    row_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                             wspace=0.25, width_ratios=[1.2, 1])
    ax_b1 = fig.add_subplot(row_b[0])
    ax_b2 = fig.add_subplot(row_b[1])
    ax_b1.text(0.01, 0.97, "B  Expert voting and cohort definition",
               transform=ax_b1.transAxes, fontsize=9, fontweight="bold", va="top")

    acc_dy30 = acc_info["records"].get("Dy30", {})
    _draw_voting_scheme(ax_b1, acc_dy30)
    _draw_cohort_def(ax_b2, cohort_counts)

    # --- Panel C ---
    ax_c = fig.add_subplot(outer[2])
    _draw_framework(ax_c)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "data_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Study design, longitudinal data, and prediction framework.

Panels:
  A  Acceptable / Not Acceptable organoid images across all 11 timepoints.
  B  Day-30 expert voting (two rows: Acceptable and Not Acceptable examples).
  C  Cohort definition — full consensus and strong-consensus counts.
  D  Metabolite concentration profiles over development.
  E  Morphology features with segmentation example.
  F  Three-modality prediction framework (independent + combined model).

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
    DAY_ORDER,
    OrganoidDataset,
    _load_idor_organoid_ids,
    get_edge_fraction,
    get_mask_area_um2,
    iter_organoid_records,
    filters_for_mode,
    HIGH_QUALITY_BATCHES,
    MIN_VOTES,
    LABEL_DAY,
    REQUIRED_METABOLITES,
)
from pipeline.splits import Splits

warnings.filterwarnings("ignore")

ALL_DATA_PATH = "data/all_data.json"

ACCEPTABLE_COLOR     = "#2196F3"
NOT_ACCEPTABLE_COLOR = "#F44336"

DAY_LABELS = [d.replace("Dy", "Day ").replace("_5", ".5") for d in DAY_ORDER]

# Metabolite display
METABOLITE_COLORS = {
    "GlucoseGlo":   "#1f77b4",
    "GlutamateGlo": "#ff7f0e",
    "LactateGlo":   "#2ca02c",
    "PyruvateGlo":  "#d62728",
    "BCAAGlo":      "#9467bd",
    "MalateGlo":    "#8c564b",
}
METABOLITE_SHORT = {
    "GlucoseGlo":   "Glucose",
    "GlutamateGlo": "Glutamate",
    "LactateGlo":   "Lactate",
    "PyruvateGlo":  "Pyruvate",
    "BCAAGlo":      "BCAA",
    "MalateGlo":    "Malate",
}

# Morphology display
MORPH_CSV_PATH  = "data/normalized/CONC_data_organoides_residualized_final.csv"
MORPH_SHAPE_COLS = ["Circ._win", "AR_win", "Solidity_win", "Complexity_win"]
MORPH_COLORS = {
    "mask_area_um2": "#2ca02c",
    "edge_fraction": "#17becf",
    "Circ._win":     "#9467bd",
    "AR_win":        "#e377c2",
    "Solidity_win":  "#bcbd22",
    "Complexity_win":"#7f7f7f",
}
MORPH_SHORT = {
    "mask_area_um2": "Area (mm²)",
    "edge_fraction": "Edge frac.",
    "Circ._win":     "Circularity",
    "AR_win":        "Aspect ratio",
    "Solidity_win":  "Solidity",
    "Complexity_win":"Complexity",
}
_DAY_STR_TO_INT = {
    "Dy03": 3, "Dy06": 6, "Dy08": 8, "Dy10": 10, "Dy13": 13,
    "Dy15": 15, "Dy17": 17, "Dy20_5": 21, "Dy24": 24, "Dy28": 28, "Dy30": 30,
}
_MORPH_CSV_CACHE = None


# ---------------------------------------------------------------------------
# Image utilities
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
    ax.plot([img_w - margin_x - bar_px, img_w - margin_x],
            [img_h - margin_y, img_h - margin_y],
            color="white", linewidth=2.0, solid_capstyle="butt", zorder=5)


def _seg_overlay(record: dict, hex_color: str) -> np.ndarray:
    """Return the 575×575 cm_source image with segmentation mask tinted in hex_color."""
    cm = (record.get("images") or {}).get("clipped_meanfill") or {}
    img_path  = cm.get("cm_source_image_abs")
    mask_path = cm.get("cm_source_mask_abs")
    if not img_path or not Path(img_path).exists():
        return None
    img = np.array(Image.open(img_path)).astype(float)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img / 255.0
    if mask_path and Path(mask_path).exists():
        mask = np.array(Image.open(mask_path)) > 0
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        alpha = 0.40
        img[mask, 0] = img[mask, 0] * (1 - alpha) + r * alpha
        img[mask, 1] = img[mask, 1] * (1 - alpha) + g * alpha
        img[mask, 2] = img[mask, 2] * (1 - alpha) + b * alpha
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Schematic utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _find_example_organoids(ds, acc_id=None, nacc_id=None):
    acc = nacc = None
    for org_id, info in ds.iter_organoids():
        if not all(d in info["records"] for d in DAY_ORDER):
            continue
        if not all(_raw_image_path(info["records"][d]) is not None for d in DAY_ORDER):
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
        raise RuntimeError("Could not find example organoids with complete 11-day image series.")
    return acc, nacc


def _get_cohort_counts(all_data_path: str) -> dict:
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
        "total":        full_acc + full_nacc + no_consensus,
        "full":         (full_acc, full_nacc),
        "strong":       (strong_acc, strong_nacc),
        "no_consensus": no_consensus,
    }


def _get_metabolites(info: dict) -> dict:
    result = {m: [] for m in REQUIRED_METABOLITES}
    for day in DAY_ORDER:
        rec  = info["records"].get(day, {})
        mets = rec.get("metabolite", {})
        for m in REQUIRED_METABOLITES:
            val = (mets.get(m) or {}).get("concentration_uM")
            result[m].append(val)
    return result


def _get_morphology(info: dict):
    areas, edges = [], []
    for day in DAY_ORDER:
        rec = info["records"].get(day, {})
        areas.append(get_mask_area_um2(rec))
        edges.append(get_edge_fraction(rec))
    return areas, edges


def _load_morph_csv():
    global _MORPH_CSV_CACHE
    if _MORPH_CSV_CACHE is None:
        import pandas as pd
        p = Path(MORPH_CSV_PATH)
        _MORPH_CSV_CACHE = pd.read_csv(p) if p.exists() else __import__("pandas").DataFrame()
    return _MORPH_CSV_CACHE


def _get_morph_csv_features(org_id: str, cols: list) -> dict:
    df  = _load_morph_csv()
    if df.empty:
        return {c: [None] * len(DAY_ORDER) for c in cols}
    sub = df[df["Organoid"] == org_id.replace(" ", "_")].set_index("Day")
    result = {}
    for c in cols:
        vals = []
        for day_str in DAY_ORDER:
            day_int = _DAY_STR_TO_INT[day_str]
            if day_int in sub.index and c in sub.columns:
                v = sub.loc[day_int, c]
                vals.append(float(v) if v == v else None)
            else:
                vals.append(None)
        result[c] = vals
    return result


# ---------------------------------------------------------------------------
# Panel A — image strips (all 11 days)
# ---------------------------------------------------------------------------

def _draw_image_strips(fig, outer_gs, acc_info, nacc_info, acc_id, nacc_id):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, len(DAY_ORDER), subplot_spec=outer_gs, wspace=0.03, hspace=0.05,
    )
    row_configs = [
        (0, acc_info,  acc_id,  ACCEPTABLE_COLOR,     "Acceptable"),
        (1, nacc_info, nacc_id, NOT_ACCEPTABLE_COLOR, "Not Acceptable"),
    ]
    for row_i, info, org_id, color, label_str in row_configs:
        for col_i, (day, day_label) in enumerate(zip(DAY_ORDER, DAY_LABELS)):
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
                ax.set_title(day_label, fontsize=6.5, pad=2)


# ---------------------------------------------------------------------------
# Panel B — two-row voting scheme (Acceptable + Not Acceptable examples)
# ---------------------------------------------------------------------------

def _draw_voting_scheme(ax, acc_dy30_rec, nacc_dy30_rec):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    mw, mh = 1.20, 0.95
    mcx = 0.80
    fig_xs = [2.20 + i * 0.85 for i in range(5)]
    eye_dy = 0.30 * 0.75

    rows = [
        (acc_dy30_rec,  ACCEPTABLE_COLOR,    "Acceptable",    3.0),
        (nacc_dy30_rec, NOT_ACCEPTABLE_COLOR, "Not Acceptable", 1.4),
    ]

    for rec, color, label_str, yc in rows:
        # Monitor
        ax.add_patch(FancyBboxPatch(
            (mcx - mw / 2, yc - mh / 2), mw, mh,
            boxstyle="round,pad=0.04", lw=2.0, edgecolor="#444", facecolor="#111", zorder=2,
        ))
        path = _raw_image_path(rec)
        if path:
            mg = 0.05
            ax.imshow(_load_image(path),
                      extent=[mcx - mw/2 + mg, mcx + mw/2 - mg,
                               yc - mh/2 + mg, yc + mh/2 - mg],
                      aspect="auto", zorder=3)
        ax.plot([mcx, mcx], [yc - mh/2, yc - mh/2 - 0.14], color="#555", lw=2.5, zorder=2)
        ax.plot([mcx - 0.18, mcx + 0.18], [yc - mh/2 - 0.14] * 2, color="#555", lw=2.5, zorder=2)
        ax.text(mcx, yc - mh/2 - 0.28, "Day 30",
                ha="center", va="top", fontsize=6, color="#555")

        # Gaze lines
        for fx in fig_xs:
            ax.plot([mcx + mw/2 + 0.03, fx - 0.06], [yc + eye_dy] * 2,
                    color="#ccc", lw=0.7, linestyle="--", zorder=1, alpha=0.8)

        # Stick figures with vote dots
        votes = (rec.get("label") or {}).get("regular_votes", {})
        n_a = votes.get("Acceptable", 0)
        n_n = votes.get("Not Acceptable", 0)
        vote_colors = [ACCEPTABLE_COLOR] * n_a + [NOT_ACCEPTABLE_COLOR] * n_n
        for fx, vc in zip(fig_xs, vote_colors):
            _stick_figure(ax, fx, yc, scale=0.30, vote_color=vc)

        # Arrow + consensus box
        ax.annotate("", xy=(6.80, yc), xytext=(6.22, yc),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#777"))
        ax.add_patch(FancyBboxPatch(
            (6.85, yc - 0.35), 2.90, 0.65,
            boxstyle="round,pad=0.08", lw=1.5,
            edgecolor=color, facecolor=color + "22", zorder=2,
        ))
        ax.text(8.30, yc, label_str, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)

    # Legend + rule
    ax.add_patch(mpatches.Circle((0.22, 0.52), 0.09, color=ACCEPTABLE_COLOR, zorder=4))
    ax.text(0.38, 0.52, "Acc vote", va="center", fontsize=6.5, color=ACCEPTABLE_COLOR)
    ax.add_patch(mpatches.Circle((2.30, 0.52), 0.09, color=NOT_ACCEPTABLE_COLOR, zorder=4))
    ax.text(2.46, 0.52, "Not Acc vote", va="center", fontsize=6.5, color=NOT_ACCEPTABLE_COLOR)
    ax.text(7.80, 0.52, "≥ 4 / 5 votes → consensus",
            ha="center", va="center", fontsize=6.5, color="#555", style="italic")

    ax.text(5.0, 3.90,
            "5 expert evaluators — Day 30 image only (no metabolite data)",
            ha="center", va="top", fontsize=7.5, color="#333", fontweight="bold")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)


# ---------------------------------------------------------------------------
# Panel C — cohort definition
# ---------------------------------------------------------------------------

def _draw_cohort_def(ax, cohort_counts: dict):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    total = cohort_counts["total"]
    full_acc, full_nacc = cohort_counts["full"]
    strong_acc, strong_nacc = cohort_counts["strong"]
    nc = cohort_counts["no_consensus"]

    def _box(cy, h, title, n_acc, n_nacc):
        w = 9.2
        ax.add_patch(FancyBboxPatch(
            (5.0 - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.10", lw=1.4, edgecolor="#aaa", facecolor="#F8F8F8", zorder=2,
        ))
        ax.text(5.0, cy + h * 0.26, title, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#333")
        ax.text(5.0 - 1.5, cy - h * 0.15, f"Acceptable: {n_acc}",
                ha="center", va="center", fontsize=7.5, color=ACCEPTABLE_COLOR, fontweight="bold")
        ax.text(5.0 + 1.6, cy - h * 0.15, f"Not Acc: {n_nacc}",
                ha="center", va="center", fontsize=7.5, color=NOT_ACCEPTABLE_COLOR, fontweight="bold")
        ax.text(5.0, cy - h * 0.42, f"N = {n_acc + n_nacc}",
                ha="center", va="center", fontsize=7, color="#666")

    def _arr(y1, y2, note=""):
        ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#888"))
        if note:
            ax.text(5.5, (y1 + y2) / 2, note, ha="left", va="center",
                    fontsize=6.5, color="#777", style="italic")

    ax.text(5.0, 3.90, f"All Day-30 surveyed:  N = {total}",
            ha="center", va="top", fontsize=8, fontweight="bold", color="#333")

    _arr(3.60, 3.22, f"≥ 4/5 votes  ({nc} excluded, 3–2 split)")
    _box(2.72, 0.82, "Full cohort  (≥ 4 / 5 votes)", full_acc, full_nacc)
    _arr(2.31, 1.94, "5 / 5 votes agree")
    _box(1.52, 0.74, "Strong-consensus cohort  (5 / 5 votes)", strong_acc, strong_nacc)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)


# ---------------------------------------------------------------------------
# Panel D — metabolite profiles
# ---------------------------------------------------------------------------

def _draw_metabolites(ax, acc_info, nacc_info):
    x = range(len(DAY_ORDER))
    for met in REQUIRED_METABOLITES:
        color = METABOLITE_COLORS[met]
        short = METABOLITE_SHORT[met]
        acc_vals  = _get_metabolites(acc_info)[met]
        nacc_vals = _get_metabolites(nacc_info)[met]
        ax.plot(x, acc_vals,  color=color, lw=1.5, marker="o", ms=3, label=short)
        ax.plot(x, nacc_vals, color=color, lw=1.5, linestyle="--", marker="s", ms=3, alpha=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([d.replace("Dy", "D").replace("_5", ".5") for d in DAY_ORDER],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Concentration (µM)", fontsize=8)
    ax.set_title("D  Metabolite Profiles", fontsize=9, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D
    feat_handles = [Line2D([0], [0], color=METABOLITE_COLORS[m], lw=1.5, label=METABOLITE_SHORT[m])
                    for m in REQUIRED_METABOLITES]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, label="Acceptable"),
        Line2D([0], [0], color="gray", lw=1.5, linestyle="--", label="Not Acceptable"),
    ]
    ax.legend(handles=feat_handles + style_handles, fontsize=5.5, ncol=2,
              loc="upper left", framealpha=0.8)


# ---------------------------------------------------------------------------
# Panel E — morphology features + segmentation example
# ---------------------------------------------------------------------------

def _draw_morphology(fig, outer_gs, acc_info, nacc_info, acc_id, nacc_id):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs, wspace=0.30, width_ratios=[1, 2.8],
    )

    # Left: segmentation overlays at Dy30
    seg_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=inner[0], hspace=0.06)
    for i, (info, color, label) in enumerate([
        (acc_info,  ACCEPTABLE_COLOR,    "Acceptable"),
        (nacc_info, NOT_ACCEPTABLE_COLOR, "Not Acceptable"),
    ]):
        ax_seg = fig.add_subplot(seg_gs[i])
        overlay = _seg_overlay(info["records"].get("Dy30", {}), color)
        if overlay is not None:
            ax_seg.imshow(overlay)
        ax_seg.set_xticks([])
        ax_seg.set_yticks([])
        for spine in ax_seg.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
        ax_seg.set_ylabel(label, color=color, fontsize=7, fontweight="bold",
                          rotation=90, labelpad=3)
        if i == 0:
            ax_seg.set_title("Day 30\nsegmentation", fontsize=6.5, pad=2)

    # Right: dual-axis morphology trajectory
    ax = fig.add_subplot(inner[1])
    ax2 = ax.twinx()

    x = list(range(len(DAY_ORDER)))
    day_labels = [d.replace("Dy", "D").replace("_5", ".5") for d in DAY_ORDER]

    acc_areas,  acc_edges  = _get_morphology(acc_info)
    nacc_areas, nacc_edges = _get_morphology(nacc_info)
    acc_shape  = _get_morph_csv_features(acc_id,  MORPH_SHAPE_COLS)
    nacc_shape = _get_morph_csv_features(nacc_id, MORPH_SHAPE_COLS)

    c_area = MORPH_COLORS["mask_area_um2"]
    ax.plot(x, [a / 1e6 if a is not None else None for a in acc_areas],
            color=c_area, lw=2, marker="o", ms=4, label=MORPH_SHORT["mask_area_um2"])
    ax.plot(x, [a / 1e6 if a is not None else None for a in nacc_areas],
            color=c_area, lw=2, marker="s", ms=4, linestyle="--", alpha=0.7)
    ax.set_ylabel("Area (mm²)", fontsize=8, color=c_area)
    ax.tick_params(axis="y", labelcolor=c_area, labelsize=7)

    for feat, acc_vals, nacc_vals in [
        ("edge_fraction", acc_edges, nacc_edges),
    ] + [(col, acc_shape[col], nacc_shape[col]) for col in MORPH_SHAPE_COLS]:
        c = MORPH_COLORS[feat]
        ax2.plot(x, acc_vals,  color=c, lw=1.5, marker="o", ms=3, label=MORPH_SHORT[feat])
        ax2.plot(x, nacc_vals, color=c, lw=1.5, marker="s", ms=3, linestyle="--", alpha=0.7)

    ax2.set_ylabel("Shape descriptors", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_ylim(-0.05, 1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("E  Morphology Features", fontsize=9, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D
    feat_handles = [
        Line2D([0], [0], color=MORPH_COLORS["mask_area_um2"], lw=2,
               label=MORPH_SHORT["mask_area_um2"]),
    ] + [Line2D([0], [0], color=MORPH_COLORS[f], lw=1.5, label=MORPH_SHORT[f])
         for f in ["edge_fraction"] + MORPH_SHAPE_COLS]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, label="Acceptable"),
        Line2D([0], [0], color="gray", lw=1.5, linestyle="--", label="Not Acceptable"),
    ]
    ax.legend(handles=feat_handles + style_handles, fontsize=5.5, ncol=2,
              loc="upper left", framealpha=0.8)


# ---------------------------------------------------------------------------
# Panel F — prediction framework (portrait-oriented, no model names)
# ---------------------------------------------------------------------------

def _draw_framework(ax):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("F  Prediction Framework", fontsize=9, fontweight="bold", loc="left")

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

    xs     = [1.3, 4.0, 6.7]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Modality input boxes
    _box(xs[0], 4.40, 2.0, 0.70, "Images",      colors[0])
    _box(xs[1], 4.40, 2.2, 0.70, "Metabolites", colors[1])
    _box(xs[2], 4.40, 2.0, 0.70, "Morphology",  colors[2])

    # Arrows: inputs → independent prediction boxes
    for x, c in zip(xs, colors):
        _arrow(x, 4.05, x, 3.60)
        ax.add_patch(FancyBboxPatch(
            (x - 0.90, 3.25), 1.80, 0.32,
            boxstyle="round,pad=0.05", lw=1.2, edgecolor="#555", facecolor="#EEE",
        ))
        ax.text(x, 3.41, "Prediction", ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="#333")

    # Section labels + dashed divider
    ax.plot([0.3, 7.7], [3.10, 3.10], color="#ccc", lw=0.8, linestyle="--")
    ax.text(0.18, 3.80, "Indep.", ha="center", va="center",
            fontsize=5.5, color="#bbb", style="italic")
    ax.text(0.18, 1.60, "Combined", ha="center", va="center",
            fontsize=5.5, color="#bbb", style="italic")

    # Curved arrows: inputs → combined box
    ax.annotate("", xy=(3.50, 2.40), xytext=(xs[0], 4.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888",
                               connectionstyle="arc3,rad=0.20"))
    ax.annotate("", xy=(4.00, 2.40), xytext=(xs[1], 4.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888"))
    ax.annotate("", xy=(4.50, 2.40), xytext=(xs[2], 4.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888",
                               connectionstyle="arc3,rad=-0.20"))

    # Combined model box
    ax.add_patch(FancyBboxPatch(
        (1.50, 1.96), 5.00, 0.42,
        boxstyle="round,pad=0.08", lw=1.5,
        edgecolor="#555555", facecolor="#55555522",
    ))
    ax.text(4.0, 2.17, "Combined Model", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#555")

    # Arrow to output
    _arrow(4.0, 1.96, 4.0, 1.55)

    # Output label
    ax.text(4.0, 1.28, "Acceptable  /  Not Acceptable",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#EEEEEE",
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
        ds, acc_id=args.acc_id, nacc_id=args.nacc_id,
    )
    print(f"Acceptable example:     {acc_id}")
    print(f"Not Acceptable example: {nacc_id}")

    cohort_counts = _get_cohort_counts(ALL_DATA_PATH)
    print(f"Cohort counts: {cohort_counts}")

    # -----------------------------------------------------------------------
    # Layout: 3 rows
    #   Row 0: A (image strips, full width)
    #   Row 1: B (voting) | C (cohort) | F (framework)
    #   Row 2: D (metabolites) | E (morphology + seg)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 17))
    fig.suptitle(
        "Study design, longitudinal data, and prediction framework",
        fontsize=13, fontweight="bold", y=0.99,
    )

    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.38,
                              height_ratios=[2.4, 2.6, 2.2])

    # Row 0: Panel A
    ax_a_lbl = fig.add_subplot(outer[0])
    ax_a_lbl.axis("off")
    ax_a_lbl.text(0.01, 0.98, "A  Organoid images across development (11 timepoints)",
                  transform=ax_a_lbl.transAxes, fontsize=9, fontweight="bold", va="top")
    ax_a_lbl.text(0.99, 0.01, "Scale bar: 500 µm",
                  transform=ax_a_lbl.transAxes, fontsize=7, va="bottom", ha="right", color="#555")
    _draw_image_strips(fig, outer[0], acc_info, nacc_info, acc_id, nacc_id)

    # Row 1: Panels B, C, F
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
                                            wspace=0.28, width_ratios=[2.8, 1.6, 1.6])
    ax_b = fig.add_subplot(row1[0])
    ax_c = fig.add_subplot(row1[1])
    ax_f = fig.add_subplot(row1[2])

    ax_b.text(0.01, 0.97, "B  Expert voting (Day 30)",
              transform=ax_b.transAxes, fontsize=9, fontweight="bold", va="top")
    ax_c.text(0.01, 0.97, "C  Cohort definition",
              transform=ax_c.transAxes, fontsize=9, fontweight="bold", va="top")

    acc_dy30  = acc_info["records"].get("Dy30", {})
    nacc_dy30 = nacc_info["records"].get("Dy30", {})
    _draw_voting_scheme(ax_b, acc_dy30, nacc_dy30)
    _draw_cohort_def(ax_c, cohort_counts)
    _draw_framework(ax_f)

    # Row 2: Panels D, E
    row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                            wspace=0.35, width_ratios=[1.1, 1])
    ax_d = fig.add_subplot(row2[0])
    _draw_metabolites(ax_d, acc_info, nacc_info)
    _draw_morphology(fig, row2[1], acc_info, nacc_info, acc_id, nacc_id)

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

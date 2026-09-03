#!/usr/bin/env python3
"""Generate PPT: 10-repeat 4-fold CV with 3 metabolite variants.

Slides:
  1.     Title
  2.     Study design
  3.     Aggregated two-panel figure
  4.     Summary table figure
  5.     All-repeats overlay (spaghetti) — key strategies
  6-15.  Per-repeat results (one slide per repeat, repeat 1-10)

Usage:
    conda run -n core_env python3 -m analysis.paper_2026_04.make_repeat_ppt
"""

import json
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

from pipeline.data_loader import DAY_ORDER, ANALYSIS_OUTPUT_DIR

COMBINED_PATH  = ANALYSIS_OUTPUT_DIR / "images" / "combined_results_kfold_series_idor_139.json"
TWO_PANEL_PATH = Path("figures/combined_kfold_two_panel_series_idor_139.png")
TABLE_PATH     = Path("figures/combined_kfold_table_series_idor_139.png")
OUT_PPT        = Path("figures/repeat_4fold_met_variants.pptx")

KEY_STYLE = {
    "met_nan":                    ("Met (nan floor)",  "#2ca02c", "o", "-",  2.0),
    "met_raw":                    ("Met (raw)",        "#98df8a", "v", "--", 1.5),
    "met_no_malate":              ("Met (no malate)",  "#17becf", "^", ":",  1.5),
    "morph":                      ("Morphology",       "#9467bd", "s", "-",  2.0),
    "img":                        ("Image",            "#1f77b4", "D", "-",  2.0),
    "met_nan+morph+img_mean_prob":("All3/nan (mean)",  "#d62728", "P", "-",  2.5),
}

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── helpers ───────────────────────────────────────────────────────────────────

def _hex(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _blank_layout(prs):
    return prs.slide_layouts[6]   # blank


def _title_layout(prs):
    return prs.slide_layouts[0]


def _add_textbox(slide, text, left, top, width, height,
                 fontsize=18, bold=False, color="#000000", align=PP_ALIGN.LEFT):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(fontsize)
    run.font.bold = bold
    run.font.color.rgb = _hex(color)
    return txb


def _fig_to_stream(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def _add_fig_stream(slide, buf, left, top, width):
    slide.shapes.add_picture(buf, left, top, width=width)


# ── figure generators ─────────────────────────────────────────────────────────

def _plot_repeat(combined, days, repeat_idx, title):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for k, (label, color, marker, ls, lw) in KEY_STYLE.items():
        xs, ys = [], []
        for i, day in enumerate(days):
            dr = combined.get(day, {})
            bas = dr.get(k, {}).get("repeat_balanced_accuracies", [])
            if repeat_idx < len(bas):
                xs.append(i); ys.append(bas[repeat_idx])
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                    linewidth=lw, markersize=6, label=label)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=45, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylabel("Balanced Accuracy (OOF)", fontsize=10)
    ax.set_xlabel("Day", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    return fig


CM_MODS = [
    ("met_nan",  "Met",   "#2ca02c"),
    ("morph",    "Morph", "#9467bd"),
    ("img",      "Image", "#1f77b4"),
]


def _plot_cm_row(combined, repeat_idx, day="Dy30"):
    """3 confusion matrices side by side for one repeat at one day."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, (k, label, color) in zip(axes, CM_MODS):
        cms = combined.get(day, {}).get(k, {}).get("repeat_confusion_matrices", [])
        if repeat_idx >= len(cms):
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label} ({day})", fontsize=10)
            ax.axis("off")
            continue
        cm = np.array(cms[repeat_idx])   # [[TN,FP],[FN,TP]]
        # Normalise by row (true class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                val_n = cm_norm[r, c]
                val_r = cm[r, c]
                txt_color = "white" if val_n > 0.6 else "black"
                ax.text(c, r, f"{val_r}\n({val_n:.0%})",
                        ha="center", va="center", fontsize=10,
                        color=txt_color, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Acc", "Pred NAcc"], fontsize=8)
        ax.set_yticklabels(["True Acc", "True NAcc"], fontsize=8)
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        ba = 0.5 * (tp / max(tp+fn, 1) + tn / max(tn+fp, 1))
        ax.set_title(f"{label} — BA={ba:.3f}", fontsize=10, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"OOF Confusion Matrices — {day}", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def _plot_spaghetti(combined, days):
    """All 10 repeats as thin lines + mean as thick line, for 4 key strategies."""
    show_keys = [
        ("met_nan",                    "Met/nan",       "#2ca02c"),
        ("morph",                      "Morphology",    "#9467bd"),
        ("img",                        "Image",         "#1f77b4"),
        ("met_nan+morph+img_mean_prob","All3/nan",      "#d62728"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (k, label, color) in zip(axes, show_keys):
        # Spaghetti: one line per repeat
        n_rep = 10
        for rep in range(n_rep):
            xs, ys = [], []
            for i, day in enumerate(days):
                bas = combined.get(day, {}).get(k, {}).get("repeat_balanced_accuracies", [])
                if rep < len(bas):
                    xs.append(i); ys.append(bas[rep])
            if xs:
                ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.35, linestyle="-")
        # Mean ± std
        xs_m, ys_m, lo, hi = [], [], [], []
        for i, day in enumerate(days):
            r = combined.get(day, {}).get(k, {})
            mn  = r.get("balanced_accuracy_mean")
            std = r.get("balanced_accuracy_std")
            if mn is not None:
                xs_m.append(i); ys_m.append(mn)
                lo.append(mn - (std or 0)); hi.append(mn + (std or 0))
        if xs_m:
            ax.plot(xs_m, ys_m, color=color, linewidth=2.5, marker="o",
                    markersize=5, label="mean")
            ax.fill_between(xs_m, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days, rotation=45, fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Balanced Accuracy (OOF)", fontsize=10)
    fig.suptitle(
        "10 Repeat × 4-Fold CV: Variance Across Repeats (thin=each repeat, thick=mean±SD)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig


def _plot_met_comparison(combined, days):
    """Side-by-side: 3 met variants standalone + their 3-way fusion."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    standalone = [
        ("met_nan",       "Met (nan floor)", "#2ca02c", "o", "-"),
        ("met_raw",       "Met (raw)",       "#98df8a", "v", "--"),
        ("met_no_malate", "Met (no malate)", "#17becf", "^", ":"),
    ]
    fusion = [
        ("met_nan+morph+img_mean_prob",       "All3/nan",       "#2ca02c", "D", "-"),
        ("met_raw+morph+img_mean_prob",       "All3/raw",       "#98df8a", "s", "--"),
        ("met_no_malate+morph+img_mean_prob", "All3/no-malate", "#17becf", "v", ":"),
        ("morph+img_mean_prob",               "Morph+Img",      "#9467bd", "^", "-"),
    ]
    for series, ax, title in [(standalone, ax1, "Met Variants — Standalone"),
                               (fusion,     ax2, "Met Variants — In Fusion (All3 mean prob)")]:
        for k, label, color, marker, ls in series:
            xs, ys, lo, hi = [], [], [], []
            for i, day in enumerate(days):
                r = combined.get(day, {}).get(k, {})
                mn  = r.get("balanced_accuracy_mean")
                std = r.get("balanced_accuracy_std", 0) or 0
                if mn is not None:
                    xs.append(i); ys.append(mn)
                    lo.append(mn - std); hi.append(mn + std)
            if xs:
                ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                        linewidth=2, markersize=6, label=label)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days, rotation=45, fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_ylabel("Balanced Accuracy (mean ± 1 SD)", fontsize=10)
        ax.set_xlabel("Day", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle(
        "Effect of Malate Treatment on Classification (10×4-fold CV, n=139)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ── PPT builder ────────────────────────────────────────────────────────────────

def build_ppt(combined, days):
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    M = Inches(0.35)   # margin

    # ── Slide 1: Title ────────────────────────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _hex("#1F3864")

    _add_textbox(slide,
        "10-Repeat 4-Fold CV\nMetabolite Variant Analysis",
        M, Inches(1.8), Inches(12.5), Inches(2.5),
        fontsize=40, bold=True, color="#FFFFFF", align=PP_ALIGN.CENTER)

    _add_textbox(slide,
        "series_idor cohort  ·  n = 139 organoids  ·  25 Not-Acceptable / 114 Acceptable",
        M, Inches(4.0), Inches(12.5), Inches(0.6),
        fontsize=18, color="#BDD7EE", align=PP_ALIGN.CENTER)

    _add_textbox(slide,
        "3 metabolite variants: nan-floor  ·  raw  ·  no-malate\n"
        "Modalities: Metabolite (LightGBM)  ·  Morphology (LightGBM)  ·  Image (EfficientNet-B0)\n"
        "Late fusion: mean probability  ·  majority vote",
        M, Inches(4.8), Inches(12.5), Inches(1.5),
        fontsize=14, color="#DDEBF7", align=PP_ALIGN.CENTER)

    # ── Slide 2: Study design ─────────────────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    _add_textbox(slide, "Experimental Design", M, M, Inches(12.5), Inches(0.55),
                 fontsize=26, bold=True, color="#1F3864")

    design_text = (
        "Cohort\n"
        "  • 139 organoids from series_idor (BA1 + BA2), including 6 stitched organoids\n"
        "  • Labels: 25 Not-Acceptable (NAcc), 114 Acceptable (Acc)\n"
        "  • 11 time points: Dy03 → Dy30\n"
        "\n"
        "Cross-Validation\n"
        "  • Stratified 4-fold CV  ×  10 repeats (different random seeds)\n"
        "  • Seed for repeat r: SEED + r × 1000  →  genuinely independent splits\n"
        "  • Same fold partition shared across all modalities within each repeat\n"
        "  • OOF balanced accuracy computed per repeat; mean ± SD reported\n"
        "\n"
        "3 Metabolite Variants (trained independently each fold)\n"
        "  • met_nan     — values below −500 µM replaced with NaN  (original approach)\n"
        "  • met_raw     — raw concentration, no floor correction\n"
        "  • met_no_malate — MalateGlo feature excluded entirely\n"
        "\n"
        "Late Fusion Strategies\n"
        "  • mean_prob: average OOF probabilities across modalities, threshold 0.5\n"
        "  • majority_vote: ≥ 2-of-3 modalities predict NAcc"
    )
    _add_textbox(slide, design_text, M, Inches(0.75), Inches(12.5), Inches(6.5),
                 fontsize=13, color="#000000")

    # ── Slide 3: Aggregated two-panel figure ──────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    _add_textbox(slide, "Aggregated Results: Mean ± SD over 10 Repeats",
                 M, M, Inches(12.5), Inches(0.55), fontsize=22, bold=True, color="#1F3864")
    if TWO_PANEL_PATH.exists():
        slide.shapes.add_picture(str(TWO_PANEL_PATH), M, Inches(0.75),
                                 width=Inches(12.5))

    # ── Slide 4: Summary table ────────────────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    _add_textbox(slide, "Summary Table: Balanced Accuracy (mean ± SD)",
                 M, M, Inches(12.5), Inches(0.55), fontsize=22, bold=True, color="#1F3864")
    if TABLE_PATH.exists():
        slide.shapes.add_picture(str(TABLE_PATH), M, Inches(0.75),
                                 width=Inches(12.5))

    # ── Slide 5: Met variant comparison ───────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    _add_textbox(slide, "Metabolite Variant Comparison",
                 M, M, Inches(12.5), Inches(0.55), fontsize=22, bold=True, color="#1F3864")
    fig = _plot_met_comparison(combined, days)
    stream = _fig_to_stream(fig); plt.close(fig)
    slide.shapes.add_picture(stream, M, Inches(0.75), width=Inches(12.5))

    # ── Slide 6: Spaghetti / variance across repeats ──────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    _add_textbox(slide, "Variance Across 10 Repeats — Key Strategies",
                 M, M, Inches(12.5), Inches(0.55), fontsize=22, bold=True, color="#1F3864")
    fig = _plot_spaghetti(combined, days)
    stream = _fig_to_stream(fig); plt.close(fig)
    slide.shapes.add_picture(stream, M, Inches(0.75), width=Inches(12.5))

    # ── Slides 7-16: Per-repeat results + Dy30 confusion matrices ────────────
    has_cms = bool(combined.get("Dy30", {}).get("met_nan", {}).get("repeat_confusion_matrices"))
    for rep in range(10):
        slide = prs.slides.add_slide(_blank_layout(prs))
        _add_textbox(slide, f"Repeat {rep+1} / 10 — OOF Balanced Accuracy by Day",
                     M, M, Inches(12.5), Inches(0.55), fontsize=22, bold=True, color="#1F3864")
        fig = _plot_repeat(combined, days, rep,
                           title=f"Repeat {rep+1}/10  (seed = {1 + rep*1000})")
        stream = _fig_to_stream(fig); plt.close(fig)
        if has_cms:
            slide.shapes.add_picture(stream, M, Inches(0.7), width=Inches(12.5))
            fig_cm = _plot_cm_row(combined, rep, day="Dy30")
            stream_cm = _fig_to_stream(fig_cm); plt.close(fig_cm)
            slide.shapes.add_picture(stream_cm, M, Inches(4.6), width=Inches(9.5))
        else:
            slide.shapes.add_picture(stream, M, Inches(0.7), width=Inches(12.5))

    OUT_PPT.parent.mkdir(exist_ok=True)
    prs.save(str(OUT_PPT))
    print(f"Saved {OUT_PPT}  ({len(prs.slides)} slides)")


def main():
    combined = json.loads(COMBINED_PATH.read_text())
    days = [d for d in DAY_ORDER if d in combined]
    build_ppt(combined, days)


if __name__ == "__main__":
    main()

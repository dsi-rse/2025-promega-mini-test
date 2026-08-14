#!/usr/bin/env python3
"""Shared helpers for paper-replication scripts.

Centralizes ``compute_classification_metrics`` (was duplicated 3× across the
legacy paper scripts with subtly different key names) and a single
``plot_balanced_accuracy_by_day`` helper that all the per-day comparison
plots can share.
"""

from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Stable-key metrics dict for binary classification.

    Convention: 1 = Not Acceptable, 0 = Acceptable (matches LABEL_TO_INT).
    Per-class precision/recall/f1 are returned at index [0]=Acceptable,
    [1]=Not Acceptable to match sklearn's labels=[0,1] ordering.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for NAcc (positive, label 1)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for Acc  (negative, label 0)

    # labels=[0,1] → index 0 = Acceptable, index 1 = Not Acceptable
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )

    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "sensitivity": round(tpr, 4),
        "specificity": round(tnr, 4),
        "precision_not_acceptable": round(prec[1], 4),
        "recall_not_acceptable": round(rec[1], 4),
        "f1_not_acceptable": round(f1[1], 4),
        "precision_acceptable": round(prec[0], 4),
        "recall_acceptable": round(rec[0], 4),
        "f1_acceptable": round(f1[0], 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return out


def plot_balanced_accuracy_by_day(
    series: Mapping[str, Dict[str, dict]],
    *,
    day_order: list,
    output_path: Path,
    title: str,
    style_overrides: Optional[Dict[str, dict]] = None,
    late_stage_shade_from_day: Optional[int] = None,
) -> None:
    """Plot balanced_accuracy by day for one or more model series.

    series: {model_label: {day: result_dict}}.  Each result_dict must have a
            'balanced_accuracy' float field.
    day_order: list of canonical day strings to use for the x-axis.
    style_overrides: per-label dict of matplotlib kwargs (color, marker, ...).
    late_stage_shade_from_day: if given (e.g. 24), shade the region from that
            day onwards in light grey.
    """
    days = []
    ys_per_label:  dict = {label: [] for label in series}
    std_per_label: dict = {label: [] for label in series}

    for day in day_order:
        present = any(day in s for s in series.values())
        if not present:
            continue
        days.append(day)
        for label, s in series.items():
            r = s.get(day)
            if r is None:
                ys_per_label[label].append(None)
                std_per_label[label].append(None)
            else:
                # Prefer mean from k-fold CV; fall back to single-split value
                ba = r.get("balanced_accuracy_mean", r.get("balanced_accuracy"))
                ys_per_label[label].append(ba)
                std_per_label[label].append(r.get("balanced_accuracy_std"))

    if not days:
        print(f"plot_balanced_accuracy_by_day: no overlap on {day_order}")
        return

    style_overrides = style_overrides or {}
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, ys in ys_per_label.items():
        stds = std_per_label[label]
        color = style_overrides.get(label, {}).get("color")
        marker = style_overrides.get(label, {}).get("marker", "o")
        linestyle = style_overrides.get(label, {}).get("linestyle", "-")

        valid = [(i, y, s) for i, (y, s) in enumerate(zip(ys, stds)) if y is not None]
        if not valid:
            continue
        xs, vals, svals = zip(*valid)
        xs, vals, svals = list(xs), list(vals), list(svals)

        line, = ax.plot(xs, vals, marker=marker, linestyle=linestyle,
                        label=label, color=color, linewidth=2)
        plot_color = line.get_color()

        # Shade ±1 std if available
        if any(s is not None for s in svals):
            lo = [v - s if s is not None else v for v, s in zip(vals, svals)]
            hi = [v + s if s is not None else v for v, s in zip(vals, svals)]
            ax.fill_between(xs, lo, hi, color=plot_color, alpha=0.15, linewidth=0)

    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=45)
    ax.set_xlabel("Day")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    if late_stage_shade_from_day is not None:
        from pipeline.data_loader import get_day_int_floor
        late_idx = next(
            (i for i, d in enumerate(days) if (get_day_int_floor(d) or 0) >= late_stage_shade_from_day),
            None,
        )
        if late_idx is not None:
            ax.axvspan(late_idx - 0.5, len(days) - 0.5, alpha=0.1, color="gray")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

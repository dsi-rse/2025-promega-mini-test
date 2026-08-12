#!/usr/bin/env python3
"""
make_organoid_timeline.py — build a PNG montage of organoid images across days.

For each organoid passed in, pulls its 11 timepoint images from the cohort
test split and arranges them as a row (day 3 → day 30). Multiple organoids
stack vertically. Useful for eyeballing organoids that the models keep
misclassifying — see which morphology / day-to-day trajectory trips them up.

Usage
-----
    # Explicit organoid IDs:
    python analysis/images/cnn_lstm/make_organoid_timeline.py \\
        --ids BA2_96_2_G6_nosplit BA1_96_1_B9_nosplit BA2_96_1_G7_nosplit \\
        --cohorts-dir data/cohorts \\
        --output-dir /net/projects2/promega/project_data/amanda_test/model_plots \\
        --tag worst_shared

    # Top-N pulled from an analyze_misses head-to-head CSV:
    python analysis/images/cnn_lstm/make_organoid_timeline.py \\
        --from-csv /net/projects2/promega/project_data/amanda_test/model_plots/misses_headtohead_idor_vs_idor_minvotes3.csv \\
        --top 6 \\
        --cohorts-dir data/cohorts \\
        --output-dir /net/projects2/promega/project_data/amanda_test/model_plots

The script searches every cohort under --cohorts-dir for each organoid's
series test.json and uses whichever match has the timepoints. If an organoid
isn't in any test split it's skipped with a warning. Both 'clipped' and 'std'
image variants are supported via --image-type.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def discover_test_splits(cohorts_dir: Path) -> dict[str, dict]:
    """
    Build a global organoid_id -> entry map by walking every cohort's series
    test.json under cohorts_dir. If the same id appears in multiple cohorts,
    the first one wins (which is fine — the image paths don't depend on cohort).
    """
    out: dict[str, dict] = {}
    if not cohorts_dir.exists():
        return out
    for cohort_dir in sorted(cohorts_dir.iterdir()):
        test_p = cohort_dir / "series" / "test.json"
        if not test_p.exists():
            continue
        try:
            with open(test_p) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for oid, entry in d.items():
            entry = dict(entry)
            entry.setdefault("_cohort", cohort_dir.name)
            out.setdefault(oid, entry)
    return out


def ids_from_csv(csv_path: Path, top: int) -> list[str]:
    """Read an analyze_misses head-to-head CSV and return the top-N organoid_ids."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    # Sort by mean_miss_rate if present, else by first 'miss_rate__*' column
    if "mean_miss_rate" in rows[0]:
        rows.sort(key=lambda r: -float(r.get("mean_miss_rate") or 0))
    else:
        rate_cols = [c for c in rows[0] if c.startswith("miss_rate__")]
        if rate_cols:
            rows.sort(key=lambda r: -sum(float(r[c] or 0) for c in rate_cols))
    return [r["organoid_id"] for r in rows[:top]]


def _ax_title_for_organoid(entry: dict, miss_rate: float | None = None) -> str:
    lab = entry.get("label", "?")
    g   = entry.get("n_votes_good")
    t   = entry.get("n_votes_total")
    vote_str = f"{g}/{t}" if t else "?"
    parts = [entry.get("_cohort", "?"), lab, vote_str]
    if miss_rate is not None:
        parts.append(f"miss={miss_rate:.2f}")
    return " · ".join(parts)


def render_timeline(organoid_ids: list[str], splits: dict[str, dict],
                    image_type: str, output_path: Path,
                    miss_rates: dict[str, float] | None = None) -> Path:
    rows = []
    for oid in organoid_ids:
        entry = splits.get(oid)
        if entry is None:
            print(f"  [skip] {oid}: not in any test split")
            continue
        rows.append((oid, entry))
    if not rows:
        raise RuntimeError("no organoids matched any test split")

    # Sort timepoints by day for each organoid
    for _, entry in rows:
        entry["timepoints"] = sorted(entry["timepoints"], key=lambda tp: tp["mdl_day"])

    n_rows = len(rows)
    n_cols = max(len(entry["timepoints"]) for _, entry in rows)

    # Figure sizing: ~1.4" per image cell + extra width for left-side labels
    cell = 1.4
    fig_w = cell * n_cols + 3.0
    fig_h = cell * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]

    for r, (oid, entry) in enumerate(rows):
        tps = entry["timepoints"]
        miss = (miss_rates or {}).get(oid)
        row_title = oid + "\n" + _ax_title_for_organoid(entry, miss)

        # First column gets the row label on the y-axis
        for c in range(n_cols):
            ax = axes[r][c] if n_rows > 1 else axes[0][c]
            if c < len(tps):
                tp = tps[c]
                day = tp["mdl_day"]
                img_path = tp["img_paths"].get(image_type)
                if img_path and Path(img_path).exists():
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"err\n{e.__class__.__name__}",
                                ha="center", va="center", transform=ax.transAxes,
                                fontsize=7)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="gray")
                # Column headers on top row
                if r == 0:
                    ax.set_title(f"Dy {day:g}", fontsize=10)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        # Row label on the leftmost cell
        left_ax = axes[r][0] if n_rows > 1 else axes[0][0]
        left_ax.set_ylabel(row_title, rotation=0, ha="right", va="center",
                           fontsize=8, labelpad=10)

    fig.suptitle(f"Organoid timelines — image_type='{image_type}'",
                 fontsize=12, fontweight="bold", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", nargs="+",
                     help="Organoid IDs to include in the montage.")
    src.add_argument("--from-csv", type=Path,
                     help="Path to an analyze_misses head-to-head CSV; uses top-N by mean_miss_rate.")
    p.add_argument("--top", type=int, default=6,
                   help="When using --from-csv, how many top organoids to include (default 6).")
    p.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"),
                   help="Root holding data/cohorts/<label>/series/test.json (default: data/cohorts/).")
    p.add_argument("--image-type", choices=("clipped", "std"), default="clipped",
                   help="Which image variant to render. Default 'clipped' (what the models actually saw).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/net/projects2/promega/project_data/amanda_test/model_plots"),
                   help="Where the montage PNG is written.")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for the output filename. Default derived from input.")
    args = p.parse_args()

    splits = discover_test_splits(args.cohorts_dir)
    if not splits:
        print(f"[error] no cohort test splits found under {args.cohorts_dir}")
        return 1

    if args.ids:
        oids = list(args.ids)
        tag = args.tag or f"custom_{len(oids)}"
        miss_rates = None
    else:
        oids = ids_from_csv(args.from_csv, args.top)
        if not oids:
            print(f"[error] no organoids parsed from {args.from_csv}")
            return 1
        # Pull mean_miss_rate values if present for the title bar
        with open(args.from_csv) as f:
            reader = list(csv.DictReader(f))
        miss_rates = {
            r["organoid_id"]: float(r.get("mean_miss_rate") or 0)
            for r in reader if "mean_miss_rate" in r
        }
        tag = args.tag or args.from_csv.stem.replace("misses_headtohead_", "top_")

    out_path = args.output_dir / f"timeline_{tag}.png"
    print(f"[timeline] organoids ({len(oids)}): {oids}")
    print(f"[timeline] writing → {out_path}")
    render_timeline(oids, splits, args.image_type, out_path, miss_rates=miss_rates)
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
cohort_summary.py — summary table of cohort sizes and label balance.

Scans cohort folders produced by `make_splits.py` (default: data/cohorts/*/)
and reports, for each cohort and each view (series / full), the number of
organoids per partition (train / val / test) broken down by label.

Outputs three things to --output-dir:
    cohort_summary.txt   — plain-text table (also printed to stdout)
    cohort_summary.csv   — same data, machine-readable
    cohort_summary.png   — rendered as a figure (for slides / model_plots/)

Usage
-----
    # auto-discover all cohorts under data/cohorts/
    python scripts/splits/cohort_summary.py

    # explicit cohorts
    python scripts/splits/cohort_summary.py \\
        --cohorts data/cohorts/idor data/cohorts/idor_minvotes3 \\
                  data/cohorts/expanded data/cohorts/expanded_minvotes3

    # change output dir
    python scripts/splits/cohort_summary.py \\
        --output-dir /net/projects2/promega/project_data/amanda_test/model_plots

By default the summary is built for the SERIES view (the LSTM-relevant one).
Use --view full to summarize the per-day view instead.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARTITIONS = ("train", "val", "test")
LABEL_ORDER = ("Acceptable", "Not Acceptable")  # column order in the table


# ============================================================
# Discovery + loading
# ============================================================

def discover_cohorts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "MANIFEST.json").exists()])


def load_partition(cohort_dir: Path, view: str, partition: str) -> dict:
    p = cohort_dir / view / f"{partition}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def cohort_stats(cohort_dir: Path, view: str) -> dict:
    """Return a dict of per-partition counts for one cohort."""
    manifest = {}
    mp = cohort_dir / "MANIFEST.json"
    if mp.exists():
        try:
            with open(mp) as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            pass

    stats = {
        "name":      cohort_dir.name,
        "min_votes": manifest.get("min_majority_votes", "?"),
        "partitions": {},
        "total":     0,
        "total_by_label": Counter(),
    }

    for part in PARTITIONS:
        records = load_partition(cohort_dir, view, part)
        labels  = Counter(r.get("label") for r in records.values())
        n = len(records)
        stats["partitions"][part] = {
            "n":      n,
            "labels": labels,
        }
        stats["total"] += n
        stats["total_by_label"].update(labels)

    return stats


# ============================================================
# Rendering
# ============================================================

def _build_rows(cohorts: list[dict]) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows) for the summary table."""
    header = ["cohort", "min_votes"]
    for part in PARTITIONS:
        header += [f"{part}_n",
                   f"{part}_Acc",
                   f"{part}_NotAcc"]
    header += ["total_n", "total_Acc", "total_NotAcc"]

    rows: list[list[str]] = []
    for c in cohorts:
        row = [c["name"], str(c["min_votes"])]
        for part in PARTITIONS:
            p = c["partitions"][part]
            row += [
                str(p["n"]),
                str(p["labels"].get("Acceptable", 0)),
                str(p["labels"].get("Not Acceptable", 0)),
            ]
        row += [
            str(c["total"]),
            str(c["total_by_label"].get("Acceptable", 0)),
            str(c["total_by_label"].get("Not Acceptable", 0)),
        ]
        rows.append(row)
    return header, rows


def render_text_table(header: list[str], rows: list[list[str]]) -> str:
    """Aligned plain-text table."""
    all_rows = [header] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]

    def fmt(r): return "  ".join(s.ljust(w) for s, w in zip(r, widths))

    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    out = [fmt(header), sep] + [fmt(r) for r in rows]
    return "\n".join(out)


def render_csv(header: list[str], rows: list[list[str]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def render_png(header: list[str], rows: list[list[str]], path: Path, view: str) -> None:
    """Render the same data as a matplotlib table image."""
    fig_w = max(10, 0.9 * len(header))
    fig_h = max(2.0, 0.45 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Pretty up header labels — replace _n with #, _Acc with Acc, etc.
    pretty_header = [
        h.replace("_n", " #").replace("_Acc", " Acc").replace("_NotAcc", " NotAcc")
        for h in header
    ]

    # Auto-size each column to its widest cell (header or value) so long
    # cohort names like "expanded_minvotes3" don't get truncated.
    col_widths_chars = [
        max(len(pretty_header[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(pretty_header))
    ]
    total = sum(col_widths_chars)
    col_widths = [w / total for w in col_widths_chars]

    tbl = ax.table(cellText=rows, colLabels=pretty_header,
                   cellLoc="center", colLoc="center", loc="center",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)

    # Header styling
    for col_idx in range(len(pretty_header)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row backgrounds for legibility
    for r_idx in range(len(rows)):
        bg = "#f4f6f8" if r_idx % 2 == 0 else "white"
        for c_idx in range(len(pretty_header)):
            tbl[r_idx + 1, c_idx].set_facecolor(bg)

    ax.set_title(f"Cohort summary — view: {view}", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Entrypoint
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Summarize cohort sizes + label balance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cohorts-root", type=Path, default=Path("data/cohorts"),
        help="Root folder containing cohort subdirs. Default: data/cohorts/",
    )
    p.add_argument(
        "--cohorts", nargs="+", type=Path, default=None,
        help="Explicit cohort dirs to summarize (skips auto-discovery).",
    )
    p.add_argument(
        "--view", choices=("series", "full"), default="series",
        help="Which view to summarize. series = complete-timeseries (LSTM); "
             "full = per-day records. Default: series.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("/net/projects2/promega/project_data/amanda_test/model_plots"),
        help=("Where to write cohort_summary.{txt,csv,png}. "
              "Default: /net/projects2/promega/project_data/amanda_test/model_plots/"),
    )
    args = p.parse_args()

    if args.cohorts:
        cohort_dirs = [Path(c) for c in args.cohorts]
        missing = [c for c in cohort_dirs if not c.exists()]
        if missing:
            raise FileNotFoundError(f"Missing cohort dirs: {missing}")
    else:
        cohort_dirs = discover_cohorts(args.cohorts_root)
        if not cohort_dirs:
            print(f"[error] no cohorts found under {args.cohorts_root}")
            return 1

    print(f"[summary] view = {args.view}")
    print(f"[summary] cohorts: {[c.name for c in cohort_dirs]}")

    cohorts = [cohort_stats(c, args.view) for c in cohort_dirs]
    header, rows = _build_rows(cohorts)

    txt = render_text_table(header, rows)
    print()
    print(txt)
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = args.output_dir / f"cohort_summary_{args.view}.txt"
    csv_path = args.output_dir / f"cohort_summary_{args.view}.csv"
    png_path = args.output_dir / f"cohort_summary_{args.view}.png"

    txt_path.write_text(txt + "\n")
    render_csv(header, rows, csv_path)
    render_png(header, rows, png_path, args.view)

    print(f"[wrote]  {txt_path}")
    print(f"[wrote]  {csv_path}")
    print(f"[wrote]  {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

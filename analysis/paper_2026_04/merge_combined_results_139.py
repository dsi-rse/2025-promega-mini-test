#!/usr/bin/env python3
"""Merge per-day combined kfold results into one JSON, then regenerate figures.

Run after all parallel combined_kfold_139 array jobs complete:
    python3 -m analysis.paper_2026_04.merge_combined_results_139
"""

import json
import subprocess
import sys
from pathlib import Path

from pipeline.data_loader import DAY_ORDER

OUTPUT_DIR = Path("analysis_output/images")
DAYS = list(DAY_ORDER)
SUFFIX = "_139"
OUT_PATH = OUTPUT_DIR / f"combined_results_kfold_series_idor{SUFFIX}.json"


def main():
    merged = {}
    missing = []

    for day in DAYS:
        p = OUTPUT_DIR / f"combined_results_kfold_series_idor{SUFFIX}_{day}.json"
        if not p.exists():
            missing.append(day)
            continue
        with open(p) as f:
            day_data = json.load(f)
        merged.update(day_data)
        print(f"  {day}: loaded {len(day_data)} entries")

    if missing:
        print(f"\nWARNING: missing results for: {missing}")
        print("Run the missing day jobs before merging.")
        if len(missing) == len(DAYS):
            sys.exit(1)

    with open(OUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged {len(merged)} days → {OUT_PATH}")

    # Print quick summary
    keys_show = [
        ("met_nan",                    "met_nan"),
        ("met_raw",                    "met_raw"),
        ("met_no_mal",                 "met_no_malate"),
        ("morph",                      "morph"),
        ("img",                        "img"),
        ("all3/nan",                   "met_nan+morph+img_mean_prob"),
        ("all3/raw",                   "met_raw+morph+img_mean_prob"),
        ("all3/nomal",                 "met_no_malate+morph+img_mean_prob"),
    ]
    col_w = 14
    print("\nBalanced accuracy mean±std over repeats:")
    header = f"{'Day':<12}" + "".join(f"{lbl:>{col_w}}" for lbl, _ in keys_show)
    print(header)
    print("-" * len(header))
    for day in DAYS:
        dr = merged.get(day, {})
        row = f"{day:<12}"
        for lbl, k in keys_show:
            r = dr.get(k, {})
            ba  = r.get("balanced_accuracy_mean", float("nan"))
            std = r.get("balanced_accuracy_std",  float("nan"))
            cell = f"{ba:.3f}±{std:.3f}" if ba == ba else "—"
            row += f"{cell:>{col_w}}"
        print(row)

    # Regenerate two-panel and table figures
    print("\nRegenerating figures...")
    r = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "core_env",
         "python3", "-m", "analysis.paper_2026_04.plot_combined_comparison"],
        capture_output=False,
    )
    if r.returncode != 0:
        print("WARNING: plot_combined_comparison failed")
    else:
        print("Done.")


if __name__ == "__main__":
    main()

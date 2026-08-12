#!/usr/bin/env python3
"""
make_splits.py — unified cohort splitter for Promega organoid data.

Reads a cohort config (JSON) and produces reproducible train/val/test splits
in two complementary views:

    full/    per-day records grouped by organoid (no series-completeness
             requirement; useful for single-timepoint models, per-day ablations)
    series/  only organoids with all 11 expected timepoints present
             (for LSTM / time-series models)

Both views share the SAME base_well -> partition assignment, so a given well
always lands in the same partition (train/val/test) regardless of which view
a student loads. This keeps comparisons across views honest.

Usage
-----
    python scripts/splits/make_splits.py scripts/splits/configs/idor.json
    python scripts/splits/make_splits.py scripts/splits/configs/expanded.json

Each run writes:
    <output_dir>/
        full/{train,val,test,summary}.json
        series/{train,val,test,summary}.json
        MANIFEST.json

MANIFEST.json records the exact config, git SHA, all_data.json hash,
identifiers CSV hash, seed, and timestamp — everything needed to regenerate
or audit the split.

Design notes
------------
- Stage 1 filter (per-day quality): drops records where
    images.edge_fraction  > max_edge_fraction
    metadata.classification in exclude_classifications
  Applied uniformly to every record in scope; IDOR and BA4 alike.
- Stage 2 filter (series completeness): series view requires all 11
  expected timepoints to be present AND to have passed Stage 1. Full view
  skips Stage 2.
- Splits happen at the BASE WELL level (not organoid, not timepoint) to
  prevent leakage between presplit daughters and across days.
- Labels come from the pre-computed value['label']['value'] field at Dy30.
  A well needs a Dy30 label to appear in either view.

Supersedes: scripts/split_series_reproducible.py, split_data_reproducible.py,
            split_data_no_stitch.py (see scripts/splits/README.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


# ============================================================
# EXPECTED TIMEPOINTS (mdl_day floats)
# Dy20 and Dy21 both map to 20.5 (same physical timepoint, different naming)
# ============================================================
EXPECTED_DAYS: list[float] = [
    3.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0, 20.5, 24.0, 28.0, 30.0,
]
LABEL_DAY: float = 30.0  # Survey labels come from Dy30


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a cohort config JSON file."""
    with open(path) as f:
        cfg = json.load(f)

    required_top = {"name", "all_data", "cohort", "stage1", "split", "output_dir"}
    missing = required_top - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    cohort = cfg["cohort"]
    if "identifiers_csv" not in cohort:
        raise ValueError("config.cohort.identifiers_csv is required")
    cohort.setdefault("add_batches", [])

    stage1 = cfg["stage1"]
    stage1.setdefault("max_edge_fraction", 0.05)
    stage1.setdefault("exclude_classifications", ["Split", "SplitStitched"])

    split = cfg["split"]
    split.setdefault("seed", 42)
    split.setdefault("test_size", 0.2)
    split.setdefault("val_size", 0.1)

    return cfg


# ============================================================
# COHORT CONSTRUCTION (which base_wells are in scope)
# ============================================================

def _classified_id_to_base_well(s: str) -> str | None:
    """BA1_96_1_Dy30_A1_nosplit_nostitch -> BA1_96_1_A1."""
    parts = s.split("_")
    try:
        dy_idx = next(i for i, p in enumerate(parts) if p.startswith("Dy"))
    except StopIteration:
        return None
    after_dy = list(parts[dy_idx + 1:])
    suffixes = {"nosplit", "nostitch", "stitched", "split",
                "presplit", "split1", "split2"}
    while after_dy and after_dy[-1] in suffixes:
        after_dy.pop()
    if not after_dy:
        return None
    return "_".join(parts[:dy_idx] + after_dy)


def load_identifier_wells(csv_path: Path) -> set[str]:
    """
    Read the identifiers CSV and return the union of base_wells from both
    columns ("All analyzed" + "Classified"). See docstring at top for format.
    """
    wells: set[str] = set()
    if not csv_path.exists():
        raise FileNotFoundError(f"Identifiers CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 1 and row[0].strip():
                wells.add(row[0].strip())
            if len(row) >= 2 and row[1].strip():
                bw = _classified_id_to_base_well(row[1].strip())
                if bw:
                    wells.add(bw)
    return wells


def get_base_well(value: dict) -> str:
    """Canonical base_well: 'BA1_96_1_A1' from value['plate']."""
    plate = value.get("plate", {})
    batch = plate.get("batch", "").replace(" ", "_")
    well = plate.get("well", "")
    return f"{batch}_{well}"


def cohort_base_wells(cfg: dict, all_data: dict) -> tuple[set[str], dict]:
    """
    Build the cohort's set of base_wells:
        identifiers CSV
        UNION
        every base_well whose batch prefix matches an entry in cohort.add_batches
    Returns (wells, debug_counts).
    """
    csv_path = Path(cfg["cohort"]["identifiers_csv"])
    idor_wells = load_identifier_wells(csv_path)

    add_batches = [b.upper() for b in cfg["cohort"].get("add_batches", [])]
    added_wells: set[str] = set()
    if add_batches:
        for record in all_data.values():
            batch_full = record.get("plate", {}).get("batch", "")
            batch_prefix = batch_full.split()[0].upper() if batch_full else ""
            if batch_prefix in add_batches:
                added_wells.add(get_base_well(record))

    wells = idor_wells | added_wells
    return wells, {
        "identifiers_csv_wells": len(idor_wells),
        "added_batches": add_batches,
        "added_batch_wells": len(added_wells),
        "cohort_total_wells": len(wells),
    }


# ============================================================
# STAGE 1 FILTER (per-day quality)
# ============================================================

def stage1_pass(record: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Return (passes, reasons_if_failed)."""
    reasons: list[str] = []

    max_ef = cfg["stage1"]["max_edge_fraction"]
    ef = record.get("images", {}).get("edge_fraction")
    if ef is None:
        reasons.append("edge_fraction_missing")
    elif float(ef) > max_ef:
        reasons.append(f"edge_fraction>{max_ef}")

    excluded = set(cfg["stage1"]["exclude_classifications"])
    cls = record.get("metadata", {}).get("classification")
    if cls in excluded:
        reasons.append(f"classification={cls}")

    return (len(reasons) == 0, reasons)


# ============================================================
# GENEALOGY + SERIES CONSTRUCTION
# ============================================================

def extract_mdl_day(value: dict) -> float | None:
    day = value.get("day", {})
    n = day.get("number")
    if n is not None:
        return float(n)
    day_id = day.get("id", "")
    m = re.match(r"^Dy(\d+(?:\.\d+)?)$", day_id)
    if not m:
        return None
    v = float(m.group(1))
    return 20.5 if v in (20, 21) else v


def parse_split_type(main_id: str) -> str:
    if not main_id:
        return "nosplit"
    s = main_id.lower()
    if "presplit" in s: return "presplit"
    if "split2" in s:   return "split2"
    if "split1" in s:   return "split1"
    return "nosplit"


def is_blank(value: dict) -> bool:
    return value.get("metadata", {}).get("verification", {}).get("blank", False) is True


def has_image(value: dict) -> bool:
    cm = value.get("images", {}).get("clipped_meanfill", {})
    return bool(cm.get("cm_image_abs") and cm.get("cm_source_mask_abs"))


def build_genealogy(
    all_data: dict, cohort_wells: set[str], cfg: dict,
) -> tuple[dict, dict]:
    """
    Group records by base_well, then by split_type, keeping only records that:
      - have a valid Dy timepoint
      - have an image (clipped_meanfill)
      - belong to a well in `cohort_wells`
      - are not blanks
      - pass Stage 1
    Returns (genealogy, skipped_counts).
    """
    genealogy: dict = defaultdict(lambda: defaultdict(list))
    skipped = defaultdict(int)

    for key, value in all_data.items():
        if is_blank(value):
            skipped["blank"] += 1
            continue
        if not has_image(value):
            skipped["no_image"] += 1
            continue
        mdl_day = extract_mdl_day(value)
        if mdl_day is None:
            skipped["no_day"] += 1
            continue
        main_id = (
            value.get("images", {}).get("main_id")
            or value.get("metadata", {}).get("verification", {}).get("main_id", "")
        )
        if not main_id:
            skipped["no_main_id"] += 1
            continue

        base_well = get_base_well(value)
        if base_well not in cohort_wells:
            skipped["not_in_cohort"] += 1
            continue

        passes, reasons = stage1_pass(value, cfg)
        if not passes:
            for r in reasons:
                skipped[f"stage1:{r}"] += 1
            continue

        genealogy[base_well][parse_split_type(main_id)].append({
            "key": key,
            "mdl_day": mdl_day,
            "value": value,
            "main_id": main_id,
        })

    return genealogy, dict(skipped)


def _emit_series(organoid_id, base_well, genealogy_type, items, container):
    days_present = sorted({i["mdl_day"] for i in items})
    missing_days = sorted(set(EXPECTED_DAYS) - set(days_present))
    container.append({
        "organoid_id": organoid_id,
        "base_well": base_well,
        "genealogy_type": genealogy_type,
        "days_present": days_present,
        "missing_days": missing_days,
        "items": items,
    })


def build_series(genealogy: dict) -> list:
    """
    Construct organoid series (per-organoid lists of per-day items), handling
    the split-organoid genealogy:
        nosplit                   -> one series
        presplit + split1/split2  -> one series per daughter (shared presplit)
    We return ALL series (complete or not); the series-view filter is applied
    separately by `require_complete_series`.
    """
    out: list = []
    for base_well, splits in genealogy.items():
        nosplit  = sorted(splits.get("nosplit",  []), key=lambda x: x["mdl_day"])
        presplit = sorted(splits.get("presplit", []), key=lambda x: x["mdl_day"])
        split1   = sorted(splits.get("split1",   []), key=lambda x: x["mdl_day"])
        split2   = sorted(splits.get("split2",   []), key=lambda x: x["mdl_day"])

        if nosplit and not presplit and not split1 and not split2:
            _emit_series(f"{base_well}_nosplit", base_well, "nosplit", nosplit, out)
        elif presplit and (split1 or split2):
            for name, items in (("split1", split1), ("split2", split2)):
                if items:
                    _emit_series(
                        f"{base_well}_{name}", base_well,
                        f"presplit+{name}", presplit + items, out,
                    )
        elif presplit:
            _emit_series(f"{base_well}_presplit_only", base_well,
                         "presplit_only", presplit, out)
        else:
            for name, items in (("split1", split1), ("split2", split2)):
                if items:
                    _emit_series(f"{base_well}_{name}_no_presplit", base_well,
                                 f"{name}_no_presplit", items, out)
    return out


def require_complete_series(series_list: list) -> list:
    """Stage 2: keep only series with ALL EXPECTED_DAYS present."""
    return [s for s in series_list if not s["missing_days"]]


# ============================================================
# LABELING (Dy30 survey label)
#
# NOTE: Labels come from the Dy30 survey (5 human evaluators rating the
# organoid), which is independent of Dy30 image quality. So we look up the
# label directly from all_data.json rather than relying on the Dy30 record
# surviving Stage 1. An organoid whose Dy30 image has high edge_fraction is
# still label-valid — it just won't have a clean Dy30 image in its timepoints.
# ============================================================

def build_label_lookup(all_data: dict, min_votes: int = 4) -> dict:
    """
    Build (base_well, split_type) -> label_obj map from all Dy30 records.

    The label is RECOMPUTED here from `lab['votes']` using `min_votes` rather
    than trusting the upstream `lab['value']` field, which is fixed at a 4/5
    supermajority. With min_votes=4 the behavior matches the original; with
    min_votes=3 borderline 3/2 and 2/3 organoids also receive labels.
    """
    lookup: dict = {}
    for v in all_data.values():
        if v.get("day", {}).get("number") != LABEL_DAY:
            continue
        lab = v.get("label") or {}
        votes = lab.get("votes") or {}
        accept = votes.get("Acceptable", 0)
        reject = votes.get("Not Acceptable", 0)

        if accept >= min_votes and accept > reject:
            computed_value = "Acceptable"
        elif reject >= min_votes and reject > accept:
            computed_value = "Not Acceptable"
        else:
            continue  # not enough agreement either way (or a tie)

        bw = get_base_well(v)
        main_id = (
            v.get("images", {}).get("main_id")
            or v.get("metadata", {}).get("verification", {}).get("main_id", "")
        )
        # Carry forward the original metadata but overwrite value with the
        # threshold-aware label, so downstream code keeps working unchanged.
        lab_out = dict(lab)
        lab_out["value"] = computed_value
        lookup[(bw, parse_split_type(main_id))] = lab_out
    return lookup


def _genealogy_to_split_type(gt: str) -> str:
    if gt == "nosplit":        return "nosplit"
    if "split1" in gt:         return "split1"
    if "split2" in gt:         return "split2"
    return "presplit"          # presplit_only → presplit


def attach_labels(
    series_list: list, all_data: dict, min_votes: int = 4,
) -> tuple[list, int]:
    """
    Attach Dy30 label to each series by looking up all_data directly. Drop
    series that have no Dy30 survey label (never evaluated / never reached Dy30).

    `min_votes` controls how many evaluators on one side are required for a
    well to receive a label. Default 4 = current behavior; 3 = relaxed.
    """
    lookup = build_label_lookup(all_data, min_votes=min_votes)
    labeled: list = []
    dropped = 0
    for s in series_list:
        key = (s["base_well"], _genealogy_to_split_type(s["genealogy_type"]))
        lab_obj = lookup.get(key)
        if lab_obj is None:
            dropped += 1
            continue
        s["label"] = lab_obj["value"]
        votes = lab_obj.get("votes", {})
        s["n_votes_good"]  = votes.get("Acceptable", 0)
        s["n_votes_total"] = lab_obj.get("total_evaluations", 0)
        labeled.append(s)
    return labeled, dropped


# ============================================================
# SPLITTING (stratified by well-majority label, seed-controlled)
# ============================================================

def split_wells(
    labeled_series: list, seed: int, test_size: float, val_size: float,
) -> dict[str, str]:
    """
    Decide a partition ('train'|'val'|'test') for every base_well present in
    labeled_series. Returns {base_well: partition}.
    """
    well_labels: dict[str, list[str]] = defaultdict(list)
    for s in labeled_series:
        well_labels[s["base_well"]].append(s["label"])

    wells = list(well_labels.keys())
    majority = [max(set(lbls), key=lbls.count) for lbls in well_labels.values()]

    train_wells, test_wells, train_maj, _ = train_test_split(
        wells, majority, test_size=test_size, stratify=majority, random_state=seed,
    )
    train_final, val_wells = train_test_split(
        train_wells, test_size=val_size, stratify=train_maj, random_state=seed,
    )

    assignment = {w: "train" for w in train_final}
    assignment.update({w: "val"  for w in val_wells})
    assignment.update({w: "test" for w in test_wells})
    return assignment


# ============================================================
# OUTPUT SERIALIZATION
# ============================================================

def _timepoint_out(item: dict) -> dict:
    value  = item["value"]
    images = value.get("images", {})
    cm     = images.get("clipped_meanfill", {})
    return {
        "key":       item["key"],
        "mdl_day":   item["mdl_day"],
        "dayID":     value.get("day", {}).get("id"),
        "main_id":   item["main_id"],
        "split_type": parse_split_type(item["main_id"]),
        "img_paths": {
            "std":     images.get("img_path"),
            "clipped": cm.get("cm_image_abs"),
        },
        "mask_paths": {
            "std":     images.get("mask_path"),
            "clipped": cm.get("cm_source_mask_abs"),
        },
        "edge_fraction": images.get("edge_fraction"),
    }


def _series_out(series: dict) -> dict:
    timepoints = []
    seen = set()
    for item in sorted(series["items"], key=lambda x: x["mdl_day"]):
        if item["mdl_day"] in seen:
            continue
        seen.add(item["mdl_day"])
        timepoints.append(_timepoint_out(item))
    return {
        "organoid_id":    series["organoid_id"],
        "base_well":      series["base_well"],
        "genealogy_type": series["genealogy_type"],
        "label":          series["label"],
        "n_votes_good":   series.get("n_votes_good", 0),
        "n_votes_total":  series.get("n_votes_total", 0),
        "n_timepoints":   len(timepoints),
        "days_present":   series["days_present"],
        "missing_days":   series["missing_days"],
        "timepoints":     timepoints,
    }


def write_view(series_list: list, out_dir: Path, assignment: dict[str, str]) -> dict:
    """Write {train,val,test,summary}.json for one view. Returns counts dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    for s in series_list:
        part = assignment.get(s["base_well"])
        if part is None:
            continue  # well didn't make the overall label requirement
        buckets[part][s["organoid_id"]] = _series_out(s)

    for part, data in buckets.items():
        with open(out_dir / f"{part}.json", "w") as f:
            json.dump(data, f, indent=2)

    summary = {
        "counts_by_partition": {p: len(d) for p, d in buckets.items()},
        "wells_by_partition": {
            p: len({v["base_well"] for v in d.values()}) for p, d in buckets.items()
        },
        "label_counts_by_partition": {
            p: {lbl: sum(v["label"] == lbl for v in d.values())
                for lbl in {v["label"] for v in d.values()}}
            for p, d in buckets.items()
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================
# MANIFEST (provenance)
# ============================================================

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_manifest(
    config_path: Path,
    cfg: dict,
    out_dir: Path,
    cohort_stats: dict,
    skipped_counts: dict,
    full_summary: dict,
    series_summary: dict,
    min_majority_votes: int = 4,
) -> Path:
    manifest = {
        "created_at_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path":      str(config_path),
        "config":           cfg,
        "git_sha":          git_sha(),
        "all_data_sha256":  file_sha256(Path(cfg["all_data"])),
        "identifiers_csv_sha256": file_sha256(Path(cfg["cohort"]["identifiers_csv"])),
        "seed":             cfg["split"]["seed"],
        "expected_days":    EXPECTED_DAYS,
        "label_day":        LABEL_DAY,
        "min_majority_votes": min_majority_votes,
        "cohort_stats":     cohort_stats,
        "skipped_counts":   skipped_counts,
        "full_summary":     full_summary,
        "series_summary":   series_summary,
    }
    path = out_dir / "MANIFEST.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ============================================================
# ENTRYPOINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified cohort splitter (full + series views).",
    )
    parser.add_argument("config", type=Path,
                        help="Path to cohort config JSON (e.g. configs/idor.json)")
    parser.add_argument(
        "--min-majority-votes", type=int, default=4,
        help=("Minimum votes for one side to win the Dy30 label. Default 4 "
              "(current behavior: requires 4/5 supermajority). Set to 3 to "
              "include borderline 3/2 and 2/3 organoids."),
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help=("Suffix appended to the config's output_dir, so strict and "
              "relaxed variants can coexist on disk. Example: with "
              "output_dir='data/cohorts/idor' and --output-suffix='_minvotes3', "
              "results land in 'data/cohorts/idor_minvotes3/'."),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_out_dir = Path(cfg["output_dir"])
    if args.output_suffix:
        out_dir = base_out_dir.with_name(base_out_dir.name + args.output_suffix)
    else:
        out_dir = base_out_dir

    print(f"[config]  {args.config}  (cohort: {cfg['name']})")
    print(f"[load]    {cfg['all_data']}")
    with open(cfg["all_data"]) as f:
        all_data = json.load(f)
    print(f"          {len(all_data)} records")

    print("[cohort]  building base-well set...")
    wells, cohort_stats = cohort_base_wells(cfg, all_data)
    print(f"          {cohort_stats}")

    print("[stage1]  filtering + building genealogy...")
    genealogy, skipped = build_genealogy(all_data, wells, cfg)
    print(f"          wells surviving stage 1: {len(genealogy)}")
    print(f"          skipped: {skipped}")

    print("[series]  constructing organoid series...")
    all_series = build_series(genealogy)
    complete   = require_complete_series(all_series)
    print(f"          total series: {len(all_series)}  (complete: {len(complete)})")

    print("[labels]  attaching Dy30 labels...")
    # Label the 'full' candidate pool and derive 'series' from labeled-complete.
    # Labels come from all_data directly (survey-based, independent of Dy30
    # image quality), so an organoid whose Dy30 image fails Stage 1 still keeps
    # its label — it just won't have a clean Dy30 in its timepoints.
    print(f"          min_majority_votes = {args.min_majority_votes} "
          f"({'relaxed' if args.min_majority_votes < 4 else 'strict'})")
    labeled_full, dropped_full = attach_labels(
        all_series, all_data, min_votes=args.min_majority_votes,
    )
    labeled_series = [s for s in labeled_full if not s["missing_days"]]
    print(f"          full labeled:   {len(labeled_full)} (dropped no-label: {dropped_full})")
    print(f"          series labeled: {len(labeled_series)}")

    print(f"[split]   seed={cfg['split']['seed']} "
          f"test={cfg['split']['test_size']} val={cfg['split']['val_size']}")
    # Use the full labeled pool to decide the split at the base_well level,
    # so series-view assignments are a strict subset of full-view assignments.
    assignment = split_wells(
        labeled_full,
        seed=cfg["split"]["seed"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
    )

    print("[write]   full/  ...")
    full_summary   = write_view(labeled_full,   out_dir / "full",   assignment)
    print(f"          {full_summary['counts_by_partition']}")
    print("[write]   series/...")
    series_summary = write_view(labeled_series, out_dir / "series", assignment)
    print(f"          {series_summary['counts_by_partition']}")

    manifest_path = write_manifest(
        args.config, cfg, out_dir,
        cohort_stats=cohort_stats,
        skipped_counts=skipped,
        full_summary=full_summary,
        series_summary=series_summary,
        min_majority_votes=args.min_majority_votes,
    )
    print(f"[done]    manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Core runtime data loader for paper reproducibility.

Loads all_data.json, applies composable filters, derives labels, and (optionally)
assigns splits from a ``Splits`` object.

Usage:
    from pipeline.data_loader import OrganoidDataset, filters_for_mode
    from pipeline.splits import Splits

    # Default canonical split:
    ds = OrganoidDataset("data/all_data.json",
                         splits=Splits.canonical(),
                         filters=filters_for_mode("base"))

    # Filter + label only (no split assignment yet):
    ds = OrganoidDataset("data/all_data.json", filters=filters_for_mode("base"))
    # ... then either:
    splits = Splits.stratified_random(ds.organoid_labels(),
                                       ratios={"train": 0.8, "test": 0.2},
                                       seed=42, name="rand_80_20")
    ds.apply_splits(splits)

    ds.summary()
    train = ds.get_split("train", day="Dy13")
    X, y, ids = ds.get_metabolite_features("train", day="Dy13")
"""

import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.splits import Splits

# ---------------------------------------------------------------------------
# Output directory (set ANALYSIS_OUTPUT_DIR env var before running scripts)
# ---------------------------------------------------------------------------

ANALYSIS_OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", "analysis_output"))
FIGURE_DIR = Path(os.environ.get("FIGURE_DIR", "figures"))

# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------

REQUIRED_METABOLITES = [
    "GlucoseGlo",
    "GlutamateGlo",
    "LactateGlo",
    "PyruvateGlo",
    "BCAAGlo",
    "MalateGlo",
]
# MalateGlo was previously gated to days > 10 on the assumption that early-day
# values were not assayed. In this dataset malate's raw concentration_uM is in
# fact present on early days at the same ~80% coverage as every other day (its
# reads just sit near the noise floor), so it is now required for all days.
CONDITIONAL_METABOLITES = {}

# Metabolite numeric fields available per assay block (see data/normalized/README.md).
# "concentration_uM" / "initial_concentration" are raw assay-derived; "win" / "win_vol_norm"
# are Promega-residualized (winsorized + per-metabolite scaled, MalateGlo behaves differently
# because its raw reads sit at the noise floor).
RAW_METABOLITE_FIELDS = ("concentration_uM", "initial_concentration")
NORMALIZED_METABOLITE_FIELDS = ("win", "win_vol_norm")

LABEL_DAY = "Dy30"
HIGH_QUALITY_BATCHES = ("BA1", "BA2")
MIN_VOTES = 4

# Canonical label encoding. Per AGENTS.md rule #9, internal training uses
# 1 = Not Acceptable (positive/minority class), 0 = Acceptable. This makes
# pos_weight = n_neg/n_pos correctly upweight the minority class with
# BCEWithLogitsLoss, and aligns with the paper's reporting axis.
LABEL_TO_INT = {"Not Acceptable": 1, "Acceptable": 0}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Day ordering used throughout analysis
DAY_ORDER = [
    "Dy03", "Dy06", "Dy08", "Dy10", "Dy13",
    "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30",
]

# Map raw day IDs (as emitted by the normalized schema) to canonical day IDs
# used internally by analysis. The normalized schema emits unpadded forms
# ('Dy3') and decimal notation ('Dy20.5'); the loader and paper scripts use
# zero-padded + underscore forms ('Dy03', 'Dy20_5').
#
# Use ``raw_day_id(record)`` to read what was on disk (preserves Dy20 vs Dy21
# vs Dy20.5). Use ``canonical_day_id(raw)`` to map to the analysis-internal
# form (loses the Dy20/Dy21 distinction — both become Dy20_5).
DAY_ALIAS = {
    "Dy3": "Dy03", "Dy6": "Dy06", "Dy8": "Dy08",
    "Dy20": "Dy20_5", "Dy21": "Dy20_5", "Dy20.5": "Dy20_5",
}

# Image-mode → image_path-key mapping. Single source of truth for translating
# user-facing mode names to record keys.
# Flat modes resolve directly under images.*; nested modes use tuple (parent, key).
# cm_source_image / cm_source_mask are aspect-ratio-conserved (resized_575_square)
# and are the correct inputs for the per-day image classifier.
# img / overlay use resized_512x384 which is NOT aspect-ratio-conserved.
IMAGE_MODE_TO_PATH_KEY = {
    "img": "img_path",
    "mask": "mask_path",
    "overlay": "overlay_path",
    "cm_source_image": ("clipped_meanfill", "cm_source_image_abs"),
    "cm_source_mask": ("clipped_meanfill", "cm_source_mask_abs"),
}


# ---------------------------------------------------------------------------
# Helpers (public — used by analysis scripts)
# ---------------------------------------------------------------------------

def extract_organoid_id(record_key: str) -> str:
    """Strip the day component to get the organoid identity.

    'BA1 96_1 Dy30 A1' → 'BA1 96_1 A1'
    'BA1 96_1 Dy20.5 A1' → 'BA1 96_1 A1'
    """
    m = re.match(r"^(.*)\s+Dy\d+(?:\.\d+)?\s+(.*)$", record_key)
    return f"{m.group(1)} {m.group(2)}" if m else record_key


def get_batch(record: dict) -> Optional[str]:
    """Extract top-level batch prefix (e.g. 'BA1') from a record.

    The normalized schema stores the full plate identifier in
    ``record['plate']['batch']`` (e.g. 'BA1 96_1'); we return the first token.
    """
    batch = record.get("plate", {}).get("batch", "")
    return batch.split()[0] if batch else None


def raw_day_id(record: dict) -> str:
    """Return the day identifier as emitted by the normalized schema.

    Preserves distinctions like Dy20 vs Dy21 vs Dy20.5. Use
    ``canonical_day_id`` to fold these together for analysis.
    """
    return record.get("day", {}).get("id", "")


def canonical_day_id(day_id: str) -> str:
    """Map a raw day id to its canonical analysis-internal form via DAY_ALIAS."""
    return DAY_ALIAS.get(day_id, day_id)


def get_day_int_floor(day_id: str) -> Optional[int]:
    """Return the integer-floor of a day id (LOSSY for half-days).

    'Dy13' → 13, 'Dy20_5' → 20, 'Dy20.5' → 20.

    Use this for filters like ``day > 10`` where the half-day distinction
    doesn't matter. For exact day arithmetic use ``get_day_float``.
    """
    m = re.match(r"Dy(\d+)", day_id or "")
    return int(m.group(1)) if m else None


def get_day_float(day_id: str) -> Optional[float]:
    """Return the day id as a float, preserving half-days.

    'Dy13' → 13.0, 'Dy20_5' → 20.5, 'Dy20.5' → 20.5.
    """
    m = re.match(r"Dy(\d+)(?:[_.](\d+))?", day_id or "")
    if not m:
        return None
    whole = int(m.group(1))
    frac = m.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)


# ---------------------------------------------------------------------------
# Per-record accessors
# ---------------------------------------------------------------------------

def get_main_id(record: dict) -> Optional[str]:
    """Return ``record["images"]["main_id"]`` (e.g. 'BA1_96_1_Dy30_A1_nosplit_nostitch')."""
    return (record.get("images") or {}).get("main_id")


def get_classification_verification(record: dict) -> Optional[str]:
    """Return ``record["metadata"]["verification"]["classification_verification"]``.

    Token values: 'NoSplitNoStitched', 'SplitNoStitched', 'NoSplitStitched',
    'SplitStitched'.
    """
    return ((record.get("metadata") or {}).get("verification") or {}).get(
        "classification_verification"
    )


def is_split_record(record: dict) -> bool:
    """True if a record's classification_verification marks it as a split organoid."""
    v = get_classification_verification(record) or ""
    return "Split" in v and "NoSplit" not in v


def is_stitched_record(record: dict) -> bool:
    """True if a record's classification_verification marks it as stitched."""
    v = get_classification_verification(record) or ""
    return "Stitched" in v and "NoStitched" not in v


def get_edge_fraction(record: dict) -> Optional[float]:
    """Return ``record["images"]["edge_fraction"]`` (None until step 11 runs)."""
    return (record.get("images") or {}).get("edge_fraction")


def get_mask_area_um2(record: dict) -> Optional[float]:
    """Return ``record["images"]["mask_area_um2"]`` (None until step 11 runs).

    Segmentation-derived organoid area (foreground pixels x per-axis um/px).
    This is our own size measurement; it tracks Promega's volume (win/win_vol_norm)
    at R^2~=0.99 on log-log, but is an area (um^2), not their volume (um^3).
    """
    return (record.get("images") or {}).get("mask_area_um2")


def get_base_well(record: dict) -> str:
    """Underscore-form well identifier: 'BA1_96_1_A1'.

    Used as the grouping key for split assignment (so daughter organoids
    from the same well land in the same partition).
    """
    plate = record.get("plate") or {}
    batch = (plate.get("batch") or "").replace(" ", "_")
    well = plate.get("well") or ""
    return f"{batch}_{well}" if batch and well else ""


def get_clipped_meanfill_image_path(record: dict) -> Optional[str]:
    """Absolute path to the 575x575 AR-conserved source image (resized_575_square)."""
    return ((record.get("images") or {}).get("clipped_meanfill") or {}).get("cm_source_image_abs")


def get_clipped_meanfill_mask_path(record: dict) -> Optional[str]:
    """Absolute path to the 575x575 source mask used to apply the mean-fill clip."""
    return ((record.get("images") or {}).get("clipped_meanfill") or {}).get("cm_source_mask_abs")


def get_survey_vote_counts(record: dict) -> Tuple[int, int]:
    """Return (n_acceptable, n_total) *regular-image* survey votes from the Dy30 label.

    Counts only the regular-image bucket (``regular_votes``) — the bucket that
    actually decides the consensus label in the merge step (see
    ``surveys_mapper.compute_survey_majority``: ``consensus_label =
    winning_reg_label``, with the inverted-image bucket used only as a
    disagreement veto). Regular surveys cap at 5 votes, so this returns totals
    in 0..5. For the full reg+inverted tally (up to 10) use
    ``get_complete_survey_vote_counts``. Returns (0, 0) if no votes.
    """
    label = record.get("label") or {}
    votes = label.get("regular_votes") or {}
    n_acceptable = int(votes.get("Acceptable", 0))
    n_total = sum(int(v) for v in votes.values())
    return n_acceptable, n_total


def get_complete_survey_vote_counts(record: dict) -> Tuple[int, int]:
    """Return (n_acceptable, n_total) *combined* survey votes from the Dy30 label.

    Counts both the regular-image and inverted-image buckets (the merged
    ``votes`` field), so a re-shown organoid (regular + inverted pass) totals up
    to 10. This is a descriptive view of all evaluations collected; it does NOT
    match the consensus-label rule, which is regular-bucket-only (see
    ``get_survey_vote_counts``). Returns (0, 0) if no votes.
    """
    label = record.get("label") or {}
    votes = label.get("votes") or {}
    n_acceptable = int(votes.get("Acceptable", 0))
    n_total = int(label.get("total_evaluations") or sum(votes.values()))
    return n_acceptable, n_total


def main_id_to_organoid_id(main_id: str) -> Optional[str]:
    """Convert an underscore-separated main_id to canonical organoid_id form.

    'BA1_96_1_Dy30_A1_nosplit_nostitch' → 'BA1 96_1 A1'

    Returns None if the input doesn't match the expected pattern.
    """
    m = re.match(r"^(BA\d+)_(\d+)_(\d+)_Dy\d+(?:[_.]\d+)?_([A-Za-z]+\d+)", main_id or "")
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}_{m.group(3)} {m.group(4)}"


# ---------------------------------------------------------------------------
# Unfiltered organoid iteration
# ---------------------------------------------------------------------------

def _group_records_by_organoid(
    all_data: dict, batches: Optional[Tuple[str, ...]] = None
) -> Dict[str, dict]:
    """Group raw records by organoid id, optionally restricted to a batch set.

    Returns ``{org_id: {"batch": str, "records_by_day": {canonical_day: rec}}}``.
    """
    organoids: Dict[str, dict] = {}
    for record_key, rec in all_data.items():
        batch = get_batch(rec)
        if batches is not None and batch not in batches:
            continue
        org_id = extract_organoid_id(record_key)
        if org_id not in organoids:
            organoids[org_id] = {"batch": batch, "records_by_day": {}}
        organoids[org_id]["records_by_day"][canonical_day_id(raw_day_id(rec))] = rec
    return organoids


def iter_organoid_records(
    all_data_path,
    batches: Optional[Sequence[str]] = None,
):
    """Yield ``(org_id, records_by_day, batch)`` for every organoid in all_data.

    Unlike ``OrganoidDataset.iter_organoids()``, this does NOT apply filters,
    require splits, or drop unlabeled organoids — useful for verification
    scripts and modality-coverage stats that need the unfiltered pool.

    Parameters
    ----------
    all_data_path : str | Path
        Path to ``all_data.json``.
    batches : sequence of str, optional
        If given (e.g. ``("BA1", "BA2")``), restrict to those batch prefixes.
    """
    with open(all_data_path) as f:
        all_data = json.load(f)
    grouped = _group_records_by_organoid(
        all_data, tuple(batches) if batches else None
    )
    for org_id, info in grouped.items():
        yield org_id, info["records_by_day"], info["batch"]


# ---------------------------------------------------------------------------
# Composable filter functions
# ---------------------------------------------------------------------------
# Each filter is  (organoid_id, records_by_day: dict) → bool (True = keep)

def require_batches(*batches: str) -> Callable:
    """Keep organoids belonging to the specified batches."""
    batch_set = set(batches)

    def f(org_id: str, records: dict) -> bool:
        # Use the first available record to get batch
        for rec in records.values():
            return get_batch(rec) in batch_set
        return False

    f.__doc__ = f"require_batches({', '.join(batches)})"
    return f


def require_complete_metabolites(
    required: Sequence[str] = REQUIRED_METABOLITES,
) -> Callable:
    """Keep organoids that have all required metabolites on days where metabolite
    data is expected.

    Days with no metabolite data at all (e.g. Dy20, a pure imaging timepoint)
    are skipped — they don't disqualify the organoid.  The check is: for every
    day that has *any* metabolite dict, all ``required`` metabolites must be
    present with non-null concentration_uM.
    """

    def f(org_id: str, records: dict) -> bool:
        has_any_met_day = False
        for day_id, rec in records.items():
            mets = rec.get("metabolite")
            if not mets:
                continue  # day has no metabolite data at all — skip
            has_any_met_day = True
            for m in required:
                if m not in mets:
                    return False
                conc = mets[m].get("concentration_uM")
                if conc is None:
                    return False
        return has_any_met_day  # must have at least one day with metabolites

    f.__doc__ = "require_complete_metabolites"
    return f


def require_valid_images() -> Callable:
    """Keep organoids where every day has an ``img_path`` and ``mask_path``."""

    def f(org_id: str, records: dict) -> bool:
        for rec in records.values():
            imgs = rec.get("images") or {}
            if not imgs.get("img_path") or not imgs.get("mask_path"):
                return False
        return True

    f.__doc__ = "require_valid_images"
    return f


IDOR_CSV_PATH_DEFAULT = Path(
    os.environ.get("IDOR_CSV_PATH", "data/idor_organoids.csv")
)


def _load_idor_organoid_ids(csv_path: Optional[Path] = None):
    """Load the IDOR partner curation list (column 1: 266 evaluated organoids).

    Returns:
        (col1_org_ids: set[str], col2_pairs: list[(org_id, main_id)])
    """
    import csv as _csv

    path = Path(csv_path) if csv_path else IDOR_CSV_PATH_DEFAULT
    with open(path) as f:
        reader = _csv.reader(f)
        next(reader)  # header
        rows = list(reader)
    col1 = [r[0] for r in rows if r and r[0]]
    col2 = [r[1] for r in rows if len(r) > 1 and r[1]]

    col1_org_ids: set = set()
    for s in col1:
        parts = s.split("_")
        if len(parts) != 4:
            raise ValueError(f"unexpected IDOR col1 row format: {s!r}")
        col1_org_ids.add(f"{parts[0]} {parts[1]}_{parts[2]} {parts[3]}")

    col2_pairs = []
    for s in col2:
        oid = main_id_to_organoid_id(s)
        if oid is None:
            raise ValueError(f"unparseable IDOR col2 main_id: {s!r}")
        col2_pairs.append((oid, s))

    return col1_org_ids, col2_pairs


def idor_organoid_filter(csv_path: Optional[Path] = None) -> Callable:
    """Keep organoids in the IDOR partner-supplied col1 list (the 266 evaluated).

    The list is loaded once at filter construction; downstream calls are O(1).
    Use ``verify_idor_list()`` separately to assert the CSV matches all_data.json.
    """
    col1_org_ids, _ = _load_idor_organoid_ids(csv_path)

    def f(org_id: str, records: dict) -> bool:
        return org_id in col1_org_ids

    f.__doc__ = f"idor_organoid_filter({csv_path or IDOR_CSV_PATH_DEFAULT})"
    return f


def idor_ba1_ba2_filters(
    csv_path: Optional[Path] = None,
    *,
    verify_against_all_data: Optional[str] = None,
) -> List[Callable]:
    """Filters for the IDOR partner curation: BA1+BA2 + the 266-organoid col1 list.

    If ``verify_against_all_data`` is a path, runs ``verify_idor_list()`` first
    and raises AssertionError on any mismatch. Recommended for paper-replication
    scripts so the data contract is checked at filter-construction time.
    """
    if verify_against_all_data is not None:
        verify_idor_list(csv_path=csv_path, all_data_path=verify_against_all_data)
    return [
        require_batches(*HIGH_QUALITY_BATCHES),
        idor_organoid_filter(csv_path),
    ]


def verify_idor_list(
    csv_path: Optional[Path] = None,
    all_data_path: str = "data/all_data.json",
    *,
    verbose: bool = False,
) -> dict:
    """Assert the IDOR partner CSV matches the partner's stated semantics.

    Verifies the claims documented at
    ``analysis/verify_ba1_ba2_idor_list/verify.py``: 266 col1 organoids,
    BA1+BA2 only, no splits in col1, col2 ⊆ col1, col2 main_ids match Dy30
    record main_ids, col1\\col2 = (didn't reach Dy30) + 5 intro-survey.

    Returns a summary dict (counts, the 5 intro-survey IDs).  Raises
    AssertionError on any mismatch so the caller can gate downstream code.
    """
    INTRO_SURVEY_ORGANOIDS = 5
    EXPECTED_COL1_COUNT = 266

    col1_org_ids, col2_pairs = _load_idor_organoid_ids(csv_path)
    col2_org_ids = {oid for oid, _ in col2_pairs}

    organoids = {
        oid: {"batch": batch, "records_by_day": records}
        for oid, records, batch in iter_organoid_records(
            all_data_path, batches=HIGH_QUALITY_BATCHES
        )
    }

    def _ok(msg: str):
        if verbose:
            print(f"[OK] {msg}")

    # 1. col1 count
    assert len(col1_org_ids) == EXPECTED_COL1_COUNT, (
        f"col1 has {len(col1_org_ids)} unique organoids, expected {EXPECTED_COL1_COUNT}"
    )
    _ok(f"col1 has {EXPECTED_COL1_COUNT} unique organoids")

    # 2. col1 only BA1+BA2
    csv_batches = {oid.split()[0] for oid in col1_org_ids}
    assert csv_batches == {"BA1", "BA2"}, (
        f"col1 has unexpected batches: {csv_batches}"
    )
    _ok("col1 batches are exactly {BA1, BA2}")

    # 3. all col1 organoids present in all_data.json
    missing = col1_org_ids - set(organoids.keys())
    assert not missing, (
        f"{len(missing)} col1 organoids not in all_data.json: {sorted(missing)[:5]}"
    )
    _ok("every col1 organoid is present in all_data.json")

    # 4. no split organoid in col1
    splits = [
        (oid, day)
        for oid in col1_org_ids
        for day, rec in organoids[oid]["records_by_day"].items()
        if is_split_record(rec)
    ]
    assert not splits, f"{len(splits)} col1 records are split-classified: {splits[:5]}"
    _ok("no col1 organoid has any split-classified record")

    # 5. col2 ⊆ col1
    extra = col2_org_ids - col1_org_ids
    assert not extra, f"col2 has {len(extra)} organoids not in col1: {sorted(extra)[:5]}"
    _ok(f"col2 ({len(col2_org_ids)}) is a subset of col1 ({len(col1_org_ids)})")

    # 6. every col2 organoid has a Dy30 record + survey evaluations
    no_dy30 = [
        oid for oid in col2_org_ids
        if LABEL_DAY not in organoids[oid]["records_by_day"]
    ]
    assert not no_dy30, f"{len(no_dy30)} col2 organoids lack a Dy30 record: {no_dy30[:5]}"
    no_evals = [
        oid for oid in col2_org_ids
        if not (
            (organoids[oid]["records_by_day"][LABEL_DAY].get("survey") or {}).get(
                "evaluations"
            )
        )
    ]
    assert not no_evals, (
        f"{len(no_evals)} col2 organoids have a Dy30 record but no survey "
        f"evaluations: {no_evals[:5]}"
    )
    no_consensus = sum(
        1 for oid in col2_org_ids
        if (organoids[oid]["records_by_day"][LABEL_DAY].get("label") or {}).get(
            "value"
        ) not in ("Acceptable", "Not Acceptable")
    )
    _ok(
        f"every col2 organoid has a Dy30 record with survey evaluations "
        f"({no_consensus}/{len(col2_org_ids)} reviewed-but-no-consensus, informational)"
    )

    # 7. col2 main_ids match the actual Dy30 record main_ids
    mismatches = []
    for oid, csv_main_id in col2_pairs:
        actual = get_main_id(organoids[oid]["records_by_day"][LABEL_DAY])
        if actual != csv_main_id:
            mismatches.append((oid, csv_main_id, actual))
    assert not mismatches, (
        f"{len(mismatches)} col2 main_ids disagree with all_data.json: {mismatches[:3]}"
    )
    _ok("every col2 main_id matches the Dy30 record's main_id in all_data.json")

    # 8. col1 \ col2 = (didn't reach Dy30) + 5 intro-survey
    unclassified = col1_org_ids - col2_org_ids
    no_dy30_in_extra = [
        oid for oid in unclassified
        if LABEL_DAY not in organoids[oid]["records_by_day"]
    ]
    intro_survey = [
        oid for oid in unclassified
        if LABEL_DAY in organoids[oid]["records_by_day"]
    ]
    assert len(intro_survey) == INTRO_SURVEY_ORGANOIDS, (
        f"expected {INTRO_SURVEY_ORGANOIDS} intro-survey organoids "
        f"(reached Dy30 but excluded from classification); found "
        f"{len(intro_survey)}: {intro_survey}"
    )
    if verbose:
        print(
            f"\nCol1 \\ Col2 breakdown ({len(unclassified)} organoids):"
            f"\n  - never reached Day 30: {len(no_dy30_in_extra)}"
            f"\n  - reached Day 30 but excluded from classification: "
            f"{len(intro_survey)} (partner said {INTRO_SURVEY_ORGANOIDS})"
        )
    _ok(f"{INTRO_SURVEY_ORGANOIDS} intro-survey organoids accounted for: {intro_survey}")

    return {
        "col1_count": len(col1_org_ids),
        "col2_count": len(col2_org_ids),
        "intro_survey_organoids": sorted(intro_survey),
        "no_dy30_organoids": sorted(no_dy30_in_extra),
        "no_consensus_count": no_consensus,
    }


def require_complete_series(
    expected_days: Sequence[str] = tuple(DAY_ORDER),
    *,
    max_edge_fraction: float = 0.05,
    require_clipped_meanfill: bool = True,
    drop_split: bool = True,
    drop_stitched: bool = True,
    drop_blank: bool = True,
) -> Callable:
    """Keep organoids whose every expected day passes per-day quality.

    Equivalent to Amanda's Stage 1 + Stage 2 in ``make_splits.py``: per-day
    edge_fraction / classification gates AND series-completeness over the 11
    canonical timepoints. Used by the ``series_idor`` filter preset.
    """

    expected = tuple(expected_days)

    def f(org_id: str, records: dict) -> bool:
        for day in expected:
            rec = records.get(day)
            if rec is None:
                return False
            if drop_blank and ((rec.get("metadata") or {}).get("verification") or {}).get(
                "blank", False
            ):
                return False
            ef = get_edge_fraction(rec)
            if ef is None or ef > max_edge_fraction:
                return False
            if drop_split and is_split_record(rec):
                return False
            if drop_stitched and is_stitched_record(rec):
                return False
            if require_clipped_meanfill and not get_clipped_meanfill_image_path(rec):
                return False
        return True

    f.__doc__ = (
        f"require_complete_series(days={len(expected)}, "
        f"edge<={max_edge_fraction})"
    )
    return f


def exclude_classification(*types: str) -> Callable:
    """Drop organoids if ANY day has a classification_verification in *types*.

    Common types: 'Stitched', 'Split', 'SplitStitched', 'PreSplit'.
    The verification field uses combined tokens like 'NoSplitNoStitched',
    'SplitNoStitched', etc.  We check the main_id for split/stitch tokens.
    """
    # We check via the main_id which encodes split/stitch info
    exclude_stitched = any("stitch" in t.lower() for t in types)
    exclude_split = any("split" in t.lower() for t in types)

    def f(org_id: str, records: dict) -> bool:
        for rec in records.values():
            imgs = rec.get("images") or {}
            mid = (imgs.get("main_id") or "").lower()
            img_path = (imgs.get("img_path") or "").lower()
            combined = mid + " " + img_path
            if exclude_stitched and "stitched" in combined and "nostitch" not in combined:
                return False
            if exclude_split and "split" in combined and "nosplit" not in combined:
                return False
        return True

    f.__doc__ = f"exclude_classification({', '.join(types)})"
    return f


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def paper_label_fn(
    org_id: str,
    records: dict,
    label_day: str = LABEL_DAY,
    **_,
) -> Optional[str]:
    """Read the merge-step label at label_day.

    The merge step (`pipeline.surveys.surveys_mapper.compute_survey_majority`)
    has already done INV/regular-image vote aggregation and stored the result
    in `record["label"]["value"]`. Single source of truth: don't recompute.

    Returns 'Acceptable', 'Not Acceptable', or None (no consensus / excluded).
    """
    rec = records.get(label_day)
    if rec is None:
        return None
    return rec.get("label", {}).get("value")


# ---------------------------------------------------------------------------
# Default configuration matching paper
# ---------------------------------------------------------------------------

def default_filters() -> List[Callable]:
    """Filters used in the paper: BA1+BA2, complete metabolites, valid images."""
    return [
        require_batches(*HIGH_QUALITY_BATCHES),
        require_complete_metabolites(),
        require_valid_images(),
    ]


ALL_BATCHES = ("BA1", "BA2", "BA3", "BA4")
VALID_MODES = ("base", "switch1", "switch2", "switch3", "series_idor")
VALID_MODALITIES = ("both", "image", "metabolite")


def filters_for_mode(mode: str, modality: str = "both") -> List[Callable]:
    """Return filters for a named split mode + modality.

    Modes replace the former `scripts/split_data_reproducible.py` presets:

    - **base**: BA1+BA2, complete metabolites, valid images. Paper default.
      Both image and metabolite models see the exact same organoids.
    - **switch1**: Image model gets BA1+BA2 with valid images (metabolite
      data optional); metabolite model gets BA1+BA2 with complete metabolites
      (images optional). Gives the image model extra training data.
    - **switch2**: All 4 batches, intersection of image + metabolite.
      BA3+BA4 flagged as lower-quality by IDOR/Promega; use with caution.
    - **switch3**: Image model gets all 4 batches with valid images;
      metabolite model stays on the BA1+BA2 intersection.
    - **series_idor**: IDOR cohort (266 partner-curated organoids) with
      complete 11-day series, per-day edge_fraction <= 0.05, no Split/
      SplitStitched/blank, clipped_meanfill image present. The runtime
      equivalent of Amanda's ``data/cohorts/idor/series/*.json``.

    For **switch1** and **switch3** the two modalities see *different*
    organoid sets, so pass `modality="image"` or `"metabolite"` to select.
    For **base** and **switch2** the three modalities are equivalent.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if modality not in VALID_MODALITIES:
        raise ValueError(f"modality must be one of {VALID_MODALITIES}, got {modality!r}")

    if mode == "base":
        return default_filters()

    if mode == "switch1":
        if modality == "image":
            return [require_batches(*HIGH_QUALITY_BATCHES), require_valid_images()]
        if modality == "metabolite":
            return [require_batches(*HIGH_QUALITY_BATCHES), require_complete_metabolites()]
        return default_filters()  # both: intersection == base

    if mode == "switch2":
        return [
            require_batches(*ALL_BATCHES),
            require_complete_metabolites(),
            require_valid_images(),
        ]

    if mode == "series_idor":
        return [
            *idor_ba1_ba2_filters(),
            require_complete_series(),
        ]

    # switch3
    if modality == "image":
        return [require_batches(*ALL_BATCHES), require_valid_images()]
    if modality == "metabolite":
        return [require_batches(*HIGH_QUALITY_BATCHES), require_complete_metabolites()]
    return default_filters()  # both: intersection == base


# ---------------------------------------------------------------------------
# OrganoidDataset
# ---------------------------------------------------------------------------

class OrganoidDataset:
    """Runtime dataset built from all_data.json with flexible split assignment.

    Groups records by organoid, applies filters, derives labels, and provides
    accessors for metabolite features and image paths by split and day.

    Split assignment is optional. When ``splits`` is provided, every filtered
    organoid must be in ``splits.mapping`` or a ``ValueError`` is raised. When
    ``splits`` is None, the dataset is filter+label only; ``get_split()`` will
    raise until ``apply_splits()`` is called.
    """

    def __init__(
        self,
        all_data_path: str,
        splits: Optional[Splits] = None,
        filters: Optional[List[Callable]] = None,
        label_fn: Optional[Callable] = None,
        strict_splits: bool = False,
    ):
        self.all_data_path = Path(all_data_path)
        self.filters = filters if filters is not None else default_filters()
        self.label_fn = label_fn or paper_label_fn
        self._splits: Optional[Splits] = None

        with open(self.all_data_path) as f:
            self.all_data: dict = json.load(f)

        # org_id → {label, split (optional), records}
        self._organoids: Dict[str, dict] = {}
        self._build()

        if splits is not None:
            self.apply_splits(splits, strict=strict_splits)

    # -- construction --------------------------------------------------------

    def _build(self):
        """Group records by organoid, apply filters, derive labels. No split assignment here."""
        grouped: Dict[str, Dict[str, dict]] = {}
        for key, rec in self.all_data.items():
            org_id = extract_organoid_id(key)
            day_raw = rec.get("day", {}).get("id", "")
            day = canonical_day_id(day_raw)
            grouped.setdefault(org_id, {})[day] = rec

        for org_id, records in grouped.items():
            keep = True
            for filt in self.filters:
                if not filt(org_id, records):
                    keep = False
                    break
            if not keep:
                continue

            label = self.label_fn(org_id, records)
            if label is None:
                continue

            self._organoids[org_id] = {
                "label": label,
                "records": records,
            }

    def apply_splits(self, splits: Splits, *, strict: bool = False) -> None:
        """Assign each filtered organoid its partition from ``splits.mapping``.

        Filtered organoids missing from the mapping are dropped (and warned about),
        matching the historical ``_build_from_csv`` behavior. Pass ``strict=True``
        to instead raise ``ValueError`` listing the unmapped IDs — useful when
        constructing a Splits in-memory and you want a coverage assertion.

        Either way, organoids in the mapping that were filtered out are silently
        ignored (after a one-line summary warning).
        """
        filtered_ids = set(self._organoids.keys())
        mapping_ids = splits.organoid_ids()

        missing = filtered_ids - mapping_ids
        if missing:
            if strict:
                sample = sorted(missing)[:10]
                raise ValueError(
                    f"{len(missing)} filtered organoids are not in Splits {splits.name!r}; "
                    f"first {len(sample)}: {sample}"
                )
            sample = sorted(missing)[:5]
            warnings.warn(
                f"{len(missing)} filtered organoids are not in Splits {splits.name!r} "
                f"and were dropped (e.g. {sample}); pass strict=True to require coverage"
            )
            for oid in missing:
                del self._organoids[oid]

        mapping_only = mapping_ids - set(self._organoids.keys())
        if mapping_only:
            warnings.warn(
                f"{len(mapping_only)} organoids in Splits {splits.name!r} were dropped by filters/label derivation"
            )

        for org_id, info in self._organoids.items():
            info["split"] = splits.mapping[org_id]
        self._splits = splits

    def organoid_labels(self) -> Dict[str, str]:
        """Return ``{organoid_id: label}`` for every filtered+labeled organoid.

        Useful as input to ``Splits.stratified_random``.
        """
        return {oid: info["label"] for oid, info in self._organoids.items()}

    # -- accessors -----------------------------------------------------------

    @property
    def organoid_ids(self) -> List[str]:
        return list(self._organoids.keys())

    @property
    def splits(self) -> List[str]:
        """Sorted list of unique split names assigned via apply_splits."""
        if self._splits is None:
            return []
        return sorted(self._splits.split_names())

    @property
    def days(self) -> List[str]:
        """All canonical days present across all organoids, sorted."""
        ds = set()
        for o in self._organoids.values():
            ds.update(o["records"].keys())
        return [d for d in DAY_ORDER if d in ds]

    def get_split(
        self, split: str, day: Optional[str] = None
    ) -> Dict[str, dict]:
        """Get organoids for a split, optionally filtered to those having a specific day.

        Returns: {org_id: {label, records: {day: record, ...}}}

        Raises ``RuntimeError`` if no splits have been assigned (construct with
        ``splits=Splits(...)`` or call ``apply_splits()`` first).
        """
        if self._splits is None:
            raise RuntimeError(
                "No splits assigned to this OrganoidDataset. "
                "Pass splits= at construction or call apply_splits(splits) first."
            )
        result = {}
        for org_id, info in self._organoids.items():
            if info.get("split") != split:
                continue
            if day is not None and day not in info["records"]:
                continue
            result[org_id] = info
        return result

    def get_metabolite_features(
        self,
        split: str,
        day: str,
        include_growth: bool = False,
        include_initial: bool = True,
        field: str = "concentration_uM",
        normalize_by_size: bool = False,
        winsorize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Extract metabolite feature matrix for split + day.

        ``field`` selects which per-assay numeric to use as the primary column:
        ``"concentration_uM"`` (default, raw uM), ``"win"`` or ``"win_vol_norm"``
        (Promega-residualized — winsorized + per-metabolite scaled, see
        ``data/normalized/README.md``). When ``include_initial`` is True the
        secondary column is always raw ``initial_concentration`` regardless of
        ``field``. Selecting a normalized ``field`` requires Step 2 to have been
        run with ``--normalized-csv``.

        Note: ``win`` is NOT on the same numeric scale as ``concentration_uM``
        (the ratio is metabolite-specific, e.g. ~2.0 for Glucose, ~0.1 for
        Glutamate/Pyruvate). Use it for within-metabolite analyses or as a
        cleaner alternative to raw for modeling, not as a unit-equivalent.

        ``normalize_by_size`` divides every metabolite measurement (the ``field``
        column, ``initial_concentration``, and — when ``include_growth`` — the
        per-day delta) by the organoid's segmentation area ``mask_area_um2`` for
        that day, yielding size-normalized features (suffix ``_per_um2``). Rows
        whose record has no ``mask_area_um2`` are dropped. The delta is computed
        on the size-normalized value (curr/area_curr − prev/area_prev), so a
        scaled+growth feature is a size-normalized exchange rate.

        ``winsorize`` clips each metabolite measurement (the ``field`` column and
        ``initial_concentration``) to that day's 1st/99th percentile across the
        split's organoids, reproducing the RehenLab per-day winsorization
        (``pipeline/metabolites/winsorize.py``). Order of operations matches the
        paper: winsorize raw -> size-normalize -> delta. Adds a ``_win`` suffix
        (composes with ``_per_um2``). Deltas, when included, are computed on the
        winsorized (and size-normalized) values.

        Returns: (X, y, feature_names, organoid_ids)
            X: (n_samples, n_features) float array
            y: (n_samples,) int array  (1=Not Acceptable, 0=Acceptable)
            feature_names: list of feature column names
            organoid_ids: list of organoid IDs corresponding to rows
        """
        if field not in RAW_METABOLITE_FIELDS + NORMALIZED_METABOLITE_FIELDS:
            raise ValueError(
                f"field={field!r} not recognized; valid: "
                f"{RAW_METABOLITE_FIELDS + NORMALIZED_METABOLITE_FIELDS}"
            )

        day_num = get_day_int_floor(day)
        subset = self.get_split(split, day=day)

        # Determine which metabolites to include for this day
        active_mets = list(REQUIRED_METABOLITES)
        for met, cond_fn in CONDITIONAL_METABOLITES.items():
            if day_num is not None and cond_fn(day_num):
                active_mets.append(met)

        # Build feature names (order: <field> winsorized, then size-normalized)
        win_suffix = "_win" if winsorize else ""
        size_suffix = "_per_um2" if normalize_by_size else ""
        feat_names = []
        for met in active_mets:
            feat_names.append(f"{met}_{field}{win_suffix}{size_suffix}")
            if include_initial:
                feat_names.append(f"{met}_initial_concentration{win_suffix}{size_suffix}")

        # Winsorized inputs are READ from the persisted per-day-winsorized columns
        # in all_data.json (``<field>_win``, written by
        # ``pipeline.metabolites.winsorize --write`` / ``make winsorize-write``) --
        # they are NOT recomputed here, so every model input comes from
        # all_data.json (concentration_uM -> concentration_uM_win, etc.).
        val_field = f"{field}_win" if winsorize else field
        init_field = "initial_concentration_win" if winsorize else "initial_concentration"

        rows = []
        labels = []
        ids = []

        for org_id, info in subset.items():
            rec = info["records"].get(day)
            if rec is None:
                continue
            mets = rec.get("metabolite", {})

            size = None
            if normalize_by_size:
                size = get_mask_area_um2(rec)
                if not size:
                    # Can't size-normalize a record with no segmentation area.
                    continue

            row = []
            skip = False
            for met in active_mets:
                met_data = mets.get(met, {})
                val = met_data.get(val_field)
                if val is None:
                    skip = True
                    break
                if include_initial:
                    init = met_data.get(init_field, np.nan)
                if normalize_by_size:
                    val = val / size
                    if include_initial:
                        init = init / size
                row.append(val)
                if include_initial:
                    row.append(init)
            if skip:
                continue

            rows.append(row)
            labels.append(1 if info["label"] == "Not Acceptable" else 0)
            ids.append(org_id)

        if not rows:
            return (
                np.empty((0, len(feat_names))),
                np.empty(0, dtype=int),
                feat_names,
                [],
            )

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)

        # Optionally add growth features (difference from previous day)
        if include_growth and day_num is not None:
            X, feat_names, ids_out = self._add_growth_features(
                X, feat_names, ids, split, day, active_mets, include_initial,
                field=field, normalize_by_size=normalize_by_size, winsorize=winsorize,
            )
            y_out = []
            id_set = set(ids_out)
            for org_id in ids_out:
                y_out.append(
                    1 if self._organoids[org_id]["label"] == "Not Acceptable" else 0
                )
            y = np.array(y_out, dtype=int)
            ids = ids_out

        return X, y, feat_names, ids

    def _add_growth_features(
        self,
        X: np.ndarray,
        feat_names: List[str],
        org_ids: List[str],
        split: str,
        day: str,
        active_mets: List[str],
        include_initial: bool,
        field: str = "concentration_uM",
        normalize_by_size: bool = False,
        winsorize: bool = False,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Add growth (delta) features from the previous available day.

        The delta is computed on ``field`` (matching the level columns), on the
        per-day-winsorized value when ``winsorize`` is set, and on the
        size-normalized value (value / ``mask_area_um2``) when
        ``normalize_by_size`` is set -- so the delta is consistent with the
        levels (winsorize -> size-normalize -> subtract).
        """
        day_idx = DAY_ORDER.index(day) if day in DAY_ORDER else -1
        if day_idx <= 0:
            # No previous day available
            return X, feat_names, org_ids

        prev_day = DAY_ORDER[day_idx - 1]
        prev_day_num = get_day_int_floor(prev_day)

        # Determine previous-day active metabolites
        prev_mets = list(REQUIRED_METABOLITES)
        for met, cond_fn in CONDITIONAL_METABOLITES.items():
            if prev_day_num is not None and cond_fn(prev_day_num):
                prev_mets.append(met)

        # Only compute growth for metabolites available in both days
        growth_mets = [m for m in active_mets if m in prev_mets]

        win_suffix = "_win" if winsorize else ""
        size_suffix = "_per_um2" if normalize_by_size else ""
        growth_names = [f"{m}_growth{win_suffix}{size_suffix}" for m in growth_mets]

        # Winsorized deltas read the persisted ``<field>_win`` columns from
        # all_data.json (same source as the level columns); nothing recomputed.
        val_field = f"{field}_win" if winsorize else field

        new_rows = []
        new_ids = []
        keep_indices = []

        for i, org_id in enumerate(org_ids):
            info = self._organoids[org_id]
            curr_rec = info["records"][day]
            prev_rec = info["records"].get(prev_day)
            if prev_rec is None:
                continue

            curr_size = prev_size = None
            if normalize_by_size:
                curr_size = get_mask_area_um2(curr_rec)
                prev_size = get_mask_area_um2(prev_rec)
                if not curr_size or not prev_size:
                    continue

            prev_mets_data = prev_rec.get("metabolite", {})
            growth_row = []
            skip = False
            for m in growth_mets:
                curr_c = curr_rec.get("metabolite", {}).get(m, {}).get(val_field)
                prev_c = prev_mets_data.get(m, {}).get(val_field)
                if curr_c is None or prev_c is None:
                    skip = True
                    break
                if normalize_by_size:
                    curr_c = curr_c / curr_size
                    prev_c = prev_c / prev_size
                growth_row.append(curr_c - prev_c)
            if skip:
                continue

            new_rows.append(growth_row)
            new_ids.append(org_id)
            keep_indices.append(i)

        n_dropped = len(org_ids) - len(new_rows)
        if n_dropped:
            warnings.warn(
                f"growth features for {day} (prev {prev_day}, split {split!r}): "
                f"dropped {n_dropped}/{len(org_ids)} organoids lacking a usable "
                f"previous-day value for one of {growth_mets}",
                stacklevel=2,
            )

        if not new_rows:
            return X, feat_names, org_ids

        X_kept = X[keep_indices]
        growth_arr = np.array(new_rows, dtype=float)
        X_combined = np.hstack([X_kept, growth_arr])
        return X_combined, feat_names + growth_names, new_ids

    def get_image_paths(
        self, split: str, day: str, mode: str = "cm_source_image"
    ) -> List[Tuple[str, str, str]]:
        """Get image paths for split+day.

        Args:
            mode: 'cm_source_image' | 'cm_source_mask' | 'img' | 'mask' | 'overlay'
                  cm_source_image/cm_source_mask are aspect-ratio-conserved (resized_575_square).
                  img/overlay use resized_512x384 (not aspect-ratio-conserved).

        Returns: list of (organoid_id, label, path)
        """
        path_key = IMAGE_MODE_TO_PATH_KEY.get(mode, mode)

        subset = self.get_split(split, day=day)
        result = []
        for org_id, info in subset.items():
            rec = info["records"].get(day)
            if rec is None:
                continue
            imgs = rec.get("images", {})
            if isinstance(path_key, tuple):
                parent, key = path_key
                path = (imgs.get(parent) or {}).get(key)
            else:
                path = imgs.get(path_key)
            if path:
                result.append((org_id, info["label"], path))
        return result

    def get_record(self, org_id: str, day: str) -> Optional[dict]:
        """Get the raw record for an organoid+day."""
        info = self._organoids.get(org_id)
        if info is None:
            return None
        return info["records"].get(day)

    def iter_organoids(self):
        """Yield (org_id, info) for every organoid in the (filtered) dataset."""
        return iter(self._organoids.items())

    def organoid_label(self, org_id: str) -> Optional[str]:
        """Return the label string ('Acceptable' / 'Not Acceptable') or None."""
        info = self._organoids.get(org_id)
        return None if info is None else info["label"]

    def organoid_records(self, org_id: str) -> Dict[str, dict]:
        """Return {day: record} for one organoid, or {} if unknown."""
        info = self._organoids.get(org_id)
        return {} if info is None else info["records"]

    # -- summary -------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable dataset summary."""
        lines = []
        lines.append(f"OrganoidDataset: {len(self._organoids)} organoids")
        lines.append(f"  Source: {self.all_data_path}")
        if self._splits is not None:
            lines.append(f"  Splits: {self._splits.name} ({self._splits.provenance})")
        else:
            lines.append("  Splits: <none assigned>")
        lines.append(f"  Days: {', '.join(self.days)}")
        lines.append("")

        for split in self.splits:
            subset = self.get_split(split)
            if not subset:
                continue
            labels = Counter(v["label"] for v in subset.values())
            total = sum(labels.values())
            lines.append(
                f"  {split:5s}: {total:3d} organoids  "
                f"(Acceptable={labels.get('Acceptable', 0)}, "
                f"Not Acceptable={labels.get('Not Acceptable', 0)})"
            )

            # Day coverage
            day_counts = Counter()
            for v in subset.values():
                for d in v["records"]:
                    day_counts[d] += 1
            day_str = "  ".join(
                f"{d}={day_counts[d]}" for d in DAY_ORDER if d in day_counts
            )
            lines.append(f"         {day_str}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"OrganoidDataset({len(self._organoids)} organoids, "
            f"splits={self.splits}, days={len(self.days)})"
        )


# ---------------------------------------------------------------------------
# Deterministic train/val/test split (replaces materialized data_splits/*.json)
# ---------------------------------------------------------------------------

def split_organoids(
    dataset: "OrganoidDataset",
    *,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """Base-well-grouped, label-stratified train/val/test partition.

    Wells (not organoids) are the unit of split — daughter organoids from the
    same well always co-locate, preventing genealogy leakage. Stratification
    is by per-well majority label.

    Logic ported from ``scripts/splits/make_splits.py::split_wells`` so the
    output is byte-identical to Amanda's pre-materialized splits when given
    the same input set + seed.

    Returns ``(train_ids, val_ids, test_ids)``.
    """
    from sklearn.model_selection import train_test_split

    well_to_orgs: Dict[str, List[str]] = {}
    well_to_labels: Dict[str, List[str]] = {}
    for org_id, info in dataset.iter_organoids():
        any_rec = next(iter(info["records"].values()))
        well = get_base_well(any_rec)
        well_to_orgs.setdefault(well, []).append(org_id)
        well_to_labels.setdefault(well, []).append(info["label"])

    wells = list(well_to_orgs.keys())
    majority = [max(set(lbls), key=lbls.count) for lbls in (well_to_labels[w] for w in wells)]

    train_wells, test_wells, train_maj, _ = train_test_split(
        wells, majority,
        test_size=test_size, stratify=majority, random_state=seed,
    )
    train_final, val_wells = train_test_split(
        train_wells,
        test_size=val_size, stratify=train_maj, random_state=seed,
    )

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    for w in train_final:
        train_ids.extend(well_to_orgs[w])
    for w in val_wells:
        val_ids.extend(well_to_orgs[w])
    for w in test_wells:
        test_ids.extend(well_to_orgs[w])
    return train_ids, val_ids, test_ids

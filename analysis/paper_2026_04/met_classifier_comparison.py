#!/usr/bin/env python3
"""Met-only 10×4-fold repeated CV: 4 classifiers × 3 malate variants.

Classifiers : lgbm, logreg, svm, mlp
Met variants: met_nan (floor→NaN), met_raw (raw values), met_no_malate (drop MalateGlo)
Protocol    : 10 repeats × 4-fold stratified CV — same seeds as combined_kfold.

No GPU required.  All computations run on CPU.

Output:
  analysis_output/images/met_classifier_comparison.json

Usage:
    python3 -m analysis.paper_2026_04.met_classifier_comparison
    python3 -m analysis.paper_2026_04.met_classifier_comparison --days Dy30
    sbatch analysis/paper_2026_04/submit_met_classifier_comparison.slurm
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    OrganoidDataset,
    filters_for_mode,
    idor_ba1_ba2_filters,
    require_complete_series,
)
from pipeline.splits import Splits

from .metabolites_train import _features_for_day_all

warnings.filterwarnings("ignore", category=UserWarning)

SEED        = 42
N_REPEATS   = 10
N_FOLDS     = 4
FILTER_MODE = "series_idor"
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "images" / "met_classifier_comparison.json"

CLF_NAMES  = ["lgbm", "logreg", "svm", "mlp"]
MAL_MODES  = [("nan", "met_nan"), ("raw", "met_raw"), ("drop", "met_no_malate")]

# ── Hyper-parameter grids ────────────────────────────────────────────────────

LGBM_GRID = {
    "max_depth":         [3, 6],
    "num_leaves":        [15, 31],
    "min_child_samples": [5, 10],
    "learning_rate":     [0.05, 0.1],
    "n_estimators":      [100, 300],
}

LOGREG_GRID = {
    "C":        [0.01, 0.1, 1.0, 10.0],
    "penalty":  ["l1", "l2"],
    "max_iter": [1000],
}

SVM_GRID = {
    "C":     [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto"],
}

MLP_GRID = {
    "hidden_layer_sizes": [(64,), (64, 32)],
    "alpha":              [0.01, 0.1],
}


# ── Fold trainer ─────────────────────────────────────────────────────────────

def _train_fold(
    clf_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    fold_seed: int,
) -> Optional[np.ndarray]:
    """Train one classifier on one fold; return test probabilities or None."""
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    spw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))

    if clf_name == "lgbm":
        model = lgb.LGBMClassifier(
            objective="binary", scale_pos_weight=spw,
            random_state=fold_seed, verbosity=-1, n_jobs=1,
        )
        grid = GridSearchCV(model, LGBM_GRID, cv=inner_cv, scoring="f1", n_jobs=-1, refit=True)
        grid.fit(X_tr, y_tr)
        return grid.predict_proba(X_te)[:, 1]

    # Impute NaNs (median), then scale — required for logreg/svm/mlp
    imputer = SimpleImputer(strategy="median")
    X_tr_i = imputer.fit_transform(X_tr)
    X_te_i = imputer.transform(X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_te_s = scaler.transform(X_te_i)

    if clf_name == "logreg":
        model = LogisticRegression(class_weight="balanced", solver="saga", random_state=fold_seed)
        grid = GridSearchCV(model, LOGREG_GRID, cv=inner_cv, scoring="f1_weighted", n_jobs=-1, refit=True)
        grid.fit(X_tr_s, y_tr)
        return grid.predict_proba(X_te_s)[:, 1]

    if clf_name == "svm":
        model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=fold_seed)
        grid = GridSearchCV(model, SVM_GRID, cv=inner_cv, scoring="f1", n_jobs=-1, refit=True)
        grid.fit(X_tr_s, y_tr)
        return grid.predict_proba(X_te_s)[:, 1]

    if clf_name == "mlp":
        model = MLPClassifier(max_iter=500, random_state=fold_seed)
        sw = np.where(y_tr == 1, spw, 1.0)
        grid = GridSearchCV(model, MLP_GRID, cv=inner_cv, scoring="f1", n_jobs=-1, refit=True)
        grid.fit(X_tr_s, y_tr, sample_weight=sw)
        return grid.predict_proba(X_te_s)[:, 1]

    raise ValueError(f"Unknown classifier: {clf_name}")


# ── Per-day runner ───────────────────────────────────────────────────────────

def run_day(day: str, ds: OrganoidDataset, n_folds: int, n_repeats: int, verbose: bool) -> dict:
    """Run all classifiers × variants for one day; return results dict."""
    # Pre-compute features for each malate mode (expensive to recompute per repeat)
    features: Dict[str, tuple] = {}
    for mode, mode_key in MAL_MODES:
        X, y, _, org_ids = _features_for_day_all(ds, day, malate_mode=mode)
        if X is None or len(X) == 0:
            print(f"  [{day}] no data for mode={mode}, skipping day")
            return {}
        features[mode_key] = (X, y, org_ids)

    all_org_ids = features["met_nan"][2]
    all_labels  = features["met_nan"][1]
    n = len(all_org_ids)

    result_keys = [f"{clf}_{mk}" for clf in CLF_NAMES for _, mk in MAL_MODES]
    repeat_bas: Dict[str, List[float]] = {k: [] for k in result_keys}

    for rep in range(n_repeats):
        rep_seed = SEED + rep * 1000
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)

        oof: Dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in result_keys}

        for fold_i, (tr_idx, te_idx) in enumerate(cv.split(all_org_ids, all_labels)):
            fold_seed = rep_seed + fold_i

            for mode, mode_key in MAL_MODES:
                X, y, _ = features[mode_key]
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr = y[tr_idx]

                for clf in CLF_NAMES:
                    k = f"{clf}_{mode_key}"
                    p = _train_fold(clf, X_tr, y_tr, X_te, fold_seed)
                    if p is not None:
                        oof[k][te_idx] = p

            if verbose:
                print(f"  [{day}] rep={rep+1}/{n_repeats}  fold={fold_i+1}/{n_folds}  done")

        for k in result_keys:
            valid = ~np.isnan(oof[k])
            if valid.sum() > 0:
                yt = all_labels[valid]
                yp = (oof[k][valid] >= 0.5).astype(int)
                repeat_bas[k].append(float(balanced_accuracy_score(yt, yp)))

    results = {}
    for k in result_keys:
        bas = repeat_bas[k]
        if bas:
            results[k] = {
                "balanced_accuracy_mean":     float(np.mean(bas)),
                "balanced_accuracy_std":      float(np.std(bas)),
                "n_repeats":                  len(bas),
                "n_folds":                    n_folds,
                "repeat_balanced_accuracies": bas,
            }
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",      nargs="+", default=None,
                        help="Days to run (default: all DAY_ORDER)")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds",   type=int, default=N_FOLDS)
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--include-stitched", action="store_true",
                        help="Include stitched organoids (n=139; default: n=132)")
    args = parser.parse_args()

    days = args.days or list(DAY_ORDER)

    if args.include_stitched:
        ds = OrganoidDataset(
            ALL_DATA_PATH, splits=None,
            filters=[*idor_ba1_ba2_filters(), require_complete_series(drop_stitched=False)],
        )
    else:
        ds = OrganoidDataset(
            ALL_DATA_PATH, splits=Splits.canonical(),
            filters=filters_for_mode(FILTER_MODE),
        )
    print(f"Dataset: {len(ds.organoid_ids)} organoids  "
          f"({'including' if args.include_stitched else 'excluding'} stitched)")
    print(f"Protocol: {args.n_repeats}×{args.n_folds}-fold  "
          f"classifiers={CLF_NAMES}  days={days}")

    # Load any existing partial results
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming from {OUTPUT_PATH}  ({len(all_results)} days already done)")
    else:
        all_results = {}

    for day in days:
        if day in all_results:
            print(f"[{day}] already done, skipping")
            continue
        print(f"\n[{day}] running {args.n_repeats}×{args.n_folds}-fold ...")
        day_results = run_day(day, ds, args.n_folds, args.n_repeats, args.verbose)
        if day_results:
            all_results[day] = day_results
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved → {OUTPUT_PATH}")

    # Print summary table
    print("\n\n=== Summary (mean BA) ===")
    hdr = f"{'Day':<8}" + "".join(f"{'lgbm_nan':>12}{'logreg_nan':>12}{'svm_nan':>12}{'mlp_nan':>12}")
    print(hdr)
    for day in DAY_ORDER:
        if day not in all_results:
            continue
        r = all_results[day]
        row = f"{day:<8}"
        for k in ["lgbm_met_nan", "logreg_met_nan", "svm_met_nan", "mlp_met_nan"]:
            v = r.get(k)
            cell = f"{v['balanced_accuracy_mean']:.3f}" if v else "—"
            row += f"{cell:>12}"
        print(row)


if __name__ == "__main__":
    main()

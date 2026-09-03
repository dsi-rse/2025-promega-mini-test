#!/usr/bin/env python3
"""Metabolite-only classifier: LightGBM and Logistic Regression per day.

Evaluation: stratified k-fold cross-validation across all labeled organoids
(no fixed train/val/test split).  For each outer fold the best hyperparameters
are selected by an inner 3-fold GridSearch, then the model is evaluated on the
held-out fold at threshold 0.5.  Results report mean ± std balanced accuracy
across k folds.

Differences between LightGBM and LogReg are encoded in MODEL_SPECS, not as
forked functions.

Outputs:
  - $ANALYSIS_OUTPUT_DIR/metabolites/results.json
  - $ANALYSIS_OUTPUT_DIR/figures/LightGBM_vs_Logistic_Regression.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.metabolites_train"
    make run ARGS="-m analysis.paper_2026_04.metabolites_train --days Dy30 --n-folds 10"
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    LABEL_TO_INT,
    OrganoidDataset,
    REQUIRED_METABOLITES,
    CONDITIONAL_METABOLITES,
    CONCENTRATION_FLOOR,
    get_day_int_floor,
    filters_for_mode,
)
from pipeline.splits import Splits

from .common import compute_classification_metrics, plot_balanced_accuracy_by_day

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "metabolites"


@dataclass
class ModelSpec:
    """All the bits that distinguish LightGBM-day from LogReg-day."""

    name: str            # 'lgbm' / 'logreg'
    display: str         # 'LightGBM' / 'Logistic Regression'
    factory: Callable    # () → unfit estimator with class-balanced base config
    param_grid: dict
    cv_scoring: str      # 'f1' (lgbm: minority class) or 'f1_weighted' (lr)
    threshold_grid: np.ndarray
    threshold_scoring: Callable[[np.ndarray, np.ndarray], float]
    use_scaler: bool
    captures_feature_importance: bool


def _f1_minority(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _lgbm_factory(*, scale_pos_weight: float = 1.0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )


def _logreg_factory():
    return LogisticRegression(
        class_weight="balanced",
        random_state=SEED,
        solver="saga",
    )


MODEL_SPECS = {
    "lgbm": ModelSpec(
        name="lgbm",
        display="LightGBM",
        factory=_lgbm_factory,
        param_grid={
            "max_depth": [3, 6],
            "num_leaves": [31, 47, 63],
            "min_child_samples": [10, 20],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
        },
        cv_scoring="f1",
        threshold_grid=np.linspace(0.3, 0.7, 9),
        threshold_scoring=_f1_minority,
        use_scaler=False,
        captures_feature_importance=True,
    ),
    "logreg": ModelSpec(
        name="logreg",
        display="Logistic Regression",
        factory=_logreg_factory,
        param_grid={
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "max_iter": [1000],
        },
        cv_scoring="f1_weighted",
        threshold_grid=np.linspace(0.1, 0.9, 17),
        threshold_scoring=_f1_weighted,
        use_scaler=True,
        captures_feature_importance=False,
    ),
}


def _features_for_day_all(ds: OrganoidDataset, day: str, malate_mode: str = "nan"):
    """Pull (X, y, feat_names, org_ids) for ALL labeled organoids on one day.

    Concatenates train + val + test splits so that k-fold CV can reassign
    organoids freely without being constrained by the canonical split.
    Falls back to iterating organoid_ids directly when no splits are assigned.

    malate_mode (no-splits path only):
      'nan'  — values < CONCENTRATION_FLOOR replaced with NaN (default)
      'raw'  — raw concentration values, no floor applied
      'drop' — MalateGlo feature excluded entirely
    """
    # Fast path: canonical splits available (malate_mode not applied here)
    if ds._splits is not None:
        parts = []
        for split in ("train", "val", "test"):
            X, y, names, ids = ds.get_metabolite_features(
                split, day, include_growth=True, include_initial=True,
            )
            if len(X):
                parts.append((X, y, names, ids))
        if not parts:
            return np.empty((0, 0)), np.empty(0), [], []
        Xs, ys, names_list, ids_list = zip(*parts)
        feat_names = names_list[0]
        return (
            np.vstack(Xs),
            np.concatenate(ys),
            feat_names,
            [oid for ids in ids_list for oid in ids],
        )

    # No-splits fallback: iterate all organoids directly
    day_num = get_day_int_floor(day)
    active_mets = list(REQUIRED_METABOLITES)
    for met, cond_fn in CONDITIONAL_METABOLITES.items():
        if day_num is not None and cond_fn(day_num):
            if malate_mode == "drop" and met == "MalateGlo":
                continue
            active_mets.append(met)

    feat_names = []
    for met in active_mets:
        feat_names.append(f"{met}_concentration_uM")
        feat_names.append(f"{met}_initial_concentration")
        feat_names.append(f"{met}_growth_rate")

    rows, labels, ids = [], [], []
    for oid in ds.organoid_ids:
        label_str = ds.organoid_label(oid)
        if label_str not in LABEL_TO_INT:
            continue
        rec = ds.get_record(oid, day)
        if rec is None:
            continue
        mets = rec.get("metabolite", {})
        row = []
        skip = False
        for met in active_mets:
            met_data = mets.get(met, {})
            val = met_data.get("concentration_uM")
            if val is None:
                skip = True; break
            if malate_mode == "nan" and val < CONCENTRATION_FLOOR:
                val = np.nan
            row.append(val)
            row.append(met_data.get("initial_concentration", np.nan))
            row.append(met_data.get("growth_rate", np.nan))
        if skip:
            continue
        rows.append(row)
        labels.append(LABEL_TO_INT[label_str])
        ids.append(oid)

    if not rows:
        return np.empty((0, len(feat_names))), np.empty(0, dtype=int), feat_names, []
    return np.vstack(rows), np.array(labels, dtype=int), feat_names, ids


def _train_kfold(
    spec: ModelSpec,
    day: str,
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list,
    *,
    n_folds: int,
    verbose: bool,
) -> Optional[dict]:
    """k-fold CV evaluation: inner 3-fold GridSearch per fold, threshold fixed at 0.5."""
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None

    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(y))
    fold_bal_accs = []
    importances_list = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if spec.use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        spw = sum(y_tr == 0) / max(sum(y_tr == 1), 1)
        base = spec.factory() if spec.name == "logreg" else spec.factory(scale_pos_weight=spw)
        grid = GridSearchCV(base, spec.param_grid, cv=inner_cv,
                            scoring=spec.cv_scoring, n_jobs=-1, refit=True)
        grid.fit(X_tr, y_tr)

        fold_probs = grid.predict_proba(X_te)[:, 1]
        oof_probs[te_idx] = fold_probs

        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_metrics = compute_classification_metrics(y_te, fold_preds, fold_probs)
        fold_bal_accs.append(fold_metrics["balanced_accuracy"])

        if spec.captures_feature_importance and hasattr(grid.best_estimator_, "feature_importances_"):
            importances_list.append(grid.best_estimator_.feature_importances_)

        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}  bal_acc={fold_bal_accs[-1]:.3f}"
                  f"  params={grid.best_params_}")

    # Aggregate OOF predictions for overall metrics
    oof_preds = (oof_probs >= 0.5).astype(int)
    metrics = compute_classification_metrics(y, oof_preds, oof_probs)
    metrics["balanced_accuracy_mean"] = float(np.mean(fold_bal_accs))
    metrics["balanced_accuracy_std"]  = float(np.std(fold_bal_accs))
    metrics["n_folds"] = n_folds
    metrics["fold_balanced_accuracies"] = [float(v) for v in fold_bal_accs]
    metrics["feature_names"] = feat_names

    if importances_list:
        mean_imp = np.mean(importances_list, axis=0)
        ranked = sorted(zip(feat_names, mean_imp), key=lambda kv: kv[1], reverse=True)
        metrics["feature_importance"] = [{"feature": f, "importance": float(i)} for f, i in ranked]

    return metrics


def _print_aggregate(results: dict) -> None:
    print(f"\n{'=' * 60}\nAGGREGATE COMPARISON (Table 3)\n{'=' * 60}")
    for spec in MODEL_SPECS.values():
        per_day = results.get(spec.name, {})
        if not per_day:
            continue
        accs, bal_accs, recall_nas = [], [], []
        zero_recall_days = 0
        best_bal_acc = 0.0
        for day in DAY_ORDER:
            m = per_day.get(day)
            if m is None:
                continue
            accs.append(m["accuracy"])
            ba = m.get("balanced_accuracy_mean", m.get("balanced_accuracy", 0.0))
            ba_std = m.get("balanced_accuracy_std", 0.0)
            bal_accs.append(ba)
            r_na = m["recall_not_acceptable"]
            recall_nas.append(r_na)
            zero_recall_days += int(r_na == 0.0)
            best_bal_acc = max(best_bal_acc, ba)
            print(f"    {day:<10}  bal_acc={ba:.3f} ± {ba_std:.3f}")
        n_days = len(accs)
        print(f"\n{spec.display}:")
        print(f"  Avg Accuracy:       {np.mean(accs):.1%}")
        print(f"  Avg Balanced Acc:   {np.mean(bal_accs):.1%}")
        print(f"  Avg Recall (N.A.):  {np.mean(recall_nas):.1%}")
        print(f"  Days Recall_NA = 0: {zero_recall_days}/{n_days}")
        print(f"  Best Balanced Acc:  {best_bal_acc:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None,
                        help="Specific days to train (e.g. Dy30 Dy24)")
    parser.add_argument("--skip-lr", action="store_true", help="Skip logistic regression")
    parser.add_argument("--skip-lgbm", action="store_true", help="Skip LightGBM")
    parser.add_argument("--n-folds", type=int, default=10,
                        help="Number of outer CV folds (default: 10)")
    args = parser.parse_args()

    enabled = []
    if not args.skip_lgbm:
        enabled.append(MODEL_SPECS["lgbm"])
    if not args.skip_lr:
        enabled.append(MODEL_SPECS["logreg"])

    # Load all labeled organoids (no fixed split — k-fold assigns them)
    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode("base"))
    print(ds.summary())
    print(f"Cross-validation: {args.n_folds}-fold stratified (threshold fixed at 0.5)")

    days_to_train = args.days if args.days else DAY_ORDER
    results: dict = {spec.name: {} for spec in enabled}

    for day in days_to_train:
        if day not in ds.days:
            print(f"\nSkipping {day} (no data)")
            continue
        X, y, feat_names, org_ids = _features_for_day_all(ds, day)
        if len(X) == 0:
            print(f"\nSkipping {day} (no features)")
            continue
        print(f"\n  {day}: {len(X)} organoids, "
              f"{int(y.sum())} Not Acceptable / {int((y==0).sum())} Acceptable")
        for spec in enabled:
            print(f"\n{'=' * 50}\n{spec.display} - {day}\n{'=' * 50}")
            m = _train_kfold(spec, day, X, y, feat_names,
                             n_folds=args.n_folds, verbose=True)
            if m is None:
                continue
            results[spec.name][day] = m
            print(f"  Balanced Acc: {m['balanced_accuracy_mean']:.4f} ± {m['balanced_accuracy_std']:.4f}")
            print(f"  Recall (NA):  {m['recall_not_acceptable']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    _print_aggregate(results)

    if results.get("lgbm") and results.get("logreg"):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_balanced_accuracy_by_day(
            {"LightGBM": results["lgbm"], "Logistic Regression": results["logreg"]},
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / "LightGBM_vs_Logistic_Regression.png",
            title="LightGBM vs Logistic Regression: Balanced Accuracy by Day",
            style_overrides={
                "LightGBM":            {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
                "Logistic Regression": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
            },
            late_stage_shade_from_day=24,
        )


if __name__ == "__main__":
    main()

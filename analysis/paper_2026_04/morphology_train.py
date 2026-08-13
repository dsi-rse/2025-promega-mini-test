#!/usr/bin/env python3
"""Morphology-only classifier: LightGBM and Logistic Regression per day.

Features: Circ._win, AR_win, Solidity_win, Complexity_win, Feret_win,
          Area_win, Volume_win  (from the normalized residualized CSV).

Evaluation: stratified 5-fold CV across all labeled organoids (same protocol
as metabolites_train.py).  LightGBM handles NaN natively (Volume_win has a
few missing values).

Outputs:
  - $ANALYSIS_OUTPUT_DIR/morphology/results.json
  - $ANALYSIS_OUTPUT_DIR/figures/morphology_LightGBM_vs_LogReg.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.morphology_train"
    make run ARGS="-m analysis.paper_2026_04.morphology_train --n-folds 5"
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    filters_for_mode,
)
from pipeline.splits import Splits

from .common import compute_classification_metrics, plot_balanced_accuracy_by_day

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ALL_DATA_PATH = "data/all_data.json"
MORPH_CSV_PATH = "data/normalized/CONC_data_organoides_residualized_final.csv"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "morphology"

MORPH_FEATURES = [
    "Circ._win",
    "AR_win",
    "Solidity_win",
    "Complexity_win",
    "Feret_win",
    "Area_win",
    "Volume_win",
]

# Map DAY_ORDER strings to integer days in the CSV
DAY_TO_INT = {
    "Dy03": 3, "Dy06": 6, "Dy08": 8, "Dy10": 10, "Dy13": 13,
    "Dy15": 15, "Dy17": 17, "Dy20_5": 21, "Dy24": 24, "Dy28": 28, "Dy30": 30,
}


def _csv_id_to_ds_id(csv_id: str) -> str:
    """Convert CSV format 'BA1_96_1_A1' to dataset format 'BA1 96_1 A1'."""
    parts = csv_id.split("_")
    # parts = ['BA1', '96', '1', 'A1', ...]
    # batch=parts[0], plate=parts[1]+'_'+parts[2], well=rest
    return parts[0] + " " + parts[1] + "_" + parts[2] + " " + "_".join(parts[3:])


def _load_morph_df() -> pd.DataFrame:
    df = pd.read_csv(MORPH_CSV_PATH)
    df["_oid"] = df["Organoid"].apply(_csv_id_to_ds_id)
    return df.set_index(["_oid", "Day"])


def _features_for_day(ds: OrganoidDataset, morph_df: pd.DataFrame, day: str):
    """Build (X, y, feat_names, org_ids) for all labeled organoids on one day."""
    day_int = DAY_TO_INT.get(day)
    if day_int is None:
        return np.empty((0, 0)), np.empty(0), MORPH_FEATURES, []

    rows, labels, ids = [], [], []
    for oid in ds.organoid_ids:
        label_str = ds.organoid_label(oid)
        if label_str not in LABEL_TO_INT:
            continue
        try:
            row = morph_df.loc[(oid, day_int), MORPH_FEATURES]
        except KeyError:
            continue
        rows.append(row.values.astype(float))
        labels.append(LABEL_TO_INT[label_str])
        ids.append(oid)

    if not rows:
        return np.empty((0, 0)), np.empty(0), MORPH_FEATURES, []

    return np.vstack(rows), np.array(labels), MORPH_FEATURES, ids


def _f1_minority(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _lgbm_factory(scale_pos_weight=1.0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )


LGBM_PARAM_GRID = {
    "max_depth": [3, 6],
    "num_leaves": [15, 31],
    "min_child_samples": [5, 10],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 300],
}

LOGREG_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "max_iter": [1000],
}


def _train_kfold_lgbm(day, X, y, feat_names, *, n_folds, verbose):
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
        spw = sum(y_tr == 0) / max(sum(y_tr == 1), 1)
        grid = GridSearchCV(_lgbm_factory(spw), LGBM_PARAM_GRID,
                            cv=inner_cv, scoring="f1", n_jobs=-1, refit=True)
        grid.fit(X_tr, y_tr)
        fold_probs = grid.predict_proba(X_te)[:, 1]
        oof_probs[te_idx] = fold_probs
        fold_metrics = compute_classification_metrics(y_te, (fold_probs >= 0.5).astype(int), fold_probs)
        fold_bal_accs.append(fold_metrics["balanced_accuracy"])
        if hasattr(grid.best_estimator_, "feature_importances_"):
            importances_list.append(grid.best_estimator_.feature_importances_)
        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}  bal_acc={fold_bal_accs[-1]:.3f}"
                  f"  params={grid.best_params_}")

    oof_preds = (oof_probs >= 0.5).astype(int)
    metrics = compute_classification_metrics(y, oof_preds, oof_probs)
    metrics["balanced_accuracy_mean"] = float(np.mean(fold_bal_accs))
    metrics["balanced_accuracy_std"] = float(np.std(fold_bal_accs))
    metrics["n_folds"] = n_folds
    metrics["fold_balanced_accuracies"] = [float(v) for v in fold_bal_accs]
    metrics["feature_names"] = feat_names
    if importances_list:
        mean_imp = np.mean(importances_list, axis=0)
        ranked = sorted(zip(feat_names, mean_imp), key=lambda kv: kv[1], reverse=True)
        metrics["feature_importance"] = [{"feature": f, "importance": float(i)} for f, i in ranked]
    return metrics


def _train_kfold_logreg(day, X, y, feat_names, *, n_folds, verbose):
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y))
    fold_bal_accs = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Impute NaN with column median before scaling
        col_medians = np.nanmedian(X_tr, axis=0)
        for col_i, med in enumerate(col_medians):
            X_tr[np.isnan(X_tr[:, col_i]), col_i] = med
            X_te[np.isnan(X_te[:, col_i]), col_i] = med

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        base = LogisticRegression(class_weight="balanced", random_state=SEED, solver="saga")
        grid = GridSearchCV(base, LOGREG_PARAM_GRID,
                            cv=inner_cv, scoring="f1_weighted", n_jobs=-1, refit=True)
        grid.fit(X_tr, y_tr)
        fold_probs = grid.predict_proba(X_te)[:, 1]
        oof_probs[te_idx] = fold_probs
        fold_metrics = compute_classification_metrics(y_te, (fold_probs >= 0.5).astype(int), fold_probs)
        fold_bal_accs.append(fold_metrics["balanced_accuracy"])
        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}  bal_acc={fold_bal_accs[-1]:.3f}"
                  f"  params={grid.best_params_}")

    oof_preds = (oof_probs >= 0.5).astype(int)
    metrics = compute_classification_metrics(y, oof_preds, oof_probs)
    metrics["balanced_accuracy_mean"] = float(np.mean(fold_bal_accs))
    metrics["balanced_accuracy_std"] = float(np.std(fold_bal_accs))
    metrics["n_folds"] = n_folds
    metrics["fold_balanced_accuracies"] = [float(v) for v in fold_bal_accs]
    metrics["feature_names"] = feat_names
    return metrics


def _print_aggregate(results):
    print(f"\n{'=' * 60}\nAGGREGATE COMPARISON\n{'=' * 60}")
    for name, label in [("lgbm", "LightGBM"), ("logreg", "Logistic Regression")]:
        per_day = results.get(name, {})
        if not per_day:
            continue
        bal_accs = []
        print(f"\n{label}:")
        for day in DAY_ORDER:
            m = per_day.get(day)
            if m is None:
                continue
            ba = m.get("balanced_accuracy_mean", m.get("balanced_accuracy", 0.0))
            std = m.get("balanced_accuracy_std", 0.0)
            bal_accs.append(ba)
            print(f"  {day:<10}  bal_acc={ba:.3f} ± {std:.3f}"
                  f"  recall_NA={m['recall_not_acceptable']:.3f}")
        if bal_accs:
            print(f"  Avg bal_acc: {np.mean(bal_accs):.3f}  Best: {max(bal_accs):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--skip-lgbm", action="store_true")
    parser.add_argument("--skip-lr", action="store_true")
    args = parser.parse_args()

    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode("base"))
    print(ds.summary())
    morph_df = _load_morph_df()
    print(f"Morphology CSV: {len(morph_df)} rows, features: {MORPH_FEATURES}")
    print(f"Cross-validation: {args.n_folds}-fold stratified (threshold fixed at 0.5)")

    days_to_train = args.days if args.days else DAY_ORDER
    results = {"lgbm": {}, "logreg": {}}

    for day in days_to_train:
        X, y, feat_names, org_ids = _features_for_day(ds, morph_df, day)
        if len(X) == 0:
            print(f"\nSkipping {day} (no data)")
            continue
        print(f"\n  {day}: {len(X)} organoids, "
              f"{int(y.sum())} Not Acceptable / {int((y==0).sum())} Acceptable")

        if not args.skip_lgbm:
            print(f"\n{'='*50}\nLightGBM - {day}\n{'='*50}")
            m = _train_kfold_lgbm(day, X.copy(), y, feat_names,
                                   n_folds=args.n_folds, verbose=True)
            if m:
                results["lgbm"][day] = m
                print(f"  Balanced Acc: {m['balanced_accuracy_mean']:.4f} ± {m['balanced_accuracy_std']:.4f}")

        if not args.skip_lr:
            print(f"\n{'='*50}\nLogistic Regression - {day}\n{'='*50}")
            m = _train_kfold_logreg(day, X.copy(), y, feat_names,
                                     n_folds=args.n_folds, verbose=True)
            if m:
                results["logreg"][day] = m
                print(f"  Balanced Acc: {m['balanced_accuracy_mean']:.4f} ± {m['balanced_accuracy_std']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    _print_aggregate(results)

    if results.get("lgbm") or results.get("logreg"):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        series = {}
        if results["lgbm"]:
            series["LightGBM"] = results["lgbm"]
        if results["logreg"]:
            series["Logistic Regression"] = results["logreg"]
        plot_balanced_accuracy_by_day(
            series,
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / "morphology_LightGBM_vs_LogReg.png",
            title="Morphology Features: Balanced Accuracy by Day",
            style_overrides={
                "LightGBM":            {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
                "Logistic Regression": {"color": "#d62728", "marker": "s", "linestyle": "--"},
            },
            late_stage_shade_from_day=24,
        )
        out_png = FIGURE_DIR / "morphology_LightGBM_vs_LogReg.png"
        import shutil
        repo_fig = Path("figures/morphology_LightGBM_vs_LogReg.png")
        repo_fig.parent.mkdir(exist_ok=True)
        shutil.copy(out_png, repo_fig)
        print(f"Copied to {repo_fig}")


if __name__ == "__main__":
    main()

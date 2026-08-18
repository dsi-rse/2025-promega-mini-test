#!/usr/bin/env python3
"""
trajectory_vs_state.py — does the developmental TRAJECTORY predict the eventual
Acceptable/Not-Acceptable outcome beyond the LATEST image at each cutoff day?

At every cutoff day t, compares four feature sets built from the fixed CNN
embeddings (from extract_embeddings.py), on the SAME organoid-level CV folds:

    state          z_t                      (latest image only)
    change         z_t - z_prev             (recent morphological change)
    state+change   [z_t, z_t - z_prev]
    trajectory     mean of z over days <= t (history summary)

Same folds are used for all four sets at each cutoff, so the comparison is paired.
Pipeline per fold (fit on train only, no leakage): StandardScaler -> PCA ->
L2 logistic regression (class_weight='balanced'). Reported as balanced accuracy
and AUC on pooled out-of-fold predictions, averaged over repeated stratified CV,
with percentile CIs. A label-permutation run gives the chance floor.

The question: does change / trajectory beat state, especially at EARLY days? If
yes, dynamics are informative before endpoint morphology separates. If no, the
honest result is that current morphology carries the signal and history adds little.

    python analysis/images/cnn_lstm/trajectory_vs_state.py \\
        --emb embeddings_idor_balsel.npz --out-prefix traj_idor_balsel
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

CUTOFFS = [6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]
K_PCA = 10
N_REPEATS = 40
N_SPLITS = 5
FEATURE_SETS = ["state", "change", "state+change", "trajectory"]


def build_features(cutoff, org_days, org_z, org_label):
    """Return X dict {featureset: array}, y, and the organoid list for one cutoff.
    Includes only organoids that have day==cutoff AND at least one earlier day."""
    rows = defaultdict(list)
    y, keep = [], []
    for oid, day2z in org_z.items():
        days = sorted(d for d in day2z if d <= cutoff)
        if cutoff not in day2z or len(days) < 2:
            continue  # need latest frame + a prior frame
        z_t = day2z[cutoff]
        z_prev = day2z[days[-2]]                 # previous available day
        traj = np.mean([day2z[d] for d in days], axis=0)
        rows["state"].append(z_t)
        rows["change"].append(z_t - z_prev)
        rows["state+change"].append(np.concatenate([z_t, z_t - z_prev]))
        rows["trajectory"].append(traj)
        y.append(org_label[oid]); keep.append(oid)
    X = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
    return X, np.asarray(y), keep


def cv_scores(X, y, seed_shift=0, permute=False):
    """Repeated stratified CV; same folds across feature sets. Returns per-repeat
    lists of balanced accuracy and AUC (pooled out-of-fold) for each feature set."""
    n = len(y)
    k = min(K_PCA, min(np.bincount(y)) * (N_SPLITS - 1) // N_SPLITS, X["state"].shape[1])
    k = max(2, k)
    bal = {f: [] for f in FEATURE_SETS}
    auc = {f: [] for f in FEATURE_SETS}
    for rep in range(N_REPEATS):
        yy = y.copy()
        if permute:
            rng = np.random.RandomState(1000 + rep); rng.shuffle(yy)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rep + seed_shift)
        folds = list(skf.split(np.zeros(n), yy))  # identical folds for every feature set
        for f in FEATURE_SETS:
            oof = np.full(n, np.nan)
            for tr, te in folds:
                pipe = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=min(k, len(tr) - 1), svd_solver="randomized", random_state=0),
                    LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
                )
                pipe.fit(X[f][tr], yy[tr])
                oof[te] = pipe.predict_proba(X[f][te])[:, 1]
            bal[f].append(balanced_accuracy_score(yy, (oof > 0.5).astype(int)))
            try:
                auc[f].append(roc_auc_score(yy, oof))
            except ValueError:
                auc[f].append(np.nan)
    return bal, auc


def summarize(vals):
    a = np.array(vals, dtype=float); a = a[~np.isnan(a)]
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out-prefix", default="traj")
    ap.add_argument("--outdir", type=Path, default=Path("trajectory_out"),
                    help="Where to write the CSV + curve (relative to cwd). Default: ./trajectory_out")
    args = ap.parse_args()

    d = np.load(args.emb, allow_pickle=True)
    ids, days, labels, Z = d["organoid_ids"], d["days"], d["labels"], d["Z"]
    org_z = defaultdict(dict); org_label = {}
    for i in range(len(ids)):
        org_z[ids[i]][float(days[i])] = Z[i]
        org_label[ids[i]] = int(labels[i])
    print(f"{len(org_z)} organoids, {Z.shape[0]} organoid-day embeddings, dim {Z.shape[1]}")

    import csv
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"{args.out_prefix}_results.csv"
    w = csv.writer(open(csv_path, "w", newline=""))
    w.writerow(["cutoff", "n", "featureset", "balacc", "balacc_lo", "balacc_hi",
                "auc", "auc_lo", "auc_hi", "vs_state_balacc_delta", "frac_reps_beat_state"])

    curve = {f: {"bal": [], "lo": [], "hi": [], "auc": []} for f in FEATURE_SETS}
    cutoffs_used = []
    print(f"\n{'day':>5} {'n':>4}  " + "  ".join(f"{f:>13}" for f in FEATURE_SETS) + "   (balanced acc)")
    for c in CUTOFFS:
        X, y, keep = build_features(c, None, org_z, org_label)
        if len(y) < 12 or min(np.bincount(y)) < 3:
            continue
        bal, auc = cv_scores(X, y)
        cutoffs_used.append(c)
        state_reps = np.array(bal["state"])
        line = f"{c:>5} {len(y):>4}  "
        for f in FEATURE_SETS:
            bm, blo, bhi = summarize(bal[f]); am, alo, ahi = summarize(auc[f])
            delta = float(np.mean(np.array(bal[f]) - state_reps))
            frac = float(np.mean(np.array(bal[f]) > state_reps))
            w.writerow([c, len(y), f, round(bm, 4), round(blo, 4), round(bhi, 4),
                        round(am, 4), round(alo, 4), round(ahi, 4), round(delta, 4), round(frac, 3)])
            curve[f]["bal"].append(bm); curve[f]["lo"].append(blo); curve[f]["hi"].append(bhi); curve[f]["auc"].append(am)
            line += f"{bm:>13.2f}  "
        print(line)

    # permutation floor at the last cutoff (sanity: should sit ~0.5)
    Xp, yp, _ = build_features(CUTOFFS[-1], None, org_z, org_label)
    balp, _ = cv_scores(Xp, yp, permute=True)
    print(f"\nlabel-permutation floor (state, day {CUTOFFS[-1]}): "
          f"{np.mean(balp['state']):.2f}  (should be ~0.50)")

    # ---- plot ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"state": "#5B8A8C", "change": "#B4553E", "state+change": "#D9A441", "trajectory": "#3A3A38"}
    marks = {"state": "o", "change": "s", "state+change": "^", "trajectory": "D"}
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.family": "DejaVu Sans", "axes.edgecolor": "#3A3A38",
                         "axes.linewidth": 1.6, "xtick.direction": "in", "ytick.direction": "in"})
    fig, a = plt.subplots(figsize=(7.4, 5))
    x = cutoffs_used
    for f in FEATURE_SETS:
        a.plot(x, curve[f]["bal"], marker=marks[f], color=colors[f], ms=6, lw=2.2, label=f)
        a.fill_between(x, curve[f]["lo"], curve[f]["hi"], color=colors[f], alpha=0.12)
    a.axhline(0.5, ls="--", lw=1.1, color="0.6")
    a.set_xlabel("cutoff day"); a.set_ylabel("balanced accuracy (CV)")
    a.set_title("Does trajectory beat the latest image?")
    a.legend(frameon=False, fontsize=11, loc="upper left")
    for e in ["png", "pdf"]:
        fig.savefig(args.outdir / f"{args.out_prefix}_curve.{e}", dpi=200, bbox_inches="tight")
    print(f"\nWrote {csv_path}\nWrote {args.outdir}/{args.out_prefix}_curve.png")


if __name__ == "__main__":
    main()

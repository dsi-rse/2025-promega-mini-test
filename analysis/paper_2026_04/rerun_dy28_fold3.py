#!/usr/bin/env python3
"""Rerun Dy28 fold 3 with multiple random seeds to test seed sensitivity.

Fold 3 (index 2) scored 0.50 (trivial predictor) in the main kfold run.
This script reruns that fold with N_TRIALS different seeds and reports
the range of balanced accuracies.

Usage:
    make run ARGS="-m analysis.paper_2026_04.rerun_dy28_fold3"
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from pipeline.data_loader import (
    DAY_ORDER,
    LABEL_TO_INT,
    OrganoidDataset,
    filters_for_mode,
)

ALL_DATA_PATH = "data/all_data.json"
from pipeline.splits import Splits

from .perday_image_kfold import (
    SEED, N_FOLDS, DEVICE, BATCH_SIZE,
    _build_transforms, _get_image_paths, _train_one_fold,
    OrganoidImageDataset, set_seed,
)

DAY = "Dy28"
TARGET_FOLD_IDX = 2      # fold 3 (0-indexed) — the one that collapsed
N_TRIALS = 10
TRIAL_SEEDS = [SEED + TARGET_FOLD_IDX * 97 + i * 37 for i in range(N_TRIALS)]


def eval_fold(model, test_paths, test_labels):
    test_loader = DataLoader(
        OrganoidImageDataset(test_paths, test_labels, _build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model.eval()
    probs, preds, trues = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            p = torch.sigmoid(model(imgs))
            probs.extend(p.cpu().numpy())
            preds.extend((p >= 0.5).long().cpu().numpy())
            trues.extend(lbls.numpy().astype(int))
    return np.array(trues), np.array(preds), np.array(probs)


def main():
    set_seed(SEED)
    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode("series_idor"))

    all_org_ids = [oid for oid in ds.organoid_ids
                   if ds.organoid_label(oid) in LABEL_TO_INT]
    all_labels  = np.array([LABEL_TO_INT[ds.organoid_label(oid)] for oid in all_org_ids])

    # Reproduce the exact same fold splits as the main run
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(outer_cv.split(all_org_ids, all_labels))
    tr_idx, te_idx = splits[TARGET_FOLD_IDX]

    tr_org_ids = [all_org_ids[i] for i in tr_idx]
    tr_labels  = all_labels[tr_idx]
    te_org_ids = [all_org_ids[i] for i in te_idx]

    print(f"Rerunning {DAY} fold {TARGET_FOLD_IDX+1} (index {TARGET_FOLD_IDX}) "
          f"with {N_TRIALS} different seeds")
    n_pos_te = sum(LABEL_TO_INT[ds.organoid_label(o)] for o in te_org_ids
                   if ds.organoid_label(o) in LABEL_TO_INT)
    print(f"Test fold: {len(te_org_ids)} organoids  ({n_pos_te} NAcc, {len(te_org_ids)-n_pos_te} Acc)\n")

    results = []
    for trial_i, trial_seed in enumerate(TRIAL_SEEDS):
        # Internal val split uses the trial seed
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=trial_seed)
        inner_tr_idx, inner_val_idx = next(sss.split(tr_org_ids, tr_labels))
        inner_tr_org  = [tr_org_ids[i] for i in inner_tr_idx]
        inner_val_org = [tr_org_ids[i] for i in inner_val_idx]

        train_paths, train_labels_list = _get_image_paths(ds, inner_tr_org,  DAY, "cm_image")
        val_paths,   val_labels_list   = _get_image_paths(ds, inner_val_org, DAY, "cm_image")
        test_paths,  test_labels_list  = _get_image_paths(ds, te_org_ids,    DAY, "cm_image")

        n_pos_tr = sum(train_labels_list)
        print(f"  Trial {trial_i+1:2d}  seed={trial_seed}  "
              f"train={len(train_paths)} ({n_pos_tr} NAcc)  val={len(val_paths)}", end="  ")

        model, best_val_acc = _train_one_fold(
            train_paths, train_labels_list, val_paths, val_labels_list,
            DAY, trial_seed, verbose=False, augment=True,
        )

        trues, preds, probs = eval_fold(model, test_paths, test_labels_list)
        ba = balanced_accuracy_score(trues, preds)
        tp = int(((trues == 1) & (preds == 1)).sum())
        fn = int(((trues == 1) & (preds == 0)).sum())
        fp = int(((trues == 0) & (preds == 1)).sum())
        tn = int(((trues == 0) & (preds == 0)).sum())
        results.append(ba)
        print(f"BA={ba:.3f}  TP={tp} FN={fn} FP={fp} TN={tn}  val_acc={best_val_acc:.3f}")

    print(f"\n{'='*50}")
    print(f"Fold {TARGET_FOLD_IDX+1} results across {N_TRIALS} seeds:")
    print(f"  BAs: {[round(x, 3) for x in results]}")
    print(f"  Mean: {np.mean(results):.3f}  Std: {np.std(results):.3f}")
    print(f"  Min: {min(results):.3f}  Max: {max(results):.3f}")
    print(f"  Collapsed (BA=0.5): {sum(1 for x in results if x <= 0.5)}/{N_TRIALS}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Late-fusion combined model: metabolite + morphology + image, 5-fold CV.

All three modalities use the same fold splits (StratifiedKFold on the 132
series_idor organoids). For each fold × day:
  - Metabolite: LightGBM with inner 3-fold GridSearchCV
  - Morphology:  LightGBM with inner 3-fold GridSearchCV
  - Image:       EfficientNet-B0 (same architecture as perday_image_kfold.py)

Combination strategies (for every 2- and 3-modality pair):
  - mean_prob:     mean(available probs) >= 0.5
  - majority_vote: >= 2-of-3 (or >=1-of-1/2 for pairs) predict NAcc

Outputs:
  analysis_output/images/combined_results_kfold_series_idor.json
  figures/combined_kfold_balanced_accuracy_series_idor.png

Usage:
    DAYS="Dy30" sbatch submit_combined_kfold.slurm   # smoke test
    sbatch submit_combined_kfold.slurm
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    IMAGE_MODE_TO_PATH_KEY,
    LABEL_TO_INT,
    OrganoidDataset,
    filters_for_mode,
    idor_ba1_ba2_filters,
    require_complete_series,
)
from pipeline.splits import Splits

from .common import (
    ForegroundColorJitter,
    compute_classification_metrics,
    plot_balanced_accuracy_by_day,
)

# Import feature-extraction helpers from sibling modules
from .metabolites_train import _features_for_day_all as _met_features_all
from analysis.multimodel.morphology_train import (
    _features_for_day as _morph_features_all,
    _load_morph_df,
)

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED        = 1
N_FOLDS     = 5
FILTER_MODE = "series_idor"
INPUT_MODE  = "cm_image"
ALL_DATA_PATH  = "data/all_data.json"
MORPH_CSV_PATH = "data/normalized/CONC_data_organoides_residualized_final.csv"
OUTPUT_DIR  = Path("analysis_output") / "images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── image model constants ──────────────────────────────────────────────────────
BATCH_SIZE   = 16
MAX_EPOCHS   = 100
LR_HEAD      = 5e-4
LR_BACKBONE  = 5e-5
UNFREEZE_AFTER = 4
PATIENCE     = 15
GRAD_CLIP    = 1.0
IMG_HEIGHT   = 384
IMG_WIDTH    = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_FILL         = [178, 178, 178]   # actual background fill of cm_image_abs (verified from data)
_BOUNDARY_DAYS = {"Dy28", "Dy30"}

# ── LGBM grid ──────────────────────────────────────────────────────────────────
LGBM_PARAM_GRID = {
    "max_depth":         [3, 6],
    "num_leaves":        [15, 31],
    "min_child_samples": [5, 10],
    "learning_rate":     [0.05, 0.1],
    "n_estimators":      [100, 300],
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _oid_index(feature_ids: List[str], target_ids: List[str]) -> List[int]:
    """Return row indices in feature_ids for each id in target_ids (skip missing)."""
    lookup = {oid: i for i, oid in enumerate(feature_ids)}
    return [lookup[o] for o in target_ids if o in lookup]


def _filter_fold(X, y, all_ids, fold_ids):
    idx = _oid_index(all_ids, fold_ids)
    if len(idx) == 0:
        return np.empty((0, X.shape[1] if X.ndim == 2 else 0)), np.empty(0), []
    valid_ids = [all_ids[i] for i in idx]
    return X[idx], y[idx], valid_ids


# ══════════════════════════════════════════════════════════════════════════════
# LGBM fold trainer (metabolite & morphology)
# ══════════════════════════════════════════════════════════════════════════════

def _train_lgbm_fold(X_tr, y_tr, X_te, fold_seed: int) -> Optional[np.ndarray]:
    """Train LightGBM with inner 3-fold CV; return test probabilities or None."""
    if len(X_tr) == 0 or len(X_te) == 0:
        return None
    if len(np.unique(y_tr)) < 2:
        return None
    spw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
    model = lgb.LGBMClassifier(
        objective="binary", scale_pos_weight=spw,
        random_state=fold_seed, verbosity=-1, n_jobs=1,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    grid = GridSearchCV(model, LGBM_PARAM_GRID, cv=inner_cv,
                        scoring="f1", n_jobs=-1, refit=True)
    grid.fit(X_tr, y_tr)
    return grid.predict_proba(X_te)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# EfficientNet fold trainer (image)
# ══════════════════════════════════════════════════════════════════════════════

class _ImgDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:   img = Image.open(self.paths[idx]).convert("RGB")
        except Exception: img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (128, 128, 128))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


class _EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 1),
        )
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_last_blocks(self):
        for block in list(self.backbone.blocks)[-2:]:
            for p in block.parameters(): p.requires_grad = True
        for attr in ("conv_head", "bn2"):
            if hasattr(self.backbone, attr):
                for p in getattr(self.backbone, attr).parameters(): p.requires_grad = True
    def forward(self, x): return self.head(self.backbone(x)).squeeze(-1)


def _build_transforms(train: bool, translate=(0.1, 0.1), degrees=180):
    base = [T.Resize((IMG_HEIGHT, IMG_WIDTH))]
    if train:
        base.append(T.RandomHorizontalFlip(0.5))
        if degrees > 0 or translate is not None:
            base.append(T.RandomAffine(degrees=degrees, translate=translate, fill=_FILL))
        base.append(ForegroundColorJitter(brightness=0.3, contrast=0.3,
                                          saturation=0.2, hue=0.05))
    base += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(base)


def _get_img_paths(ds, org_ids, day, input_mode=INPUT_MODE):
    path_key = IMAGE_MODE_TO_PATH_KEY.get(input_mode, input_mode)
    paths, labels, valid_ids = [], [], []
    for oid in org_ids:
        rec = ds.get_record(oid, day)
        if rec is None: continue
        imgs = rec.get("images", {})
        if isinstance(path_key, tuple):
            parent, key = path_key
            path = (imgs.get(parent) or {}).get(key)
        else:
            path = imgs.get(path_key)
        if not path: continue
        lbl = ds.organoid_label(oid)
        if lbl not in LABEL_TO_INT: continue
        paths.append(path); labels.append(LABEL_TO_INT[lbl]); valid_ids.append(oid)
    return paths, labels, valid_ids


def _train_efficientnet_fold(
    train_paths, train_labels, val_paths, val_labels,
    test_paths, test_labels, day: str, fold_seed: int,
) -> Optional[np.ndarray]:
    n_pos = sum(train_labels); n_neg = len(train_labels) - n_pos
    if n_pos == 0 or n_neg == 0 or len(test_paths) == 0:
        return None
    set_seed(fold_seed)
    degrees = 0 if day in _BOUNDARY_DAYS else 180
    train_loader = DataLoader(
        _ImgDataset(train_paths, train_labels, _build_transforms(True, (0.1, 0.1), degrees)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        _ImgDataset(val_paths, val_labels, _build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        _ImgDataset(test_paths, test_labels, _build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = _EfficientNet().to(DEVICE)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam([p for p in model.head.parameters() if p.requires_grad], lr=LR_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_val_acc, best_state, patience_ctr, unfrozen = 0.0, None, 0, False

    for epoch in range(MAX_EPOCHS):
        if epoch == UNFREEZE_AFTER and not unfrozen:
            model.unfreeze_last_blocks(); unfrozen = True
            optimizer = optim.Adam(model.parameters(), lr=LR_BACKBONE)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        model.eval()
        vp, vt, vl_sum, vn = [], [], 0.0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                out = model(imgs.to(DEVICE))
                vl_sum += criterion(out, lbls.to(DEVICE)).item() * len(lbls)
                vn += len(lbls)
                vp.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy())
                vt.extend(lbls.numpy().astype(int))
        val_acc = accuracy_score(vt, vp) if vt else 0.0
        scheduler.step(vl_sum / max(vn, 1))
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            probs.extend(torch.sigmoid(model(imgs.to(DEVICE))).cpu().numpy())
    return np.array(probs)


# ══════════════════════════════════════════════════════════════════════════════
# Combination logic
# ══════════════════════════════════════════════════════════════════════════════

COMBO_KEYS = ["met", "morph", "img",
              "met+morph", "met+img", "morph+img", "met+morph+img"]

MAJORITY_THRESHOLD = {
    "met+morph": 2, "met+img": 2, "morph+img": 2, "met+morph+img": 2,
}


def _combine(prob_map: Dict[str, np.ndarray], strategy: str) -> np.ndarray:
    """Given {mod: prob_array} of equal length, return combined prob or binary preds."""
    probs = np.stack(list(prob_map.values()), axis=1)  # (n, k)
    if strategy == "mean_prob":
        return probs.mean(axis=1)
    else:  # majority_vote: >= ceil(k/2)
        k = probs.shape[1]
        threshold = (k // 2) + 1
        return (probs >= 0.5).sum(axis=1) >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# Main per-day k-fold loop
# ══════════════════════════════════════════════════════════════════════════════

def run_day(
    day: str,
    ds: OrganoidDataset,
    morph_df: pd.DataFrame,
    all_org_ids: List[str],
    all_labels: np.ndarray,
    n_folds: int,
    n_repeats: int,
    verbose: bool,
) -> Optional[Dict]:
    """Run n_repeats × n_folds CV for one day; 3 met variants + morph + img."""

    # ── Pre-compute feature arrays once per day (3 met variants) ──
    X_met_nan,       y_met_nan,       _, met_ids_nan       = _met_features_all(ds, day, malate_mode="nan")
    X_met_raw,       y_met_raw,       _, met_ids_raw       = _met_features_all(ds, day, malate_mode="raw")
    X_met_no_malate, y_met_no_malate, _, met_ids_no_malate = _met_features_all(ds, day, malate_mode="drop")
    X_morph, y_morph, _, morph_ids = _morph_features_all(ds, morph_df, day)

    met_variants = {
        "met_nan":       (X_met_nan,       y_met_nan,       met_ids_nan),
        "met_raw":       (X_met_raw,       y_met_raw,       met_ids_raw),
        "met_no_malate": (X_met_no_malate, y_met_no_malate, met_ids_no_malate),
    }

    if all(len(X) == 0 for X, _, _ in met_variants.values()) and len(X_morph) == 0:
        if verbose: print(f"  {day}: no metabolite or morphology data, skipping")
        return None

    # ── OOF BA, confusion-matrix and detail storage across repeats ──
    mod_keys = ["met_nan", "met_raw", "met_no_malate", "morph", "img"]
    repeat_oof_bas: Dict[str, List[float]] = {k: [] for k in mod_keys}
    repeat_cms:     Dict[str, List]        = {k: [] for k in mod_keys}  # list of [[TN,FP],[FN,TP]]
    repeat_details: List[Dict]             = []  # one entry per repeat: fold_assignments + oof_probs

    fusion_keys: List[str] = []
    for met_k in ["met_nan", "met_raw", "met_no_malate"]:
        for combo in [[met_k, "morph"], [met_k, "img"], [met_k, "morph", "img"]]:
            for strategy in ["mean_prob", "majority_vote"]:
                fusion_keys.append(f"{'+'.join(combo)}_{strategy}")
    fusion_keys += ["morph+img_mean_prob", "morph+img_majority_vote"]
    repeat_fusion_bas: Dict[str, List[float]] = {k: [] for k in fusion_keys}

    for rep in range(n_repeats):
        rep_seed = SEED + rep * 1000
        outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)

        oof: Dict[str, np.ndarray] = {k: np.full(len(all_org_ids), np.nan) for k in mod_keys}

        # Record which fold each organoid is assigned to for this repeat
        fold_assignments = np.full(len(all_org_ids), -1, dtype=int)

        for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(all_org_ids, all_labels)):
            fold_assignments[te_idx] = fold_i
            fold_seed = rep_seed + fold_i * 97
            tr_oids = [all_org_ids[i] for i in tr_idx]
            te_oids = [all_org_ids[i] for i in te_idx]

            sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=fold_seed)
            inner_tr_idx, inner_val_idx = next(sss.split(tr_oids, all_labels[tr_idx]))
            inner_tr_oids  = [tr_oids[i] for i in inner_tr_idx]
            inner_val_oids = [tr_oids[i] for i in inner_val_idx]

            if verbose:
                print(f"  Rep {rep+1}/{n_repeats} Fold {fold_i+1}/{n_folds}  "
                      f"train={len(inner_tr_oids)}  val={len(inner_val_oids)}  test={len(te_oids)}")

            fold_probs: Dict[str, np.ndarray] = {}
            fold_te_ids: Dict[str, List[str]] = {}

            # ── 3 Metabolite variants ───────────────────────────────────────
            for met_k, (X_met, y_met, met_ids) in met_variants.items():
                X_tr_m, y_tr_m, _ = _filter_fold(X_met, y_met, met_ids, inner_tr_oids)
                X_te_m, y_te_m, valid_te_m = _filter_fold(X_met, y_met, met_ids, te_oids)
                if len(X_tr_m) > 0 and len(X_te_m) > 0:
                    p = _train_lgbm_fold(X_tr_m, y_tr_m, X_te_m, fold_seed)
                    if p is not None:
                        fold_probs[met_k] = p
                        fold_te_ids[met_k] = valid_te_m
                        for oid, prob in zip(valid_te_m, p):
                            oof[met_k][all_org_ids.index(oid)] = prob
                        if verbose:
                            ba = balanced_accuracy_score(y_te_m, (p >= 0.5).astype(int))
                            print(f"    {met_k} bal_acc={ba:.3f}")

            # ── Morphology ──────────────────────────────────────────────────
            X_tr_o, y_tr_o, _ = _filter_fold(X_morph, y_morph, morph_ids, inner_tr_oids)
            X_te_o, y_te_o, valid_te_o = _filter_fold(X_morph, y_morph, morph_ids, te_oids)
            if len(X_tr_o) > 0 and len(X_te_o) > 0:
                p = _train_lgbm_fold(X_tr_o, y_tr_o, X_te_o, fold_seed)
                if p is not None:
                    fold_probs["morph"] = p
                    fold_te_ids["morph"] = valid_te_o
                    for oid, prob in zip(valid_te_o, p):
                        oof["morph"][all_org_ids.index(oid)] = prob
                    if verbose:
                        ba = balanced_accuracy_score(y_te_o, (p >= 0.5).astype(int))
                        print(f"    morph bal_acc={ba:.3f}")

            # ── Image ────────────────────────────────────────────────────────
            train_paths, train_labels, _ = _get_img_paths(ds, inner_tr_oids, day)
            val_paths,   val_labels,   _ = _get_img_paths(ds, inner_val_oids, day)
            test_paths,  test_labels,  valid_te_i = _get_img_paths(ds, te_oids, day)
            if len(train_paths) > 0 and len(test_paths) > 0:
                p = _train_efficientnet_fold(
                    train_paths, train_labels, val_paths, val_labels,
                    test_paths, test_labels, day, fold_seed,
                )
                if p is not None:
                    fold_probs["img"] = p
                    fold_te_ids["img"] = valid_te_i
                    for oid, prob in zip(valid_te_i, p):
                        oof["img"][all_org_ids.index(oid)] = prob
                    if verbose:
                        ba = balanced_accuracy_score(test_labels, (p >= 0.5).astype(int))
                        print(f"    img  bal_acc={ba:.3f}")

        # ── Per-repeat OOF BAs + confusion matrices (single modalities) ───────
        for k in mod_keys:
            valid = ~np.isnan(oof[k])
            if valid.sum() < 2: continue
            yt = all_labels[valid]; yp = (oof[k][valid] >= 0.5).astype(int)
            if len(np.unique(yt)) < 2: continue
            repeat_oof_bas[k].append(float(balanced_accuracy_score(yt, yp)))
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            repeat_cms[k].append(cm.tolist())

        # ── Per-repeat detail record (probs + fold assignments) ─────────────
        detail: Dict = {
            "seed":             rep_seed,
            "org_ids":          all_org_ids,
            "true_labels":      all_labels.tolist(),
            "fold_assignments": fold_assignments.tolist(),
            "oof_probs": {
                k: [None if np.isnan(v) else round(float(v), 6)
                    for v in oof[k]]
                for k in mod_keys
            },
            "oof_preds": {
                k: [None if np.isnan(oof[k][i]) else int(oof[k][i] >= 0.5)
                    for i in range(len(all_org_ids))]
                for k in mod_keys
            },
        }
        repeat_details.append(detail)

        # ── Per-repeat fusion BAs ────────────────────────────────────────────
        for met_k in ["met_nan", "met_raw", "met_no_malate"]:
            for combo_mods in [[met_k, "morph"], [met_k, "img"], [met_k, "morph", "img"]]:
                combo_name = "+".join(combo_mods)
                valid = np.ones(len(all_org_ids), dtype=bool)
                for m in combo_mods:
                    valid &= ~np.isnan(oof[m])
                if valid.sum() < 2: continue
                yt = all_labels[valid]
                if len(np.unique(yt)) < 2: continue
                for strategy in ["mean_prob", "majority_vote"]:
                    prob_map = {m: oof[m][valid] for m in combo_mods}
                    combined = _combine(prob_map, strategy)
                    preds = (combined.astype(int) if strategy == "majority_vote"
                             else (combined >= 0.5).astype(int))
                    fkey = f"{combo_name}_{strategy}"
                    repeat_fusion_bas[fkey].append(float(balanced_accuracy_score(yt, preds)))

        for strategy in ["mean_prob", "majority_vote"]:
            valid = ~np.isnan(oof["morph"]) & ~np.isnan(oof["img"])
            if valid.sum() < 2: continue
            yt = all_labels[valid]
            if len(np.unique(yt)) < 2: continue
            prob_map = {"morph": oof["morph"][valid], "img": oof["img"][valid]}
            combined = _combine(prob_map, strategy)
            preds = (combined.astype(int) if strategy == "majority_vote"
                     else (combined >= 0.5).astype(int))
            repeat_fusion_bas[f"morph+img_{strategy}"].append(float(balanced_accuracy_score(yt, preds)))

    # ── Aggregate over repeats ───────────────────────────────────────────────
    results: Dict = {}
    for k in mod_keys:
        bas = repeat_oof_bas[k]
        if not bas: continue
        results[k] = {
            "balanced_accuracy_mean":       float(np.mean(bas)),
            "balanced_accuracy_std":        float(np.std(bas)),
            "n_repeats":                    len(bas),
            "n_folds":                      n_folds,
            "repeat_balanced_accuracies":   bas,
            "repeat_confusion_matrices":    repeat_cms[k],
        }
    for fk in fusion_keys:
        bas = repeat_fusion_bas.get(fk, [])
        if not bas: continue
        results[fk] = {
            "balanced_accuracy_mean":       float(np.mean(bas)),
            "balanced_accuracy_std":        float(np.std(bas)),
            "n_repeats":                    len(bas),
            "n_folds":                      n_folds,
            "repeat_balanced_accuracies":   bas,
        }

    if repeat_details:
        results["repeat_details"] = repeat_details

    return results if results else None


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="Number of repeated k-fold runs (different seeds)")
    parser.add_argument("--include-stitched", action="store_true",
                        help="Include stitched organoids (n=139, no Splits canonical)")
    args = parser.parse_args()

    set_seed(SEED)
    if args.include_stitched:
        ds = OrganoidDataset(ALL_DATA_PATH, splits=None,
                             filters=[*idor_ba1_ba2_filters(),
                                      require_complete_series(drop_stitched=False)])
    else:
        ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                             filters=filters_for_mode(FILTER_MODE))
    morph_df = _load_morph_df()
    try:
        print(ds.summary())
    except RuntimeError:
        print(f"OrganoidDataset: {len(list(ds.organoid_ids))} organoids (no splits assigned)")
    print(f"Device:   {DEVICE}")
    print(f"Folds:    {args.n_folds}")
    print(f"Repeats:  {args.n_repeats}")

    all_org_ids = [o for o in ds.organoid_ids if ds.organoid_label(o) in LABEL_TO_INT]
    all_labels  = np.array([LABEL_TO_INT[ds.organoid_label(o)] for o in all_org_ids])
    print(f"Organoids: {len(all_org_ids)}  ({all_labels.sum()} NAcc, {(all_labels==0).sum()} Acc)\n")

    days_to_run = args.days if args.days else DAY_ORDER
    all_results: Dict[str, Dict] = {}   # day -> {combo_key -> metrics}

    for day in days_to_run:
        if day not in ds.days:
            print(f"Skipping {day} (no data)")
            continue
        print(f"\n{'='*55}\nCombined — {day}\n{'='*55}")
        day_res = run_day(day, ds, morph_df, all_org_ids, all_labels,
                          n_folds=args.n_folds, n_repeats=args.n_repeats, verbose=True)
        if day_res:
            all_results[day] = day_res

    # ── Save JSON ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_139" if args.include_stitched else ""
    # Per-day output when a single day is requested (parallel jobs mode)
    if args.days and len(args.days) == 1:
        day_tag = args.days[0]
        out_path = OUTPUT_DIR / f"combined_results_kfold_series_idor{suffix}_{day_tag}.json"
    else:
        out_path = OUTPUT_DIR / f"combined_results_kfold_series_idor{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    COMBO_DISPLAY = [
        "met_nan", "met_raw", "met_no_malate", "morph", "img",
        "met_nan+morph+img_mean_prob", "met_nan+morph+img_majority_vote",
        "met_raw+morph+img_mean_prob",
        "met_no_malate+morph+img_mean_prob",
        "morph+img_mean_prob",
    ]
    header = f"{'Day':<10}" + "".join(f"{k:>30}" for k in COMBO_DISPLAY)
    print(f"\n{'='*len(header)}")
    print(header)
    print('='*len(header))
    for day in DAY_ORDER:
        dr = all_results.get(day)
        if not dr: continue
        row = f"{day:<10}"
        for k in COMBO_DISPLAY:
            m = dr.get(k)
            ba = m.get("balanced_accuracy_mean", float("nan")) if m else float("nan")
            std = m.get("balanced_accuracy_std", float("nan")) if m else float("nan")
            cell = f"{ba:.3f}±{std:.3f}" if not (ba != ba) else "—"
            row += f"{cell:>30}"
        print(row)

    # ── Figure: met_nan+morph+img vs single modalities ───────────────────────
    if all_results:
        import shutil
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        key_style = {
            "met_nan":                        ("Met (nan floor)",    "#2ca02c", "o", "-"),
            "met_raw":                        ("Met (raw)",          "#98df8a", "v", "--"),
            "met_no_malate":                  ("Met (no malate)",    "#17becf", "^", ":"),
            "morph":                          ("Morphology",         "#9467bd", "s", "-"),
            "img":                            ("Image",              "#1f77b4", "^", "-"),
            "met_nan+morph+img_mean_prob":    ("All3/nan (mean)",    "#d62728", "D", "-"),
            "met_nan+morph+img_majority_vote":("All3/nan (vote)",    "#ff7f0e", "P", "--"),
        }
        series = {}
        for k, (label, color, marker, ls) in key_style.items():
            day_metrics = {d: all_results[d][k]
                           for d in DAY_ORDER if d in all_results and k in all_results[d]}
            if day_metrics:
                series[label] = day_metrics

        n_str = "139" if args.include_stitched else "132"
        cv_str = f"{args.n_repeats}×{args.n_folds}-fold"
        fig_name = f"combined_kfold_balanced_accuracy_series_idor{'_139' if args.include_stitched else ''}.png"
        plot_balanced_accuracy_by_day(
            series,
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / fig_name,
            title=f"Combined Model: Balanced Accuracy by Day (series_idor, n={n_str}, {cv_str} CV)",
            style_overrides={
                label: {"color": color, "marker": marker, "linestyle": ls}
                for _, (label, color, marker, ls) in key_style.items()
            },
        )
        repo_fig = Path("figures") / fig_name
        repo_fig.parent.mkdir(exist_ok=True)
        shutil.copy(FIGURE_DIR / fig_name, repo_fig)
        print(f"Copied figure to {repo_fig}")


if __name__ == "__main__":
    main()

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
_FILL         = [int(v * 255) for v in IMAGENET_MEAN]
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
    translate = None if day in _BOUNDARY_DAYS else (0.1, 0.1)
    degrees   = 0   if day in _BOUNDARY_DAYS else 180
    train_loader = DataLoader(
        _ImgDataset(train_paths, train_labels, _build_transforms(True, translate, degrees)),
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
    verbose: bool,
) -> Optional[Dict]:
    """Run n_folds CV for one day across all modalities; return results dict."""

    # ── Pre-compute full-day feature arrays (once per day) ──
    X_met, y_met, _, met_ids = _met_features_all(ds, day)
    X_morph, y_morph, _, morph_ids = _morph_features_all(ds, morph_df, day)

    if len(X_met) == 0 and len(X_morph) == 0:
        if verbose: print(f"  {day}: no metabolite or morphology data, skipping")
        return None

    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # OOF probability stores: {mod: array of nan, filled per fold}
    oof = {k: np.full(len(all_org_ids), np.nan) for k in ["met", "morph", "img"]}
    fold_results = []   # per-fold metrics for each combo

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(all_org_ids, all_labels)):
        fold_seed = SEED + fold_i * 97
        tr_oids = [all_org_ids[i] for i in tr_idx]
        te_oids = [all_org_ids[i] for i in te_idx]

        # Inner val split (15 % of train) for EfficientNet early stopping
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=fold_seed)
        inner_tr_idx, inner_val_idx = next(sss.split(tr_oids, all_labels[tr_idx]))
        inner_tr_oids  = [tr_oids[i] for i in inner_tr_idx]
        inner_val_oids = [tr_oids[i] for i in inner_val_idx]

        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}  "
                  f"train={len(inner_tr_oids)}  val={len(inner_val_oids)}  test={len(te_oids)}")

        fold_probs: Dict[str, np.ndarray] = {}
        fold_te_ids: Dict[str, List[str]] = {}

        # ── Metabolite ──────────────────────────────────────────────────────
        X_tr_m, y_tr_m, _ = _filter_fold(X_met, y_met, met_ids, inner_tr_oids)
        X_te_m, y_te_m, valid_te_m = _filter_fold(X_met, y_met, met_ids, te_oids)
        if len(X_tr_m) > 0 and len(X_te_m) > 0:
            p = _train_lgbm_fold(X_tr_m, y_tr_m, X_te_m, fold_seed)
            if p is not None:
                fold_probs["met"] = p
                fold_te_ids["met"] = valid_te_m
                for oid, prob in zip(valid_te_m, p):
                    oof["met"][all_org_ids.index(oid)] = prob
                ba = balanced_accuracy_score(y_te_m, (p >= 0.5).astype(int))
                if verbose: print(f"    met  bal_acc={ba:.3f}")

        # ── Morphology ──────────────────────────────────────────────────────
        X_tr_o, y_tr_o, _ = _filter_fold(X_morph, y_morph, morph_ids, inner_tr_oids)
        X_te_o, y_te_o, valid_te_o = _filter_fold(X_morph, y_morph, morph_ids, te_oids)
        if len(X_tr_o) > 0 and len(X_te_o) > 0:
            p = _train_lgbm_fold(X_tr_o, y_tr_o, X_te_o, fold_seed)
            if p is not None:
                fold_probs["morph"] = p
                fold_te_ids["morph"] = valid_te_o
                for oid, prob in zip(valid_te_o, p):
                    oof["morph"][all_org_ids.index(oid)] = prob
                ba = balanced_accuracy_score(y_te_o, (p >= 0.5).astype(int))
                if verbose: print(f"    morph bal_acc={ba:.3f}")

        # ── Image ────────────────────────────────────────────────────────────
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
                ba = balanced_accuracy_score(test_labels, (p >= 0.5).astype(int))
                if verbose: print(f"    img  bal_acc={ba:.3f}")

        # ── Fold-level combined metrics ──────────────────────────────────────
        fold_combo: Dict = {}
        combo_defs = {
            "met+morph":    ["met", "morph"],
            "met+img":      ["met", "img"],
            "morph+img":    ["morph", "img"],
            "met+morph+img":["met", "morph", "img"],
        }
        for combo_name, mods in combo_defs.items():
            # Organoids present in ALL listed modalities for this fold
            common_ids = set(fold_te_ids.get(mods[0], []))
            for m in mods[1:]:
                common_ids &= set(fold_te_ids.get(m, []))
            if not common_ids:
                continue
            common_ids = [o for o in te_oids if o in common_ids]
            for strategy in ("mean_prob", "majority_vote"):
                prob_map = {}
                true_labels = None
                for m in mods:
                    te_ids_m = fold_te_ids[m]
                    id_to_prob = dict(zip(te_ids_m, fold_probs[m]))
                    prob_map[m] = np.array([id_to_prob[o] for o in common_ids])
                    if true_labels is None:
                        # recover true labels
                        lbl_map = {}
                        for oid in common_ids:
                            lbl = ds.organoid_label(oid)
                            if lbl in LABEL_TO_INT:
                                lbl_map[oid] = LABEL_TO_INT[lbl]
                        true_labels = np.array([lbl_map[o] for o in common_ids if o in lbl_map])
                        common_ids_valid = [o for o in common_ids if o in lbl_map]
                        for m2 in mods:
                            id_to_prob = dict(zip(fold_te_ids[m2], fold_probs[m2]))
                            prob_map[m2] = np.array([id_to_prob[o] for o in common_ids_valid])

                combined = _combine(prob_map, strategy)
                if strategy == "majority_vote":
                    preds = combined.astype(int)
                    ba = balanced_accuracy_score(true_labels, preds) if len(true_labels) > 0 else 0.0
                else:
                    preds = (combined >= 0.5).astype(int)
                    ba = balanced_accuracy_score(true_labels, preds) if len(true_labels) > 0 else 0.0
                fold_combo[f"{combo_name}_{strategy}"] = ba
        fold_results.append(fold_combo)

    # ── Aggregate OOF across all folds ──────────────────────────────────────
    results: Dict = {}

    # Single-modality OOF metrics
    for mod in ["met", "morph", "img"]:
        valid = ~np.isnan(oof[mod])
        if valid.sum() == 0: continue
        yt = all_labels[valid]; yp_prob = oof[mod][valid]
        yp = (yp_prob >= 0.5).astype(int)
        if len(np.unique(yt)) < 2: continue
        m = compute_classification_metrics(yt, yp, yp_prob)
        fold_bas = [f.get(mod, np.nan) for f in fold_results
                    if not np.isnan(f.get(mod, np.nan))]
        # fold_bas from individual modality not tracked above; use OOF only
        results[mod] = m

    # Combined OOF metrics
    combo_defs = {
        "met+morph":    ["met", "morph"],
        "met+img":      ["met", "img"],
        "morph+img":    ["morph", "img"],
        "met+morph+img":["met", "morph", "img"],
    }
    for combo_name, mods in combo_defs.items():
        # Organoids where ALL mods have OOF probs
        valid = np.ones(len(all_org_ids), dtype=bool)
        for m in mods:
            valid &= ~np.isnan(oof[m])
        if valid.sum() == 0: continue
        yt = all_labels[valid]
        if len(np.unique(yt)) < 2: continue

        for strategy in ("mean_prob", "majority_vote"):
            prob_map = {m: oof[m][valid] for m in mods}
            combined = _combine(prob_map, strategy)
            if strategy == "majority_vote":
                preds = combined.astype(int)
            else:
                preds = (combined >= 0.5).astype(int)
            key = f"{combo_name}_{strategy}"
            # Use mean-prob array as proxy probability for AUC
            prob_for_metrics = np.stack([oof[m][valid] for m in mods], axis=1).mean(axis=1)
            results[key] = compute_classification_metrics(yt, preds, prob_for_metrics)

    # Attach per-fold mean bal_acc for combined models
    for key in list(results.keys()):
        fold_bas = [f[key] for f in fold_results if key in f]
        if fold_bas:
            results[key]["balanced_accuracy_mean"] = float(np.mean(fold_bas))
            results[key]["balanced_accuracy_std"]  = float(np.std(fold_bas))
            results[key]["n_folds"] = len(fold_bas)

    return results if results else None


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    set_seed(SEED)
    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode(FILTER_MODE))
    morph_df = _load_morph_df()
    print(ds.summary())
    print(f"Device:  {DEVICE}")
    print(f"Folds:   {args.n_folds}")

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
                          n_folds=args.n_folds, verbose=True)
        if day_res:
            all_results[day] = day_res

    # ── Save JSON ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "combined_results_kfold_series_idor.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    COMBO_DISPLAY = ["met", "morph", "img",
                     "met+morph_mean_prob", "met+morph_majority_vote",
                     "met+img_mean_prob",   "met+img_majority_vote",
                     "morph+img_mean_prob", "morph+img_majority_vote",
                     "met+morph+img_mean_prob", "met+morph+img_majority_vote"]
    header = f"{'Day':<10}" + "".join(f"{k:>22}" for k in COMBO_DISPLAY)
    print(f"\n{'='*len(header)}")
    print(header)
    print('='*len(header))
    for day in DAY_ORDER:
        dr = all_results.get(day)
        if not dr: continue
        row = f"{day:<10}"
        for k in COMBO_DISPLAY:
            m = dr.get(k)
            ba = m.get("balanced_accuracy_mean", m["balanced_accuracy"]) if m else float("nan")
            row += f"{ba:>22.3f}"
        print(row)

    # ── Figure: three-modality mean-prob vs single modalities ───────────────
    if all_results:
        import shutil
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        series = {}
        key_style = {
            "met":                    ("Metabolite LGBM",      "#2ca02c", "o", "-"),
            "morph":                  ("Morphology LGBM",      "#9467bd", "s", "-"),
            "img":                    ("Image EfficientNet",   "#1f77b4", "^", "-"),
            "met+morph+img_mean_prob":("Combined (mean prob)", "#d62728", "D", "-"),
            "met+morph+img_majority_vote":("Combined (vote)",  "#ff7f0e", "P", "--"),
        }
        for k, (label, color, marker, ls) in key_style.items():
            day_metrics = {d: all_results[d][k]
                           for d in DAY_ORDER if d in all_results and k in all_results[d]}
            if day_metrics:
                series[label] = day_metrics

        fig_name = "combined_kfold_balanced_accuracy_series_idor.png"
        plot_balanced_accuracy_by_day(
            series,
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / fig_name,
            title="Combined Model: Balanced Accuracy by Day (series_idor, 5-fold CV)",
            style_overrides={
                label: {"color": color, "marker": marker, "linestyle": ls}
                for _, (label, color, marker, ls) in key_style.items()
            },
        )
        repo_fig = Path(f"figures/{fig_name}")
        repo_fig.parent.mkdir(exist_ok=True)
        shutil.copy(FIGURE_DIR / fig_name, repo_fig)
        print(f"Copied figure to {repo_fig}")


if __name__ == "__main__":
    main()

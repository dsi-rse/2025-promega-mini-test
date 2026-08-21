#!/usr/bin/env python3
"""Per-day EfficientNet-B0 image classifier with 5-fold stratified CV.

Same architecture and augmentation as perday_image_study.py (fixed-split version).
Cross-validation is done at the organoid level: organoids are split into 5 folds,
ensuring no organoid appears in both train and test within any fold.

For each outer fold:
  - 80 % of organoids → train;  10 % (stratified random) → val (early stopping)
  - 20 % → test
  For each day: a fresh EfficientNet is trained on the train organoids' images.
  OOF predictions are collected; final metrics are computed across all folds.

LABEL CONVENTION: 1 = Not Acceptable, 0 = Acceptable (AGENTS.md rule #9).

Outputs:
  analysis_output/images/perday_results_kfold{suffix}.json
  figures/perday_image_kfold_balanced_accuracy{suffix}.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.perday_image_kfold"
    make run ARGS="-m analysis.paper_2026_04.perday_image_kfold --days Dy30"
    DAYS="Dy30" sbatch submit_perday_image_kfold.slurm
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 1
N_FOLDS = 5
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_DIR = Path("analysis_output") / "images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
MAX_EPOCHS = 100
LR_HEAD = 5e-4
LR_BACKBONE = 5e-5
UNFREEZE_AFTER = 4
PATIENCE = 15
GRAD_CLIP = 1.0
IMG_HEIGHT = 384
IMG_WIDTH = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_FILL = [int(v * 255) for v in IMAGENET_MEAN]   # [123, 116, 103]
_BOUNDARY_DAYS = {"Dy28", "Dy30"}



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class OrganoidImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self):
        for block in list(self.backbone.blocks)[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "conv_head"):
            for p in self.backbone.conv_head.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for p in self.backbone.bn2.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def _build_transforms(train: bool, translate: tuple = (0.1, 0.1), degrees: int = 180,
                      augment: bool = True):
    base = [T.Resize((IMG_HEIGHT, IMG_WIDTH))]
    if train and augment:
        base.append(T.RandomHorizontalFlip(p=0.5))
        if degrees > 0 or translate is not None:
            base.append(T.RandomAffine(degrees=degrees, translate=translate, fill=_FILL))
        # ForegroundColorJitter: jitters organoid pixels only; background and
        # affine-fill corners are restored to the ImageNet mean fill value.
        base.append(ForegroundColorJitter(brightness=0.3, contrast=0.3,
                                          saturation=0.2, hue=0.05))
    base.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(base)


def _get_image_paths(ds: OrganoidDataset, org_ids: List[str], day: str,
                     input_mode: str) -> Tuple[List[str], List[int]]:
    """Get image paths and labels for a set of organoid IDs on a given day."""
    path_key = IMAGE_MODE_TO_PATH_KEY.get(input_mode, input_mode)
    paths, labels = [], []
    for org_id in org_ids:
        rec = ds.get_record(org_id, day)
        if rec is None:
            continue
        imgs = rec.get("images", {})
        if isinstance(path_key, tuple):
            parent, key = path_key
            path = (imgs.get(parent) or {}).get(key)
        else:
            path = imgs.get(path_key)
        if not path:
            continue
        label_str = ds.organoid_label(org_id)
        if label_str not in LABEL_TO_INT:
            continue
        paths.append(path)
        labels.append(LABEL_TO_INT[label_str])
    return paths, labels


def _train_one_fold(train_paths, train_labels, val_paths, val_labels,
                    day: str, fold_seed: int, verbose: bool,
                    augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Train one EfficientNet fold; return (probs, preds) arrays — empty if skipped."""
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos == 0 or n_neg == 0 or len(train_paths) == 0:
        return np.array([]), np.array([])

    set_seed(fold_seed)
    translate = None if day in _BOUNDARY_DAYS else (0.1, 0.1)
    degrees   = 0   if day in _BOUNDARY_DAYS else 180

    train_loader = DataLoader(
        OrganoidImageDataset(train_paths, train_labels, _build_transforms(True, translate=translate, degrees=degrees, augment=augment)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        OrganoidImageDataset(val_paths, val_labels, _build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = EfficientNetClassifier().to(DEVICE)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam([p for p in model.head.parameters() if p.requires_grad], lr=LR_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    backbone_unfrozen = False

    for epoch in range(MAX_EPOCHS):
        if epoch == UNFREEZE_AFTER and not backbone_unfrozen:
            model.unfreeze_last_blocks()
            backbone_unfrozen = True
            optimizer = optim.Adam(model.parameters(), lr=LR_BACKBONE)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                              factor=0.5, patience=5)
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        val_preds, val_true, val_loss_total, val_n = [], [], 0.0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls_dev = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                val_loss_total += criterion(logits, lbls_dev).item() * len(lbls_dev)
                val_n += len(lbls_dev)
                val_preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().numpy())
                val_true.extend(lbls.numpy().astype(int))

        val_loss = val_loss_total / max(val_n, 1)
        val_acc = accuracy_score(val_true, val_preds) if val_true else 0.0
        scheduler.step(val_loss)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            if verbose:
                print(f"    Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


def train_one_day_kfold(
    ds: OrganoidDataset,
    day: str,
    all_org_ids: List[str],
    all_labels: np.ndarray,
    *,
    input_mode: str = "cm_source_image",
    n_folds: int = N_FOLDS,
    augment: bool = True,
    verbose: bool = True,
) -> Optional[dict]:
    """Run n_folds CV for one day; return metrics dict or None."""
    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_probs = np.full(len(all_org_ids), np.nan)
    fold_bal_accs = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(all_org_ids, all_labels)):
        fold_seed = SEED + fold_i * 97

        tr_org_ids = [all_org_ids[i] for i in tr_idx]
        tr_labels  = all_labels[tr_idx]
        te_org_ids = [all_org_ids[i] for i in te_idx]

        # Internal val split (15% of train) for early stopping
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_seed)
        inner_tr_idx, inner_val_idx = next(sss.split(tr_org_ids, tr_labels))
        inner_tr_org  = [tr_org_ids[i] for i in inner_tr_idx]
        inner_val_org = [tr_org_ids[i] for i in inner_val_idx]

        train_paths, train_labels = _get_image_paths(ds, inner_tr_org,  day, input_mode)
        val_paths,   val_labels   = _get_image_paths(ds, inner_val_org, day, input_mode)
        test_paths,  test_labels  = _get_image_paths(ds, te_org_ids,    day, input_mode)

        if len(test_paths) == 0:
            if verbose:
                print(f"  Fold {fold_i+1}: no test images for {day}, skipping")
            continue

        n_pos_tr = sum(train_labels)
        n_neg_tr = len(train_labels) - n_pos_tr
        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}  train={len(train_paths)} "
                  f"({n_pos_tr} NAcc, {n_neg_tr} Acc)  val={len(val_paths)}  test={len(test_paths)}")

        if n_pos_tr == 0 or n_neg_tr == 0:
            if verbose:
                print(f"    Single class in train, skipping fold")
            continue

        model, best_val_acc = _train_one_fold(
            train_paths, train_labels, val_paths, val_labels,
            day, fold_seed, verbose, augment=augment,
        )

        # Evaluate on test
        test_loader = DataLoader(
            OrganoidImageDataset(test_paths, test_labels, _build_transforms(False)),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        model.eval()
        fold_probs, fold_preds, fold_true = [], [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(DEVICE)
                probs = torch.sigmoid(model(imgs))
                fold_probs.extend(probs.cpu().numpy())
                fold_preds.extend((probs >= 0.5).long().cpu().numpy())
                fold_true.extend(lbls.numpy().astype(int))

        fold_ba = balanced_accuracy_score(fold_true, fold_preds)
        fold_bal_accs.append(fold_ba)

        # Assign OOF probs — map back through te_org_ids to all_org_ids positions
        # (some organoids may have no image at this day; skip them)
        te_org_path_ids = [te_org_ids[j] for j in range(len(te_org_ids))
                           if ds.get_record(te_org_ids[j], day) is not None
                           and ds.organoid_label(te_org_ids[j]) in LABEL_TO_INT]
        for j, (oid, prob) in enumerate(zip(te_org_path_ids, fold_probs)):
            global_idx = all_org_ids.index(oid)
            oof_probs[global_idx] = prob

        if verbose:
            print(f"    BalAcc={fold_ba:.3f}  best_val_acc={best_val_acc:.3f}")

    # Aggregate OOF where we have predictions
    valid = ~np.isnan(oof_probs)
    if valid.sum() == 0 or len(fold_bal_accs) == 0:
        return None

    oof_true  = all_labels[valid]
    oof_probs_ = oof_probs[valid]
    oof_preds = (oof_probs_ >= 0.5).astype(int)

    if len(np.unique(oof_true)) < 2:
        return None

    metrics = compute_classification_metrics(oof_true, oof_preds, oof_probs_)
    metrics["balanced_accuracy_mean"] = float(np.mean(fold_bal_accs))
    metrics["balanced_accuracy_std"]  = float(np.std(fold_bal_accs))
    metrics["n_folds"]                = len(fold_bal_accs)
    metrics["fold_balanced_accuracies"] = [float(v) for v in fold_bal_accs]
    metrics["n_oof"]                  = int(valid.sum())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--input-mode", default="cm_image",
                        choices=["cm_image", "cm_source_image", "cm_source_mask", "overlay", "img", "mask"])
    parser.add_argument("--filter-mode", default="series_idor",
                        choices=["base", "series_idor"])
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable training augmentation (resize + normalize only)")
    args = parser.parse_args()
    augment = not args.no_augmentation

    set_seed(SEED)
    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode(args.filter_mode))
    print(ds.summary())
    print(f"Device:      {DEVICE}")
    print(f"Filter mode: {args.filter_mode}")
    print(f"CV folds:    {args.n_folds}")
    print(f"Augment:     {augment}")

    # Build per-organoid array (label constant across days)
    all_org_ids = [oid for oid in ds.organoid_ids
                   if ds.organoid_label(oid) in LABEL_TO_INT]
    all_labels  = np.array([LABEL_TO_INT[ds.organoid_label(oid)] for oid in all_org_ids])
    n_pos = int(all_labels.sum())
    n_neg = int((all_labels == 0).sum())
    print(f"Organoids:   {len(all_org_ids)} ({n_pos} NAcc, {n_neg} Acc)\n")

    days_to_run = args.days if args.days else DAY_ORDER
    results: Dict = {}

    for day in days_to_run:
        if day not in ds.days:
            print(f"Skipping {day} (no data)")
            continue
        print(f"\n{'='*50}\nImage Kfold CV — {day}\n{'='*50}")
        m = train_one_day_kfold(
            ds, day, all_org_ids, all_labels,
            input_mode=args.input_mode,
            n_folds=args.n_folds,
            augment=augment,
            verbose=True,
        )
        if m:
            results[day] = m
            print(f"  OOF BalAcc={m['balanced_accuracy']:.4f}  "
                  f"FoldMean={m['balanced_accuracy_mean']:.4f}±{m['balanced_accuracy_std']:.4f}")

    suffix = f"_{args.filter_mode}" if args.filter_mode != "base" else ""
    if not augment:
        suffix += "_noaug"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"perday_results_kfold{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    if results:
        print(f"\n{'='*60}\nPER-DAY IMAGE KFOLD SUMMARY\n{'='*60}")
        bal_accs, specs = [], []
        for day in DAY_ORDER:
            m = results.get(day)
            if not m:
                continue
            ba = m.get("balanced_accuracy_mean", m["balanced_accuracy"])
            std = m.get("balanced_accuracy_std", 0.0)
            bal_accs.append(ba)
            specs.append(m["specificity"])
            print(f"  {day}: bal_acc={ba:.4f}±{std:.4f}  "
                  f"OOF_bal_acc={m['balanced_accuracy']:.4f}  "
                  f"spec={m['specificity']:.4f}")
        print(f"\n  Avg FoldMean BalAcc: {np.mean(bal_accs):.1%}")

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig_name = f"perday_image_kfold_balanced_accuracy{suffix}.png"
        plot_balanced_accuracy_by_day(
            {f"EfficientNet-B0 ({args.n_folds}-fold CV)": results},
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / fig_name,
            title=f"Per-Day Image Classifier: {args.n_folds}-Fold CV Balanced Accuracy ({args.filter_mode})",
            style_overrides={
                f"EfficientNet-B0 ({args.n_folds}-fold CV)": {"color": "#1f77b4", "marker": "o"},
            },
        )
        repo_fig = Path(f"figures/{fig_name}")
        repo_fig.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copy(FIGURE_DIR / fig_name, repo_fig)
        print(f"Copied figure to {repo_fig}")


if __name__ == "__main__":
    main()

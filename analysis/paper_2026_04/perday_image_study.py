#!/usr/bin/env python3
"""Reproduce per-day EfficientNet image classifier results from the paper.

Architecture: EfficientNet-B0 (ImageNet pretrained) → 128-dim → binary logit
Input: aspect-ratio-conserved images (resized_575_square via cm_source_image_abs), 575×575
Training: warmup with frozen backbone → unfreeze last 2 blocks
Loss: BCEWithLogitsLoss with class weights

LABEL CONVENTION: 1 = Not Acceptable, 0 = Acceptable (matches LABEL_TO_INT,
per AGENTS.md rule #9). pos_weight = n_neg/n_pos correctly upweights the
Not Acceptable minority class.

Outputs:
  - $ANALYSIS_OUTPUT_DIR/images/perday_results.json
  - $ANALYSIS_OUTPUT_DIR/figures/perday_image_balanced_accuracy.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.perday_image_study"
    make run ARGS="-m analysis.paper_2026_04.perday_image_study --days Dy30"
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    LABEL_TO_INT,
    OrganoidDataset,
)
from pipeline.splits import Splits

from .common import compute_classification_metrics, plot_balanced_accuracy_by_day

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 1
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "images"

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
IMAGENET_STD = [0.229, 0.224, 0.225]


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


# Fill colour for areas revealed by rotation/translation: ImageNet mean (≈ mean-fill background)
_FILL = [int(v * 255) for v in IMAGENET_MEAN]  # [123, 116, 103]


def _build_transforms(train: bool, translate: tuple = (0.1, 0.1)):
    """Build transform pipeline.

    Geometric design (no redundancy):
      - RandomHorizontalFlip(p=0.5) + full rotation covers all 8 dihedral
        symmetries.  VerticalFlip is omitted: V-flip = H-flip + 180° rotation.
      - RandomAffine combines rotation [-180°, 180°] and translation [±10%]
        in one interpolation pass to avoid double resampling.
      - translate=None disables translation (use for days where organoids
        touch the image boundary, e.g. Dy28/Dy30 in the base cohort).

    Photometric: brightness/contrast/saturation jitter; hue kept small
    because the mean-fill background is fixed and does not shift with hue.
    """
    base = [T.Resize((IMG_HEIGHT, IMG_WIDTH))]
    if train:
        base.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=180,
                translate=translate,
                fill=_FILL,
            ),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ])
    base.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(base)


def _split_data(ds: OrganoidDataset, split: str, day: str, input_mode: str) -> Tuple[list, list]:
    items = ds.get_image_paths(split, day, mode=input_mode)
    paths = [p for _, _, p in items]
    labels = [LABEL_TO_INT[lbl] for _, lbl, _ in items]
    return paths, labels


# Days where large organoids may touch the image boundary in the base cohort.
# Translation is disabled for these days to avoid shifting boundary-clipped cells.
_BOUNDARY_DAYS = {"Dy28", "Dy30"}


def train_one_day(ds: OrganoidDataset, day: str, *, input_mode: str = "overlay",
                  verbose: bool = True) -> Optional[dict]:
    set_seed(SEED)
    train_paths, train_labels = _split_data(ds, "train", day, input_mode)
    val_paths,   val_labels   = _split_data(ds, "val",   day, input_mode)
    test_paths,  test_labels  = _split_data(ds, "test",  day, input_mode)

    if len(train_paths) == 0 or len(test_paths) == 0:
        if verbose:
            print(f"  Skipping {day}: no data")
        return None

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        if verbose:
            print(f"  Skipping {day}: single class in train")
        return None

    if verbose:
        print(f"  Train: {len(train_paths)} ({n_pos} NAcc, {n_neg} Acc)")
        print(f"  Val:   {len(val_paths)}")
        print(f"  Test:  {len(test_paths)}")

    # Disable translation for days where cells commonly reach the image boundary
    translate = None if day in _BOUNDARY_DAYS else (0.1, 0.1)
    train_loader = DataLoader(
        OrganoidImageDataset(train_paths, train_labels, _build_transforms(True, translate=translate)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        OrganoidImageDataset(val_paths, val_labels, _build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        OrganoidImageDataset(test_paths, test_labels, _build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = EfficientNetClassifier().to(DEVICE)

    # pos_weight = #neg / #pos: label 1 = Not Acceptable (minority), so this
    # ratio (>1) upweights the minority class, per AGENTS.md rule #9.
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

        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_true, val_loss_total, val_n = [], [], 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels_dev = labels.to(DEVICE)
                logits = model(imgs)
                vloss = criterion(logits, labels_dev)
                val_loss_total += vloss.item() * len(labels_dev)
                val_n += len(labels_dev)
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.numpy().astype(int))
        val_loss = val_loss_total / max(val_n, 1)
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_loss)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: val_acc={val_acc:.4f}, "
                  f"val_bal_acc={balanced_accuracy_score(val_true, val_preds):.4f}")

    # Test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_probs, test_true = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend((probs >= 0.5).long().cpu().numpy())
            test_true.extend(labels.numpy().astype(int))

    metrics = compute_classification_metrics(
        np.array(test_true), np.array(test_preds), np.array(test_probs),
    )
    metrics["best_val_accuracy"] = round(float(best_val_acc), 4)
    metrics["threshold"] = 0.5

    if verbose:
        print(f"  Test: bal_acc={metrics['balanced_accuracy']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, "
              f"specificity={metrics['specificity']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--input-mode", default="cm_source_image",
                        choices=["cm_source_image", "cm_source_mask", "overlay", "img", "mask"])
    parser.add_argument("--filter-mode", default="base",
                        choices=["base", "series_idor"],
                        help="'base' (canonical 205 org) or 'series_idor' (132 org, ef<=0.05 "
                             "every day — same cohort as Amanda's idor_minvotes3/series split)")
    args = parser.parse_args()

    from pipeline.data_loader import filters_for_mode
    set_seed(SEED)
    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode(args.filter_mode))
    print(ds.summary())
    print(f"Device: {DEVICE}")
    print(f"Filter mode: {args.filter_mode}")

    days_to_train = args.days if args.days else DAY_ORDER
    results: dict = {}
    for day in days_to_train:
        if day not in ds.days:
            print(f"\nSkipping {day} (no data)")
            continue
        print(f"\n{'=' * 50}\nImage Classifier - {day}\n{'=' * 50}")
        m = train_one_day(ds, day, input_mode=args.input_mode)
        if m:
            results[day] = m

    suffix = f"_{args.filter_mode}" if args.filter_mode != "base" else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"perday_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    if results:
        print(f"\n{'=' * 60}\nPER-DAY IMAGE RESULTS SUMMARY\n{'=' * 60}")
        bal_accs, specs, f1_nas = [], [], []
        for day in DAY_ORDER:
            m = results.get(day)
            if not m:
                continue
            bal_accs.append(m["balanced_accuracy"])
            specs.append(m["specificity"])
            f1_nas.append(m["f1_not_acceptable"])
            print(f"  {day}: bal_acc={m['balanced_accuracy']:.4f}, "
                  f"specificity={m['specificity']:.4f}, "
                  f"f1_NA={m['f1_not_acceptable']:.4f}")
        n = len(bal_accs)
        zero_spec_days = sum(1 for s in specs if s == 0.0)
        print(f"\n  Avg Specificity: {np.mean(specs):.1%}")
        print(f"  Avg Bal Acc:     {np.mean(bal_accs):.1%}")
        print(f"  Days Spec=0:     {zero_spec_days}/{n}")
        print(f"  Avg F1(NA):      {np.mean(f1_nas):.1%}")

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_balanced_accuracy_by_day(
            {"Per-day EfficientNet": results},
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / "perday_image_balanced_accuracy.png",
            title="Per-Day Image Classifier: Balanced Accuracy by Day",
            style_overrides={"Per-day EfficientNet": {"color": "#1f77b4", "marker": "o"}},
        )


if __name__ == "__main__":
    main()

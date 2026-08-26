#!/usr/bin/env python3
"""GradCAM visualisation for EfficientNet-B0 image classifier.

Trains one CV fold for a given day, then shows GradCAM heatmaps alongside
original and augmented views for a selection of test examples.

Output: figures/gradcam_{day}_fold{fold}.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.gradcam_demo --day Dy30 --fold 2"
    sbatch analysis/paper_2026_04/submit_gradcam_demo.slurm
"""

import argparse
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pipeline.data_loader import (
    DAY_ORDER, LABEL_TO_INT, OrganoidDataset, filters_for_mode,
    IMAGE_MODE_TO_PATH_KEY,
)
from pipeline.splits import Splits

from .common import ForegroundColorJitter
from .perday_image_kfold import (
    SEED, N_FOLDS, DEVICE, BATCH_SIZE,
    IMG_HEIGHT, IMG_WIDTH, IMAGENET_MEAN, IMAGENET_STD,
    _FILL, _BOUNDARY_DAYS,
    EfficientNetClassifier, OrganoidImageDataset,
    _build_transforms, _get_image_paths, _train_one_fold, set_seed,
)

warnings.filterwarnings("ignore")

ALL_DATA_PATH = "data/all_data.json"
N_AUG_COLS    = 4   # augmented views per row
N_EXAMPLES    = 3   # examples per class (TP, TN, FN/FP if available)
OUT_DIR       = Path("figures")


# ── GradCAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping for a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts  = None
        self._grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, output):
        self._acts = output.detach()

    def _bwd(self, _m, _gi, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Return normalised CAM (H×W float32 in [0,1]) for x (1×C×H×W)."""
        self.model.zero_grad()
        logit = self.model(x)
        logit.backward()                          # gradient of raw score

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._acts).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _overlay(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay jet heatmap on image; both H×W×3 uint8 / H×W float."""
    from PIL import Image as PILImage
    import matplotlib.cm as cm
    h, w = img_np.shape[:2]
    cam_resized = np.array(
        PILImage.fromarray((cam * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
    ) / 255.0
    heat = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * img_np + alpha * heat).astype(np.uint8)


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize ImageNet-normalised tensor → uint8 H×W×3."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── augmentation (PIL → PIL, no ToTensor) ───────────────────────────────────

def _aug_pil(day: str) -> T.Compose:
    degrees = 0 if day in _BOUNDARY_DAYS else 180
    ops = [
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if degrees > 0:
        ops.append(T.RandomAffine(degrees=degrees, translate=(0.1, 0.1), fill=_FILL))
    else:
        ops.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=_FILL))
    ops.append(ForegroundColorJitter(brightness=0.3, contrast=0.3,
                                     saturation=0.2, hue=0.05))
    return T.Compose(ops)


def _resize_pil() -> T.Compose:
    return T.Compose([T.Resize((IMG_HEIGHT, IMG_WIDTH))])


# ── main logic ───────────────────────────────────────────────────────────────

def train_fold(ds, day, all_org_ids, all_labels, fold_idx):
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(outer_cv.split(all_org_ids, all_labels))
    tr_idx, te_idx = splits[fold_idx]

    tr_org_ids = [all_org_ids[i] for i in tr_idx]
    tr_labels  = all_labels[tr_idx]
    te_org_ids = [all_org_ids[i] for i in te_idx]

    fold_seed = SEED + fold_idx * 97
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_seed)
    inner_tr_idx, inner_val_idx = next(sss.split(tr_org_ids, tr_labels))
    inner_tr_org  = [tr_org_ids[i] for i in inner_tr_idx]
    inner_val_org = [tr_org_ids[i] for i in inner_val_idx]

    train_paths, train_lbl = _get_image_paths(ds, inner_tr_org,  day, "cm_image")
    val_paths,   val_lbl   = _get_image_paths(ds, inner_val_org, day, "cm_image")
    test_paths,  test_lbl  = _get_image_paths(ds, te_org_ids,    day, "cm_image")
    test_org_ids = [oid for oid in te_org_ids
                    if ds.get_record(oid, day) is not None
                    and ds.organoid_label(oid) in LABEL_TO_INT]

    n_pos = sum(train_lbl)
    print(f"  Fold {fold_idx+1}: train={len(train_paths)} ({n_pos} NAcc)  "
          f"val={len(val_paths)}  test={len(test_paths)}")

    model, _ = _train_one_fold(train_paths, train_lbl, val_paths, val_lbl,
                                day, fold_seed, verbose=True, augment=True)

    # Evaluate
    test_loader = DataLoader(
        OrganoidImageDataset(test_paths, test_lbl, _build_transforms(False)),
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

    ba = balanced_accuracy_score(trues, preds)
    print(f"  Fold {fold_idx+1} BA = {ba:.3f}")
    return model, list(zip(test_org_ids, test_paths, trues, preds, probs))


def pick_examples(results, n_per_group=2):
    """Pick TP, TN, and error examples."""
    tps = [(oid, p, t, pr, pb) for oid, p, t, pr, pb in results if t == 1 and pr == 1]
    tns = [(oid, p, t, pr, pb) for oid, p, t, pr, pb in results if t == 0 and pr == 0]
    fns = [(oid, p, t, pr, pb) for oid, p, t, pr, pb in results if t == 1 and pr == 0]
    fps = [(oid, p, t, pr, pb) for oid, p, t, pr, pb in results if t == 0 and pr == 1]
    chosen = []
    for group, label in [(tps, "TP (NAcc→NAcc)"), (tns, "TN (Acc→Acc)"),
                          (fns, "FN (NAcc→Acc)"),  (fps, "FP (Acc→NAcc)")]:
        for item in group[:n_per_group]:
            chosen.append((label, *item))
    return chosen


def make_figure(chosen, model, day, fold_idx, out_path):
    gradcam = GradCAM(model, model.backbone.conv_head)
    aug_tf   = _aug_pil(day)
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    resize_tf = _resize_pil()

    n_rows = len(chosen)
    # columns: original | aug×N_AUG_COLS | gradcam-orig | gradcam-overlay
    n_cols = 1 + N_AUG_COLS + 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 1.8),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.18})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = (["Original"] +
                  [f"Aug {i+1}" for i in range(N_AUG_COLS)] +
                  ["GradCAM", "Overlay"])
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=8, fontweight="bold" if j == 0 else "normal", pad=3)

    for row_i, (group_label, oid, img_path, true_lbl, pred_lbl, prob) in enumerate(chosen):
        pil_img  = Image.open(img_path).convert("RGB")
        orig_np  = np.array(resize_tf(pil_img))

        # ── col 0: original ──────────────────────────────────────────────
        axes[row_i, 0].imshow(orig_np)
        axes[row_i, 0].axis("off")
        row_label = (f"{oid}\n"
                     f"True: {'NAcc' if true_lbl else 'Acc'}  "
                     f"Pred: {'NAcc' if pred_lbl else 'Acc'}  "
                     f"p={prob:.2f}")
        axes[row_i, 0].annotate(
            row_label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-6, 0), textcoords="offset points",
            fontsize=5.5, va="center", ha="right",
        )
        # colour border by outcome
        border_col = {"TP (NAcc→NAcc)": "#27ae60", "TN (Acc→Acc)": "#2980b9",
                      "FN (NAcc→Acc)": "#e74c3c",  "FP (Acc→NAcc)": "#e67e22"
                      }.get(group_label, "gray")
        for spine in axes[row_i, 0].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2)

        # ── cols 1..N_AUG_COLS: augmented ───────────────────────────────
        for col_i in range(N_AUG_COLS):
            seed_i = 42 + row_i * 13 + col_i
            random.seed(seed_i); np.random.seed(seed_i); torch.manual_seed(seed_i)
            aug_np = np.array(aug_tf(pil_img))
            axes[row_i, 1 + col_i].imshow(aug_np)
            axes[row_i, 1 + col_i].axis("off")

        # ── col N+1: GradCAM heatmap ─────────────────────────────────────
        model.eval()
        x = to_tensor(resize_tf(pil_img)).unsqueeze(0).to(DEVICE)
        x.requires_grad_(False)
        # Need grad for GradCAM
        x_grad = x.clone().requires_grad_(True)
        cam = gradcam(x_grad)
        # Resize cam to img size
        cam_img = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (orig_np.shape[1], orig_np.shape[0]), Image.BILINEAR)
        ) / 255.0
        axes[row_i, 1 + N_AUG_COLS].imshow(cam_img, cmap="jet", vmin=0, vmax=1)
        axes[row_i, 1 + N_AUG_COLS].axis("off")

        # ── col N+2: overlay ─────────────────────────────────────────────
        overlay = _overlay(orig_np, cam_img, alpha=0.4)
        axes[row_i, 1 + N_AUG_COLS + 1].imshow(overlay)
        axes[row_i, 1 + N_AUG_COLS + 1].axis("off")

    # Vertical separator after "Original"
    for row_i in range(n_rows):
        axes[row_i, 0].spines["right"].set_visible(True)
        axes[row_i, 0].spines["right"].set_linewidth(1)
        axes[row_i, 0].spines["right"].set_edgecolor("#aaa")

    # Legend patches
    patches = [
        mpatches.Patch(color="#27ae60", label="TP — NAcc correctly caught"),
        mpatches.Patch(color="#2980b9", label="TN — Acc correctly passed"),
        mpatches.Patch(color="#e74c3c", label="FN — NAcc missed"),
        mpatches.Patch(color="#e67e22", label="FP — Acc falsely flagged"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"GradCAM · EfficientNet-B0 · {day}  (fold {fold_idx+1})  "
        f"· target layer: conv_head",
        fontsize=11, fontweight="bold", y=1.01,
    )

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day",  default="Dy30")
    parser.add_argument("--fold", type=int, default=2,
                        help="1-indexed fold number")
    parser.add_argument("--n-per-group", type=int, default=2,
                        help="Examples per outcome group (TP/TN/FN/FP)")
    args = parser.parse_args()

    fold_idx = args.fold - 1  # convert to 0-indexed
    set_seed(SEED)

    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical(),
                         filters=filters_for_mode("series_idor"))
    all_org_ids = [oid for oid in ds.organoid_ids
                   if ds.organoid_label(oid) in LABEL_TO_INT]
    all_labels  = np.array([LABEL_TO_INT[ds.organoid_label(oid)] for oid in all_org_ids])

    print(f"Day: {args.day}  Fold: {args.fold}  Device: {DEVICE}")
    model, test_results = train_fold(ds, args.day, all_org_ids, all_labels, fold_idx)

    chosen = pick_examples(test_results, n_per_group=args.n_per_group)
    print(f"\nSelected {len(chosen)} examples:")
    for group_label, oid, _, true_lbl, pred_lbl, prob in chosen:
        print(f"  {group_label:22s}  {oid}  p={prob:.3f}")

    out_path = OUT_DIR / f"gradcam_{args.day}_fold{args.fold}.png"
    make_figure(chosen, model, args.day, fold_idx, out_path)


if __name__ == "__main__":
    main()

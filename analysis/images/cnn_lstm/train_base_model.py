#!/usr/bin/env python3
"""
Baseline EfficientNet (single timepoint) for comparison with LSTM models.
Trains on each day range separately: [8, 10, 13, 15, 17, 20.5, 24, 30]
Uses the same data splits as CNN-LSTM temporal models for fair comparison.
Run: python train_baseline_effnet.py
"""

import sys, json, random, argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----- Repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json, resolve_split_path

# -------- Config --------
DAY_RANGES = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]  # Full per-day set (11 days).
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 100
PATIENCE = 15
LR = 5e-4
GRAD_CLIP = 1.0
SEED = 1
TARGET_SIZE = (384, 512)  # (H, W) to match coworker's code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Dataset ----------
class SingleDayOrganoidDataset(Dataset):
    """
    Dataset for single timepoint organoid images.
    Uses the LSTM processed images (same as LSTM but picks one timepoint).
    """
    def __init__(self, organoid_ids, series_metadata, target_day, transform=None,
                 image_type='std', bbox_crop=False, bbox_pad=10):
        """
        bbox_crop: if True, crop each image to its mask bounding box (with bbox_pad
                   pixels of padding) before applying transforms. This removes the
                   bulk-size signal from the input — every organoid fills the same
                   display area regardless of its biological size. Used to test
                   whether the model is relying on size as a shortcut.
        """
        self.samples = []

        for org_id in organoid_ids:
            metadata = series_metadata.get(org_id, {})
            label_str = str(metadata.get("label", "")).strip().lower()
            label = 1 if label_str in ("good", "acceptable", "accepted") else 0

            timepoints = metadata.get('timepoints', [])
            if not timepoints:
                continue

            # Find the timepoint closest to target_day
            best_tp = min(timepoints, key=lambda tp: abs(tp['mdl_day'] - target_day))

            img_path = best_tp.get('img_paths', {}).get(image_type)
            if img_path is None or not Path(img_path).exists():
                continue
            mask_path = best_tp.get('mask_paths', {}).get(image_type)
            # mask is only required if bbox_crop is on
            if bbox_crop and (mask_path is None or not Path(mask_path).exists()):
                continue

            self.samples.append({
                "img_path":  img_path,
                "mask_path": mask_path,
                "label":     label,
                "org_id":    org_id,
                "actual_day": best_tp['mdl_day'],
            })

        self.transform = transform
        self.bbox_crop = bbox_crop
        self.bbox_pad = bbox_pad
        print(f"  Loaded {len(self.samples)} samples for day ~{target_day}"
              + ("  [bbox-crop enabled]" if bbox_crop else ""))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image (same as LSTM)
        from skimage.io import imread
        img = imread(sample["img_path"])

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Optional bbox-crop: crop the image to its mask's bounding box, then
        # LETTERBOX it (pad with the image's mean color) to a target aspect
        # ratio (384/512 = 0.75 = H/W) so the subsequent Resize to (384, 512)
        # does NOT stretch the organoid. This removes the bulk-size signal
        # while preserving morphology.
        if self.bbox_crop:
            mask_arr = imread(sample["mask_path"])
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            ys, xs = np.where(mask_arr > 127)
            if ys.size > 0 and xs.size > 0:
                y0 = max(0, int(ys.min()) - self.bbox_pad)
                y1 = min(img.shape[0], int(ys.max()) + self.bbox_pad)
                x0 = max(0, int(xs.min()) - self.bbox_pad)
                x1 = min(img.shape[1], int(xs.max()) + self.bbox_pad)
                if y1 > y0 and x1 > x0:
                    crop = img[y0:y1, x0:x1]
                    # Letterbox to match the model's target H/W ratio (0.75)
                    target_h_over_w = 384.0 / 512.0
                    ch, cw = crop.shape[:2]
                    cur_ratio = ch / cw
                    # Use the original image's background color (a corner pixel,
                    # which under mean-fill clipping is always the global gray)
                    # so letterbox padding matches the existing background.
                    # Falls back to the crop mean only if the original is too small.
                    if img.shape[0] >= 4 and img.shape[1] >= 4:
                        fill = img[:2, :2].reshape(-1, img.shape[-1]).mean(axis=0)
                    else:
                        fill = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
                    if cur_ratio > target_h_over_w:
                        # too tall — pad sides
                        new_w = int(round(ch / target_h_over_w))
                        pad_w = new_w - cw
                        left = pad_w // 2
                        right = pad_w - left
                        pad = np.full((ch, pad_w, crop.shape[-1]), 0.0, dtype=crop.dtype)
                        pad[:] = fill
                        crop = np.concatenate([pad[:, :left], crop, pad[:, :right]], axis=1)
                    elif cur_ratio < target_h_over_w:
                        # too wide — pad top/bottom
                        new_h = int(round(cw * target_h_over_w))
                        pad_h = new_h - ch
                        top = pad_h // 2
                        bot = pad_h - top
                        pad = np.full((pad_h, cw, crop.shape[-1]), 0.0, dtype=crop.dtype)
                        pad[:] = fill
                        crop = np.concatenate([pad[:top], crop, pad[:bot]], axis=0)
                    img = crop
            # else: empty mask, leave img as-is

        # Apply transforms (if any)
        if self.transform:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = self.transform(img_pil)
            img = np.array(img_pil).astype(np.float32) / 255.0
        
        # Convert to tensor and apply ImageNet normalization (SAME AS LSTM!)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img).float()
        
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img = (img - imagenet_mean) / imagenet_std
        
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return img, label, sample["org_id"]


# ---------- Model ----------
class BaselineEfficientNet(nn.Module):
    """Single image classifier using EfficientNet-B0."""
    
    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        eff.classifier = nn.Identity()
        self.backbone = eff
        
        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def unfreeze_backbone(self, n_blocks=2):
        """Unfreeze last n blocks of EfficientNet."""
        feats = getattr(self.backbone, "features", None)
        if feats is None:
            return
        start = max(0, len(feats) - n_blocks)
        for i in range(start, len(feats)):
            for p in feats[i].parameters():
                p.requires_grad = True
        print(f"  Unfroze last {n_blocks} blocks of backbone")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(1)
        return logits


# ---------- Evaluation ----------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    losses = []
    
    for imgs, labels, ids in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_ids.extend(ids)
    
    if len(all_probs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float('nan'), float('nan'), [], [], 0.0

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs > 0.5).int()

    acc = (preds == labels.int()).float().mean().item()
    bal_acc = float(balanced_accuracy_score(labels.numpy(), preds.numpy()))

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds.numpy(), average="binary", zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        auc = float("nan")
    
    try:
        ap = average_precision_score(labels.numpy(), probs.numpy())
    except ValueError:
        ap = float("nan")
    
    # Get false positives/negatives
    fp_ids = [all_ids[i] for i in range(len(all_ids)) if preds[i] == 1 and labels[i] == 0]
    fn_ids = [all_ids[i] for i in range(len(all_ids)) if preds[i] == 0 and labels[i] == 1]
    
    return (
        float(np.mean(losses)),
        acc,
        float(prec),
        float(rec),
        float(f1),
        float(auc),
        float(ap),
        fp_ids,
        fn_ids,
        bal_acc,
    )


# ---------- Training ----------
def train_for_day(target_day, train_ids, val_ids, test_ids,
                  train_meta, val_meta, test_meta, device, output_dir,
                  image_type='std', pos_weight_scale=1.0, bbox_crop=False):
    print(f"\n{'='*70}\nTRAINING BASELINE for DAY {target_day}\n{'='*70}")

    train_tf = T.Compose([
        T.Resize(TARGET_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])

    eval_tf = T.Compose([
        T.Resize(TARGET_SIZE),
    ])

    train_dataset = SingleDayOrganoidDataset(train_ids, train_meta, target_day, transform=train_tf, image_type=image_type, bbox_crop=bbox_crop)
    val_dataset   = SingleDayOrganoidDataset(val_ids,   val_meta,   target_day, transform=eval_tf,  image_type=image_type, bbox_crop=bbox_crop)
    test_dataset  = SingleDayOrganoidDataset(test_ids,  test_meta,  target_day, transform=eval_tf,  image_type=image_type, bbox_crop=bbox_crop)
    
    if len(train_dataset) == 0:
        print(f"  ⚠ No training samples for day {target_day}, skipping")
        return None
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Class balance
    train_labels = [s["label"] for s in train_dataset.samples]
    n_good = sum(train_labels)
    n_bad = len(train_labels) - n_good
    if n_good == 0: n_good = 1
    if n_bad == 0: n_bad = 1
    raw_pw = n_bad / n_good
    pos_weight = torch.tensor([raw_pw * pos_weight_scale], device=device)
    print(f"  Class balance: good={n_good}, bad={n_bad}, "
          f"pos_weight={pos_weight.item():.3f}  "
          f"(raw={raw_pw:.3f} × scale={pos_weight_scale:.2f})")
    
    # Model
    model = BaselineEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0
    history = []  # track per-epoch metrics for plotting

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        # Unfreeze backbone after 3 epochs
        if epoch == 4:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_ap, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        print(
            f"Epoch {epoch:02d} | Train {train_acc:.3f}/{train_loss:.4f} | "
            f"Val {val_acc:.3f}/{val_loss:.4f} (P {val_prec:.3f} R {val_rec:.3f} F1 {val_f1:.3f})"
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print("  * new best")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Test with best model
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state, strict=True)
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_ap, test_fp, test_fn, test_bal_acc = evaluate(
        model, test_loader, criterion, device
    )
    
    # Save model
    model_dir = output_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_day_{target_day}.pth"
    torch.save({
        "state_dict": best_state,
        "target_day": target_day,
        "best_val_acc": best_val_acc,
    }, model_path)
    
    print(f"\nFinal TEST results:")
    print(f"  Acc {test_acc:.3f} | F1 {test_f1:.3f} | P {test_prec:.3f} | R {test_rec:.3f}")
    print(f"  Saved → {model_path}")
    
    # Save confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.int().cpu().numpy())
    
    if len(all_preds) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix (Test Set):")
        print(f"              Predicted")
        print(f"              Good   Bad")
        print(f"Actual Good   {cm[1,1]:<6} {cm[1,0]:<6}")
        print(f"Actual Bad    {cm[0,1]:<6} {cm[0,0]:<6}")

        # --- Save confusion matrix image ---
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        classes = ['Bad/Neg', 'Good/Pos']
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted', ylabel='Actual',
               title=f'Confusion Matrix – Day {target_day} (Test)')
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        cm_path = model_dir / f'confusion_matrix_day_{target_day}.png'
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix saved → {cm_path}")

    # --- Save accuracy & loss plot ---
    if history:
        epochs = [h['epoch'] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, [h['train_acc'] for h in history], label='Train Acc')
        ax1.plot(epochs, [h['val_acc'] for h in history], label='Val Acc')
        ax1.set(xlabel='Epoch', ylabel='Accuracy', title=f'Accuracy – Day {target_day}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [h['train_loss'] for h in history], label='Train Loss')
        ax2.plot(epochs, [h['val_loss'] for h in history], label='Val Loss')
        ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Loss – Day {target_day}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = model_dir / f'training_curves_day_{target_day}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Training curves saved → {plot_path}")

    del model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    
    return {
        "target_day": target_day,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_balanced_acc": float(test_bal_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "model_path": str(model_path),
        "test_false_positives": test_fp,
        "test_false_negatives": test_fn,
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='Train single-day EfficientNet baseline')
    parser.add_argument('--output-dir', type=str, default='outputs/base_models/base_effnet',
                        help='Output directory for model checkpoints and results')
    parser.add_argument('--image-type', type=str, default='std', choices=['clipped', 'std'],
                        help='Image variant: std (512x384) or clipped (575x575 AR meanfill)')
    parser.add_argument('--splits-dir', type=str, default='data_splits',
                        help=('Directory holding train/val/test split JSONs. Accepts both '
                              'cohort layout (<dir>/{train,val,test}.json) and legacy '
                              'layout (<dir>/{train,val,test}_idor_series.json). Default: '
                              'data_splits/ (legacy).'))
    parser.add_argument('--bbox-crop', action='store_true',
                        help=('If set, crop each image to the organoid mask bounding box '
                              '(+10 px padding) before applying transforms. Removes the bulk '
                              'size signal from the input — every organoid fills the same '
                              'display area regardless of biological size. Used to test '
                              'whether the model is relying on a size shortcut.'))
    parser.add_argument('--pos-weight-scale', type=float, default=1.0,
                        help=('Multiplier applied to the auto-computed pos_weight (= n_bad/n_good). '
                              '1.0 = default behavior. Use <1 (e.g. 0.3) to penalize missing the '
                              'Bad class more aggressively; that downweights Acceptable errors '
                              'further so the optimizer cannot ignore large-Bad misclassifications '
                              'as "cheap" losses. Useful to test whether the model is willing to '
                              'learn morphology beyond its current size shortcut.'))
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Load data (same splits as LSTM!)
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    print(f"Splits dir: {args.splits_dir}")
    train_ids, train_meta = load_split_from_json(resolve_split_path(args.splits_dir, 'train'))
    val_ids,   val_meta   = load_split_from_json(resolve_split_path(args.splits_dir, 'val'))
    test_ids,  test_meta  = load_split_from_json(resolve_split_path(args.splits_dir, 'test'))

    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    print("\n" + "="*70)
    print("STARTING BASELINE TRAINING")
    print("="*70)
    
    # Train for each day range (same as LSTM)
    results = []
    for target_day in DAY_RANGES:
        result = train_for_day(
            target_day, train_ids, val_ids, test_ids,
            train_meta, val_meta, test_meta, device,
            out_dir / f"day_{target_day}",
            image_type=args.image_type,
            pos_weight_scale=args.pos_weight_scale,
            bbox_crop=args.bbox_crop,
        )
        if result:
            results.append(result)
    
    # Save all results (matching LSTM format)
    results_path = out_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("BASELINE TRAINING SUMMARY")
    print("="*70)
    print(f"{'Day':<15} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*70)
    for r in results:
        print(f"{str(r['target_day']):<15} {r['best_val_acc']:<12.3f} {r['test_acc']:<12.3f} {r['test_f1']:<12.3f}")
    
    best = max(results, key=lambda x: x["test_acc"]) if results else None
    if best:
        print(f"\nBest on test (day {best['target_day']}): Acc={best['test_acc']:.3f}, F1={best['test_f1']:.3f}")
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
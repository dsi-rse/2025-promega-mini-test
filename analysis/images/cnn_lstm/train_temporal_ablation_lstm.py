"""
Temporal ablation with EfficientNet features + Temporal Attention (BCE) + LSTM?
Run: python analysis/images/cnn_lstm/train_temporal_ablation_attn.py
"""

import sys, json, math, argparse
import os, random
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import precision_recall_fscore_support

from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    load_split_from_json,
    resolve_split_path,
)





def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Config ----------------
DAY_RANGES = [
    6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30
]
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 100
WARMUP_EPOCHS = 3
LR_HEAD = 5e-4           # higher: new layers adapt quickly
LR_CNN_UNFREEZE = 1e-4   # lower: slow fine-tuning of pretrained CNN
GRAD_CLIP = 1.0
PATIENCE = 15            # faster convergence / less wasted epochs
ATTN_DROPOUT = 0.4       # same as your best-performing LSTM run
SEED = 1


class OrganoidCNN_LSTM(nn.Module):
    def __init__(self, d_cnn=1280, hidden_size=256, num_layers=1, bidirectional=False):
        super().__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        eff.classifier = nn.Identity()
        self.cnn = eff
        for p in self.cnn.parameters():
            p.requires_grad = False

        # these should be inside __init__, not unfreeze_last_blocks
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_cnn // 2),
            nn.ReLU(),
            nn.Linear(d_cnn // 2, d_cnn),
        )

        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.5),
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )


    def unfreeze_last_blocks(self, n_blocks: int = 2):
        """Unfreeze last EfficientNet feature blocks for fine-tuning."""
        feats = getattr(self.cnn, "features", None)
        if feats is None:
            return
        start = max(0, len(feats) - n_blocks)
        for i in range(start, len(feats)):
            for p in feats[i].parameters():
                p.requires_grad = True

    def forward(self, x, days_norm):  # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.cnn(x[:, t])
            dt = days_norm[:, t].unsqueeze(1).to(f.device)
            f = f + self.time_proj(dt)
            feats.append(f)
        feats = torch.stack(feats, dim=1)

        lstm_out, _ = self.lstm(feats)
        last_hidden = lstm_out[:, -1, :]
        logit = self.head(last_hidden).squeeze(1)
        return logit


# -------------- Metrics --------------
@torch.no_grad()
def evaluate_binary(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels, losses = [], [], []
    false_pos, false_neg = [], []

    for seqs, days, labels, weights, ids in loader:
        seqs   = seqs.to(device)
        days   = days.to(device).float()
        labels = labels.float().to(device)

        logits = model(seqs, days)

        # criterion has reduction='none' → average for reporting
        loss_raw = criterion(logits, labels)   # (B,)
        losses.append(loss_raw.mean().item())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu()
        labels_cpu = labels.int().cpu()

        # FP/FN with organoid ids
        for oid, pred, true in zip(ids, preds, labels_cpu):
            if pred == 1 and true == 0:
                false_pos.append(oid)
            elif pred == 0 and true == 1:
                false_neg.append(oid)

        all_probs.append(probs.cpu())
        all_labels.append(labels_cpu)

    if len(all_probs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false_pos, false_neg

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds  = (probs > 0.5).int()

    acc = (preds == labels.int()).float().mean().item()

    from sklearn.metrics import (
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
        balanced_accuracy_score,
    )
    bal_acc = float(balanced_accuracy_score(labels.numpy(), preds.numpy()))

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds.numpy(), average="binary", zero_division=0
    )

    # --- new metrics ---
    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(labels.numpy(), probs.numpy())
    except ValueError:
        ap = float("nan")

    return (
        float(np.mean(losses)),
        acc,
        float(prec),
        float(rec),
        float(f1),
        float(auc),
        float(ap),
        false_pos,
        false_neg,
        bal_acc,
    )

# -------------- Training (one day range) --------------
def train_for_day_range(max_day, train_ids, val_ids, test_ids,
                        train_meta, val_meta, test_meta, device, output_dir, image_type='clipped'):
    print(f"\n{'='*70}\nTRAINING WITH DAYS 3–{max_day}\n{'='*70}")

    # ---- ADD/REPLACE THIS SECTION ----
    from torchvision.transforms import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR

    train_tf = transforms.Compose([
        transforms.Resize((384, 384), interpolation=BILINEAR),
        transforms.RandomRotation(degrees=15, fill=128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    ])


    eval_tf = transforms.Compose([
        transforms.Resize((384, 384), interpolation=BILINEAR),
    ])

    train_dataset = OrganoidTimeSeriesDataset(train_ids, train_meta, max_day=max_day, transform=train_tf, image_type=image_type)
    val_dataset   = OrganoidTimeSeriesDataset(val_ids,   val_meta,   max_day=max_day, transform=eval_tf, image_type=image_type)
    test_dataset  = OrganoidTimeSeriesDataset(test_ids,  test_meta,  max_day=max_day, transform=eval_tf, image_type=image_type)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    # ---- END OF INSERT ----

    # class balance from train IDs (sequence-level)
    train_labels = []
    for org_id in train_ids:
        s = str(train_meta[org_id].get("label","")).strip().lower()
        lab = 1 if s in ("good","acceptable","accepted") else 0
        train_labels.append(lab)

    n_good = int(np.sum(train_labels))
    n_bad  = int(len(train_labels) - n_good)
    # avoid div-by-zero
    if n_good == 0: n_good = 1
    if n_bad  == 0: n_bad  = 1
    pos_weight = torch.tensor([n_bad / n_good], device=device, dtype=torch.float32)
    print(f"class balance (train): good={n_good}, bad={n_bad}, pos_weight={pos_weight.item():.3f}")

    model = OrganoidCNN_LSTM().to(device)


    # two phase optimizer setup (we'll swap LR when unfreezing)
    def make_optimizer(lr_cnn, lr_head):
        params_cnn = [p for n,p in model.cnn.named_parameters() if p.requires_grad]
        params_head = [p for n,p in model.named_parameters()
                       if not n.startswith("cnn.") and p.requires_grad]
        groups = []
        if len(params_cnn) > 0:
            groups.append({"params": params_cnn, "lr": lr_cnn})
        if len(params_head) > 0:
            groups.append({"params": params_head, "lr": lr_head})
        return optim.Adam(groups)

    # warmup: CNN frozen → only head gets LR
    optimizer = make_optimizer(lr_cnn=0.0, lr_head=LR_HEAD)
    # replace your criterion with reduction='none' and no pos_weight


    criterion = nn.BCEWithLogitsLoss(
    pos_weight = torch.tensor([max(1.0, n_bad / n_good)], device=device),

    reduction="none"
    )



    # before training loop (you already computed these counts)
    # w_pos = n_bad / n_good       # ~0.87
    # w_neg = n_good / n_bad       # ~1.15  <-- upweight negatives slightly
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0
    history = []  # per-epoch metrics for plotting

    for epoch in range(1, MAX_EPOCHS + 1):
        # unfreeze last blocks after warmup
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_last_blocks()
            optimizer = make_optimizer(lr_cnn=LR_CNN_UNFREEZE, lr_head=LR_HEAD)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            print("→ Unfroze last CNN blocks; using small LR for CNN.")

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for seqs, days, labels, weights, ids in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            seqs   = seqs.to(device)
            days   = days.to(device).float()
            labels = labels.float().to(device)
            weights = weights.to(device).float()

            optimizer.zero_grad()
            logits = model(seqs, days)


            # If you want per-sample weighting, re-enable this:
            # cls_w = labels * w_pos + (1 - labels) * w_neg
            # loss = (loss_raw * weights * cls_w).mean()

            # For now: no weighting, use scalar directly
            loss_raw = criterion(logits, labels)          # shape (B,)
            cls_w = labels * (n_bad / n_good) + (1 - labels) * (n_good / n_bad)
            loss = (loss_raw * weights * cls_w).mean()


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)



        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_ap, val_fp, val_fn, val_bal_acc = evaluate_binary(
            model, val_loader, criterion, device
        )


        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | Train {train_acc:.3f} / {train_loss:.4f} "
            f"| Val {val_acc:.3f} / {val_loss:.4f} "
            f"(P {val_prec:.3f} R {val_rec:.3f} F1 {val_f1:.3f} AUC {val_auc:.3f} AP {val_ap:.3f})"
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
            print("  * new best on val acc")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # test with best
    model.load_state_dict(best_state, strict=True)

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_ap, test_fp, test_fn, test_bal_acc = evaluate_binary(
        model, test_loader, criterion, device
    )

    model_dir = output_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions for confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, days, labels, weights, ids in test_loader:
            seqs = seqs.to(device)
            days = days.to(device).float()
            logits = model(seqs, days)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.int().cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (Test Set):")
    print(f"              Predicted")
    print(f"              Bad    Good")
    print(f"Actual Bad    {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"Actual Good   {cm[1,0]:<6} {cm[1,1]:<6}")

    # Save visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Days 3-{max_day}')
    plt.savefig(model_dir / f'confusion_matrix_days_3-{max_day}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {model_dir / f'confusion_matrix_days_3-{max_day}.png'}")

    # --- Training curves ---
    if history:
        epochs = [h['epoch'] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, [h['train_acc'] for h in history], label='Train Acc')
        ax1.plot(epochs, [h['val_acc'] for h in history], label='Val Acc')
        ax1.axvline(x=WARMUP_EPOCHS + 1, color='gray', linestyle='--', alpha=0.6, label='CNN unfreeze')
        ax1.set(xlabel='Epoch', ylabel='Accuracy', title=f'Accuracy – Days 3–{max_day}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [h['train_loss'] for h in history], label='Train Loss')
        ax2.plot(epochs, [h['val_loss'] for h in history], label='Val Loss')
        ax2.axvline(x=WARMUP_EPOCHS + 1, color='gray', linestyle='--', alpha=0.6, label='CNN unfreeze')
        ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Loss – Days 3–{max_day}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = model_dir / f'training_curves_days_3-{max_day}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Training curves saved → {plot_path}")

    model_path = model_dir / f"model_days_3-{max_day}.pth"
    torch.save({"state_dict": best_state, "max_day": max_day, "best_val_acc": best_val_acc}, model_path)

    print("\nFinal (best-val checkpoint) on TEST")
    print(f"Acc {test_acc:.3f} | F1 {test_f1:.3f} | P {test_prec:.3f} | R {test_rec:.3f} | loss {test_loss:.4f}")
    print(f"Saved → {model_path}")

    del model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    torch.cuda.empty_cache()

    return {
        "max_day": max_day,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_balanced_acc": float(test_bal_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "model_path": str(model_path),
        "val_false_positives": val_fp,
        "val_false_negatives": val_fn,
        "test_false_positives": test_fp,
        "test_false_negatives": test_fn,
    }


# -------------- Orchestrator --------------
def main():
    parser = argparse.ArgumentParser(description='Temporal ablation: LSTM pool')
    parser.add_argument('--output-dir', type=str, default='outputs/cnn_lstm/temporal_ablation_lstm',
                        help='Output directory')
    parser.add_argument('--image-type', type=str, default='clipped', choices=['clipped', 'std'],
                        help='Image variant: clipped (575x575 AR meanfill) or std (512x384)')
    parser.add_argument('--splits-dir', type=str, default='data_splits',
                        help=('Directory holding train/val/test split JSONs. Accepts both '
                              'cohort layout (<dir>/{train,val,test}.json) and legacy '
                              'layout (<dir>/{train,val,test}_idor_series.json). Default: '
                              'data_splits/ (legacy).'))
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    print(f"Splits dir: {args.splits_dir}")
    train_ids, train_meta = load_split_from_json(resolve_split_path(args.splits_dir, 'train'))
    val_ids,   val_meta   = load_split_from_json(resolve_split_path(args.splits_dir, 'val'))
    test_ids,  test_meta  = load_split_from_json(resolve_split_path(args.splits_dir, 'test'))
    print(f"Using image type: {args.image_type}")

    print("\n" + "="*70)
    print("STARTING TEMPORAL ABLATION (LSTM POOL)")
    print("="*70)

    results = []
    for max_day in DAY_RANGES:
        res = train_for_day_range(
            max_day, train_ids, val_ids, test_ids,
            train_meta, val_meta, test_meta, device,
            out_dir / f"days_3-{max_day}",
            image_type=args.image_type
        )
        results.append(res)

    results_path = out_dir / "temporal_ablation_results_lstm.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("TEMPORAL ABLATION SUMMARY")
    print("="*70)
    print(f"{'Day Range':<15} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*70)
    for r in results:
        print(f"3–{str(r['max_day']):<12} {r['best_val_acc']:<12.3f} {r['test_acc']:<12.3f} {r['test_f1']:<12.3f}")

    best = max(results, key=lambda x: x["test_acc"])
    print("\nBest on test:", best)
    print(f"Results saved → {results_path}")

if __name__ == "__main__":
    main()

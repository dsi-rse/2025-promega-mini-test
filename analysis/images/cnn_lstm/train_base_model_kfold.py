#!/usr/bin/env python3
"""
train_base_model_kfold.py — 5-fold CV version of the single-image baseline.

Faithful to the coworker's evaluation approach: instead of one train/val/test
split, pool all organoids and run K-fold cross-validation per day. Each fold
trains on the other folds (with a small inner val split for early stopping) and
predicts the held-out fold; per-day metrics are the out-of-fold aggregate plus
the across-fold mean/std. This averages out the single-split checkpoint-selection
noise that collapsed the strong-aug run.

Reuses the model, dataset, training loop, and augmentation from train_base_model
unchanged — only the CV scaffold is new. Folds are well-grouped (StratifiedGroupKFold
on the base well) so daughter organoids never straddle folds (leakage-safe).

    python analysis/images/cnn_lstm/train_base_model_kfold.py \\
        --image-type clipped --strong-aug --n-folds 5 \\
        --splits-dir data/cohorts/idor_balsel/series \\
        --output-dir .../base_effnet_kfold_strongaug
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import (
    BaselineEfficientNet, SingleDayOrganoidDataset, evaluate, ForegroundColorJitter,
    TARGET_SIZE, DAY_RANGES, BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, PATIENCE, GRAD_CLIP, LR,
    set_seed, _BOUNDARY_DAYS,
)
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json

SEED = 1


def _label(meta, oid):
    s = str((meta.get(oid) or {}).get("label", "")).strip().lower()
    return 1 if s in ("good", "acceptable", "accepted") else 0


def _well(oid):
    # base well id: BA1_96_1_B9_nosplit -> BA1_96_1_B9 (groups daughters together)
    return "_".join(oid.split("_")[:4])


def _make_train_tf(target_day, strong_aug):
    if strong_aug:
        degrees = 0 if float(target_day) in _BOUNDARY_DAYS else 180
        return T.Compose([
            T.Resize(TARGET_SIZE),
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=degrees, translate=(0.1, 0.1), fill=[178, 178, 178]),
            ForegroundColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ])
    return T.Compose([
        T.Resize(TARGET_SIZE), T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5), T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])


def _train_one(train_ds, val_ds, device, pos_weight, select):
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=(device.type == "cuda"))
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = BaselineEfficientNet().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(model.classifier.parameters(), lr=LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == 4:  # biphasic: unfreeze last 2 blocks, drop LR (same as single-split)
            model.unfreeze_backbone()
            opt = optim.Adam(model.parameters(), lr=LR * 0.1)
            sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        model.train()
        for imgs, labels, _ in tl:
            imgs = imgs.to(device); labels = labels.to(device)
            opt.zero_grad(); logits = model(imgs); loss = crit(logits, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        vloss, vacc, vp, vr, vf, vauc, vap, _, _, vbal = evaluate(model, vl, crit, device)
        sch.step(vloss)
        score = vbal if select == "bal" else vacc
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state, strict=True)
    return model


@torch.no_grad()
def _predict(model, ds, device):
    model.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    out = {}
    for imgs, labels, ids in loader:
        p = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
        for oid, pr, lab in zip(ids, np.atleast_1d(p), labels.numpy()):
            out[oid] = (float(pr), int(lab))
    return out


def run_day(day, ids, meta, device, args):
    y = np.array([_label(meta, o) for o in ids])
    groups = np.array([_well(o) for o in ids])
    if len(np.unique(y)) < 2 or np.bincount(y).min() < args.n_folds:
        print(f"  day {day}: too few of a class for {args.n_folds}-fold, skipping")
        return None
    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    train_tf = _make_train_tf(day, args.strong_aug)
    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    oof, fold_bal = {}, []
    for fi, (tr_idx, te_idx) in enumerate(skf.split(ids, y, groups)):
        set_seed(SEED + fi)
        tr = [ids[i] for i in tr_idx]; te = [ids[i] for i in te_idx]
        tr_y = [_label(meta, o) for o in tr]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + fi)
        it_idx, iv_idx = next(sss.split(tr, tr_y))
        itr = [tr[i] for i in it_idx]; iv = [tr[i] for i in iv_idx]
        train_ds = SingleDayOrganoidDataset(itr, meta, day, transform=train_tf,
                                            image_type=args.image_type, bbox_crop=args.bbox_crop)
        val_ds = SingleDayOrganoidDataset(iv, meta, day, transform=eval_tf,
                                          image_type=args.image_type, bbox_crop=args.bbox_crop)
        test_ds = SingleDayOrganoidDataset(te, meta, day, transform=eval_tf,
                                           image_type=args.image_type, bbox_crop=args.bbox_crop)
        if len(train_ds) == 0 or len(test_ds) == 0:
            continue
        tl_labels = [s["label"] for s in train_ds.samples]
        ng = max(sum(tl_labels), 1); nb = max(len(tl_labels) - sum(tl_labels), 1)
        pw = torch.tensor([(nb / ng) * args.pos_weight_scale], device=device)
        model = _train_one(train_ds, val_ds, device, pw, args.select)
        preds = _predict(model, test_ds, device)
        oof.update(preds)
        yy = [v[1] for v in preds.values()]; pp = [1 if v[0] > 0.5 else 0 for v in preds.values()]
        if len(set(yy)) > 1:
            fold_bal.append(balanced_accuracy_score(yy, pp))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  day {day} fold {fi+1}/{args.n_folds}: test n={len(preds)} "
              f"bal_acc={fold_bal[-1]:.3f}" if fold_bal else f"  day {day} fold {fi+1}: (single-class fold)")
    if not oof:
        return None
    yy = [v[1] for v in oof.values()]; probs = [v[0] for v in oof.values()]
    pp = [1 if p > 0.5 else 0 for p in probs]
    res = {
        "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
        "balanced_accuracy_mean": float(np.mean(fold_bal)) if fold_bal else float("nan"),
        "balanced_accuracy_std": float(np.std(fold_bal)) if fold_bal else float("nan"),
        "roc_auc": float(roc_auc_score(yy, probs)) if len(set(yy)) > 1 else float("nan"),
        "n": len(oof), "n_folds": len(fold_bal),
    }
    print(f"  day {day}: OOF bal_acc {res['balanced_accuracy']:.3f}  "
          f"fold mean {res['balanced_accuracy_mean']:.3f}±{res['balanced_accuracy_std']:.3f}  "
          f"AUC {res['roc_auc']:.3f}  (n={res['n']})")
    return res


def main():
    ap = argparse.ArgumentParser(description="5-fold CV single-image baseline")
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--splits-dir", default="data/cohorts/idor_balsel/series")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--strong-aug", action="store_true")
    ap.add_argument("--bbox-crop", action="store_true")
    ap.add_argument("--pos-weight-scale", type=float, default=1.0)
    ap.add_argument("--select", default="bal", choices=["bal", "acc"],
                    help="Inner-val checkpoint metric: bal (balanced acc, default) or acc.")
    args = ap.parse_args()
    if args.strong_aug and args.image_type != "clipped":
        print("[warn] --strong-aug fill=178 assumes --image-type clipped")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   folds: {args.n_folds}   strong_aug: {args.strong_aug}")

    # Pool train+val+test into one organoid set + one meta dict.
    from analysis.images.cnn_lstm.organoid_dataset import resolve_split_path
    meta, ids = {}, []
    for phase in ["train", "val", "test"]:
        p = resolve_split_path(args.splits_dir, phase)
        pids, pmeta = load_split_from_json(p)
        meta.update(pmeta); ids.extend(pids)
    ids = sorted(set(ids))
    y = [_label(meta, o) for o in ids]
    print(f"Pooled organoids: {len(ids)}  ({sum(y)} Acceptable, {len(y)-sum(y)} Not)")

    results = {}
    for day in DAY_RANGES:
        print(f"\n{'='*60}\nDAY {day} — {args.n_folds}-fold CV\n{'='*60}")
        r = run_day(day, ids, meta, device, args)
        if r:
            results[str(day)] = r

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "baseline_kfold_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
    print(f"\n{'day':>6} {'OOF bal':>8} {'fold mean±std':>16} {'AUC':>6}")
    for day in DAY_RANGES:
        r = results.get(str(day))
        if r:
            print(f"{day:>6} {r['balanced_accuracy']:>8.3f} "
                  f"{r['balanced_accuracy_mean']:>7.3f}±{r['balanced_accuracy_std']:<7.3f} {r['roc_auc']:>6.3f}")


if __name__ == "__main__":
    main()

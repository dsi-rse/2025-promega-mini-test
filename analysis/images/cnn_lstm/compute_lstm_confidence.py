#!/usr/bin/env python3
"""
compute_lstm_confidence.py — per-organoid confidence for the temporal LSTM model.

Runs the trained EffNet+LSTM checkpoint over the test sequences and writes each
organoid's predicted probability, prediction, correctness, and confidence
(|p-0.5|). No training and no Grad-CAM — inference only. Join the output to the
size (mask_frac) column from the base focus CSV to check whether the temporal
model's confidence also tracks organoid size/maturity.

Run on the cluster (needs the checkpoint + image sequences + GPU):
    conda activate /net/projects2/promega
    python analysis/images/cnn_lstm/compute_lstm_confidence.py \\
        --label idor_balsel --day 30 --image-type clipped \\
        --out lstm_conf_idor_balsel_Dy30.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_temporal_ablation_lstm import OrganoidCNN_LSTM
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, load_split_from_json,
)


def day_str(day: float) -> str:
    return str(int(day)) if float(day) == int(day) else str(day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="checkpoint cohort label, e.g. idor_balsel")
    ap.add_argument("--split-label", default=None, help="test-split label if different from --label")
    ap.add_argument("--day", type=float, default=30)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_day = day_str(args.day)
    split_label = args.split_label or args.label

    test_json = args.cohorts_dir / split_label / "series" / "test.json"
    test_ids, test_meta = load_split_from_json(test_json)
    eval_tf = transforms.Compose([transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)])
    ds = OrganoidTimeSeriesDataset(test_ids, test_meta, max_day=args.day,
                                   transform=eval_tf, image_type=args.image_type)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    ckpt = args.runs_root / args.label / "temporal_ablation_lstm" / f"days_3-{ds_day}" / f"model_days_3-{ds_day}.pth"
    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    state = state.get("state_dict", state)
    model = OrganoidCNN_LSTM().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for seqs, days, labels, weights, ids in loader:
            seqs = seqs.to(device); days = days.to(device).float()
            logits = model(seqs, days)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labs = labels.view(-1).cpu().numpy()
            for oid, p, lab in zip(ids, probs, labs):
                pred = int(p > 0.5); lab = int(lab)
                rows.append({
                    "organoid_id": oid,
                    "true_label": "Acceptable" if lab == 1 else "Not Acceptable",
                    "prob_acceptable": round(float(p), 4),
                    "pred": "Acceptable" if pred == 1 else "Not Acceptable",
                    "correct": int(pred == lab),
                    "confidence": round(abs(float(p) - 0.5), 4),
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nWrote {len(rows)} organoids -> {args.out}   (accuracy {acc*100:.0f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_embeddings.py — dump frozen CNN image embeddings per organoid per day.

Runs every organoid image through a FIXED, ImageNet-pretrained EfficientNet-B0
(classifier removed -> 1280-d pooled vector). The encoder is deliberately NOT
fine-tuned on the outcome labels: an outcome-trained backbone would leak label
information into the features and bias the state-vs-trajectory comparison. A
fixed encoder keeps every feature set (state, change, trajectory) honest.

Pools train+val+test so downstream analysis can do organoid-level CV on the full
labeled set (~140 organoids), not just the 27-organoid test split.

Inference only — no training, no Grad-CAM. Fast. Run on the cluster (needs images):
    conda activate /net/projects2/promega
    python analysis/images/cnn_lstm/extract_embeddings.py \\
        --cohort idor_balsel --image-type clipped \\
        --out embeddings_idor_balsel.npz

Output npz arrays (aligned, one row per organoid-day):
    organoid_ids (str), days (float), labels (int 1=Acceptable), splits (str),
    Z (float32, N x 1280)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, load_split_from_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="idor_balsel")
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--splits", default="train,val,test",
                    help="Which splits to pool (default all three).")
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Fixed ImageNet encoder (frozen, no classifier -> 1280-d pooled features).
    enc = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    enc.classifier = torch.nn.Identity()
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)

    eval_tf = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
    ])

    ids_out, days_out, labels_out, splits_out, Z = [], [], [], [], []

    for split in args.splits.split(","):
        test_json = args.cohorts_dir / args.cohort / "series" / f"{split}.json"
        org_ids, meta = load_split_from_json(test_json)
        ds = OrganoidTimeSeriesDataset(org_ids, meta, max_day=None,
                                       transform=eval_tf, image_type=args.image_type)
        print(f"[{split}] {len(ds)} organoids")
        for i in range(len(ds)):
            seq, days_norm, label, weight, oid = ds[i]  # seq: (T,C,H,W)
            # recover raw days for this organoid, in dataset order
            raw_days = [tp["mdl_day"] for tp in ds.series_metadata[oid]["timepoints"]]
            with torch.no_grad():
                z = enc(seq.to(device)).cpu().numpy()   # (T, 1280)
            lab = int(label.item()) if hasattr(label, "item") else int(label)
            for t in range(z.shape[0]):
                ids_out.append(oid)
                days_out.append(float(raw_days[t]) if t < len(raw_days) else float("nan"))
                labels_out.append(lab)
                splits_out.append(split)
                Z.append(z[t])

    Z = np.asarray(Z, dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        organoid_ids=np.array(ids_out),
        days=np.array(days_out, dtype=np.float32),
        labels=np.array(labels_out, dtype=np.int64),
        splits=np.array(splits_out),
        Z=Z,
    )
    n_org = len(set(ids_out))
    print(f"\nWrote {Z.shape[0]} organoid-day embeddings ({n_org} organoids, "
          f"dim={Z.shape[1]}) -> {args.out}")


if __name__ == "__main__":
    main()

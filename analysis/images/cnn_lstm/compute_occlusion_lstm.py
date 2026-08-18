#!/usr/bin/env python3
"""
compute_occlusion_lstm.py — per-day occlusion importance for the temporal LSTM.

For each TEST organoid, runs the trained EffNet+LSTM once to get the full-sequence
probability, then re-runs the model with each day's frame occluded (replaced by the
ImageNet mean, i.e. zero in normalized space) and records how much the predicted
probability moves. A day the model relies on produces a large swing.

    importance(day) = | p_full - p_occluded(day) |

Inference only — no training, no Grad-CAM. Uses the SAME checkpoint, dataset, and
model as compute_lstm_confidence.py, so results match the balanced-accuracy re-runs.

Run on the cluster (needs checkpoint + image sequences + GPU):
    conda activate /net/projects2/promega
    python analysis/images/cnn_lstm/compute_occlusion_lstm.py \\
        --label idor_balsel --day 30 --image-type clipped \\
        --out occlusion_idor_balsel_Dy30.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
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


def days_used_for(ds, organoid_id, max_day):
    """Reconstruct the raw mdl_day sequence for one organoid, in dataset order."""
    out = []
    for tp in ds.series_metadata[organoid_id]["timepoints"]:
        if max_day is not None and tp["mdl_day"] > max_day:
            break
        out.append(tp["mdl_day"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="checkpoint cohort label, e.g. idor_balsel")
    ap.add_argument("--split-label", default=None, help="test-split label if different from --label")
    ap.add_argument("--day", type=float, default=30)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--order", default="chrono", choices=["chrono", "reverse", "shuffle"],
                    help="Frame order before occlusion. reverse/shuffle test RECENCY: each "
                         "frame keeps its true day tag, only its sequence position changes. "
                         "If importance follows position (not day), the model is recency-driven. "
                         "Note: reordering is out-of-distribution, so absolute predictions "
                         "degrade — read the importance-by-position pattern, not accuracy.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --order shuffle.")
    ap.add_argument("--occlude-set", default=None,
                    help="Comma-separated days to occlude JOINTLY (redundancy test), e.g. "
                         "'28,30'. Adds one row per organoid with day='set:...'. Compare the "
                         "joint swing to the single-day swings: if joint >> each single, the "
                         "days are substitutable/redundant.")
    ap.add_argument("--keep-last-k", type=int, default=None,
                    help="TRUNCATION test: run the model on only the last K frames and compare "
                         "to the full-sequence prediction (no occlusion). Writes prob_full, "
                         "prob_lastk, and whether the hard call flipped. K=1 tests whether the "
                         "model is effectively a single-last-frame classifier.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    occ_set = [float(x) for x in args.occlude_set.split(",")] if args.occlude_set else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_day = day_str(args.day)
    split_label = args.split_label or args.label

    test_json = args.cohorts_dir / split_label / "series" / "test.json"
    test_ids, test_meta = load_split_from_json(test_json)
    eval_tf = transforms.Compose([transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)])
    ds = OrganoidTimeSeriesDataset(test_ids, test_meta, max_day=args.day,
                                   transform=eval_tf, image_type=args.image_type)
    # batch_size=1: sequences have variable length; occlude one frame at a time.
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ckpt = args.runs_root / args.label / "temporal_ablation_lstm" / f"days_3-{ds_day}" / f"model_days_3-{ds_day}.pth"
    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    state = state.get("state_dict", state)
    model = OrganoidCNN_LSTM().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rng = np.random.RandomState(args.seed)
    rows = []
    with torch.no_grad():
        for seqs, days, labels, weights, ids in loader:
            seqs = seqs.to(device); days = days.to(device).float()
            oid = ids[0]
            raw_days = days_used_for(ds, oid, args.day)
            T = seqs.shape[1]

            # Reorder frames (with their day tags) to test recency. Each frame keeps
            # its true day embedding; only its position in the sequence changes.
            if args.order == "reverse":
                perm = list(range(T))[::-1]
            elif args.order == "shuffle":
                perm = list(rng.permutation(T))
            else:
                perm = list(range(T))
            if perm != list(range(T)):
                seqs = seqs[:, perm]; days = days[:, perm]
                raw_days = [raw_days[i] if i < len(raw_days) else None for i in perm]

            p_full = torch.sigmoid(model(seqs, days)).item()

            # Truncation test: predict from only the last K frames, compare to full.
            if args.keep_last_k is not None:
                k = min(args.keep_last_k, T)
                p_lastk = torch.sigmoid(model(seqs[:, -k:], days[:, -k:])).item()
                lab = int(labels.item())
                rows.append({
                    "organoid_id": oid,
                    "k": k,
                    "true_label": "Acceptable" if lab == 1 else "Not Acceptable",
                    "prob_full": round(p_full, 4),
                    "prob_lastk": round(p_lastk, 4),
                    "prob_diff": round(abs(p_full - p_lastk), 4),
                    "flip": int((p_full > 0.5) != (p_lastk > 0.5)),
                })
                continue  # truncation mode: skip per-frame occlusion

            for t in range(T):
                occ = seqs.clone()
                occ[:, t] = 0.0  # ImageNet mean in normalized space = neutral frame
                p_occ = torch.sigmoid(model(occ, days)).item()
                rows.append({
                    "position": t,               # 0 = first fed, T-1 = last (readout)
                    "organoid_id": oid,
                    "day": raw_days[t] if t < len(raw_days) else "",
                    "prob_full": round(p_full, 4),
                    "prob_occluded": round(p_occ, 4),
                    "importance": round(abs(p_full - p_occ), 4),
                })

            # Joint occlusion of a set of days (redundancy test).
            if occ_set:
                idxs = [i for i, d in enumerate(raw_days) if d in occ_set]
                if idxs:
                    occ = seqs.clone()
                    for i in idxs:
                        occ[:, i] = 0.0
                    p_occ = torch.sigmoid(model(occ, days)).item()
                    rows.append({
                        "position": "",
                        "organoid_id": oid,
                        "day": "set:" + ",".join(day_str(d) for d in occ_set),
                        "prob_full": round(p_full, 4),
                        "prob_occluded": round(p_occ, 4),
                        "importance": round(abs(p_full - p_occ), 4),
                    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({len(set(r['organoid_id'] for r in rows))} organoids) -> {args.out}")


if __name__ == "__main__":
    main()

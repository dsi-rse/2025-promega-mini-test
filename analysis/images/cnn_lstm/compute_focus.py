#!/usr/bin/env python3
"""
compute_focus.py — exact Grad-CAM content-focus for the single-image base model.

For each TEST organoid at a given day, loads the trained base checkpoint, computes
Grad-CAM on the last conv layer, and measures the FRACTION of CAM "energy" inside the
organoid mask, plus the size-corrected ENRICHMENT (= focus / mask_area_fraction).

    focus       in [0,1]   fraction of CAM on the organoid
    mask_frac              organoid's share of the frame
    enrichment  = focus/mask_frac   (>1 = denser on organoid than chance)

Also writes prob/confidence/correct per organoid.

Examples:
    # single day
    python analysis/images/cnn_lstm/compute_focus.py --label idor_balsel --day 30 \\
        --image-type clipped --out focus_idor_balsel_Dy30.csv

    # ALL days (adds a 'day' column) -> does focus change over development?
    python analysis/images/cnn_lstm/compute_focus.py --label idor_balsel --all-days \\
        --image-type clipped --out focus_idor_balsel_alldays.csv

    # bbox model (size removed) — prob/confidence valid, focus meaningless
    python analysis/images/cnn_lstm/compute_focus.py --label idor_bbox_balsel \\
        --split-label idor_balsel --day 30 --bbox-crop --out focus_bbox.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import (
    BaselineEfficientNet, SingleDayOrganoidDataset, TARGET_SIZE, DAY_RANGES,
)
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json
from torchvision import transforms as T


def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def day_str(day: float) -> str:
    return str(int(day)) if float(day) == int(day) else str(day)


def load_mask(mask_path, hw):
    m = Image.open(mask_path).convert("L").resize((hw[1], hw[0]), Image.NEAREST)
    a = np.asarray(m).astype(np.float32)
    return (a > (0.5 * a.max() if a.max() > 0 else 0.5)).astype(np.float32)


def process_day(day, test_ids, test_meta, args, device):
    """Compute focus/confidence for every test organoid at one day. Returns rows."""
    ds_day = day_str(day)
    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    ds = SingleDayOrganoidDataset(test_ids, test_meta, day, transform=eval_tf,
                                  image_type=args.image_type, bbox_crop=args.bbox_crop)
    ckpt = args.runs_root / args.label / args.model_subdir / f"day_{ds_day}" / f"model_day_{ds_day}.pth"
    if not ckpt.exists():
        print(f"  [skip] no checkpoint for day {day}: {ckpt}")
        return [], 0
    state = torch.load(ckpt, map_location=device); state = state.get("state_dict", state)
    model = BaselineEfficientNet().to(device)
    model.load_state_dict(state, strict=True); model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = find_last_conv(model)
    acts, grads = {}, {}
    def fwd_hook(mod, inp, out):
        acts["v"] = out
        if out.requires_grad:
            out.register_hook(lambda g: grads.__setitem__("v", g))
    h = target_layer.register_forward_hook(fwd_hook)

    rows, n_nomask = [], 0
    for i in range(len(ds)):
        samp = ds.samples[i]
        if not samp.get("mask_path"):
            n_nomask += 1
            continue
        x, label, org_id = ds[i]
        x = x.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True); acts.clear(); grads.clear()
        logit = model(x); prob = torch.sigmoid(logit).item(); logit.backward()
        a = acts["v"]; g = grads["v"]
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=TARGET_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        mask = load_mask(samp["mask_path"], TARGET_SIZE)
        denom = cam.sum()
        focus = float((cam * mask).sum() / denom) if denom > 0 else float("nan")
        mask_frac = float(mask.mean())
        enrichment = float(focus / mask_frac) if mask_frac > 0 else float("nan")
        lab = int(label.item()) if hasattr(label, "item") else int(label)
        pred = int(prob > 0.5)
        rows.append({
            "day": day, "organoid_id": org_id,
            "true_label": "Acceptable" if lab == 1 else "Not Acceptable",
            "prob_acceptable": round(prob, 4),
            "pred": "Acceptable" if pred == 1 else "Not Acceptable",
            "correct": int(pred == lab), "confidence": round(abs(prob - 0.5), 4),
            "focus": round(focus, 4), "mask_frac": round(mask_frac, 4),
            "enrichment": round(enrichment, 3),
        })
    h.remove()
    return rows, n_nomask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--day", type=float, default=30)
    ap.add_argument("--all-days", action="store_true",
                    help="Loop over all base day checkpoints (adds a 'day' column).")
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--split-label", default=None,
                    help="Test-split cohort label if different from --label.")
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--model-subdir", default="base_effnet",
                    help="Checkpoint folder under runs_root/label/ (e.g. base_effnet, "
                         "base_effnet_strongaug) — lets you Grad-CAM the augmented model.")
    ap.add_argument("--bbox-crop", action="store_true",
                    help="Crop to mask bbox + letterbox (size removed). focus/enrichment "
                         "meaningless under bbox; only prob/confidence/correct are valid.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_label = args.split_label or args.label
    test_json = args.cohorts_dir / split_label / "series" / "test.json"
    test_ids, test_meta = load_split_from_json(test_json)

    days = list(DAY_RANGES) if args.all_days else [args.day]
    all_rows = []
    for day in days:
        rows, nn = process_day(day, test_ids, test_meta, args, device)
        all_rows += rows
        if rows:
            fs = [r["enrichment"] for r in rows if r["enrichment"] == r["enrichment"]]
            print(f"day {day}: {len(rows)} organoids  mean enrichment {np.mean(fs):.2f}x  ({nn} no-mask)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader(); wtr.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

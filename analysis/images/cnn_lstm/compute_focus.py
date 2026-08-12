#!/usr/bin/env python3
"""
compute_focus.py — exact Grad-CAM content-focus for the single-image base model.

For each TEST organoid at a given day, this loads the trained base checkpoint,
computes Grad-CAM on the last conv layer, and measures the FRACTION of CAM
"energy" that falls inside the organoid's segmentation mask:

    focus = sum(CAM * mask) / sum(CAM)      in [0, 1]

1.0 = all activation on the organoid, 0.0 = all on the background/well.
Unlike the earlier estimate (decoded from rendered heatmap PNGs), this uses the
raw CAM array and the real mask, so the numbers are exact.

Writes a CSV with one row per organoid:
    organoid_id, true_label, prob_acceptable, pred, correct, confidence, focus

Run on the cluster (needs the checkpoint, images/masks, and a GPU), e.g.:
    conda activate /net/projects2/promega
    python analysis/images/cnn_lstm/compute_focus.py \\
        --label idor_balsel --day 30 \\
        --runs-root /net/projects2/promega/project_data/model_tests/lstm_runs \\
        --cohorts-dir data/cohorts \\
        --image-type clipped \\
        --out focus_idor_balsel_Dy30.csv
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
    BaselineEfficientNet, SingleDayOrganoidDataset, TARGET_SIZE,
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
    """Load a mask, resize to (H,W), return a float {0,1} array."""
    m = Image.open(mask_path).convert("L").resize((hw[1], hw[0]), Image.NEAREST)
    a = np.asarray(m).astype(np.float32)
    return (a > (0.5 * a.max() if a.max() > 0 else 0.5)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="cohort label, e.g. idor_balsel")
    ap.add_argument("--day", type=float, default=30)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_day = day_str(args.day)

    # --- test split for this cohort ---
    test_json = args.cohorts_dir / args.label / "series" / "test.json"
    test_ids, test_meta = load_split_from_json(test_json)
    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    ds = SingleDayOrganoidDataset(test_ids, test_meta, args.day, transform=eval_tf,
                                  image_type=args.image_type, bbox_crop=False)

    # --- model ---
    ckpt = args.runs_root / args.label / "base_effnet" / f"day_{ds_day}" / f"model_day_{ds_day}.pth"
    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    state = state.get("state_dict", state)
    model = BaselineEfficientNet().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = find_last_conv(model)
    acts, grads = {}, {}
    def fwd_hook(mod, inp, out):
        acts["v"] = out
        if out.requires_grad:
            out.register_hook(lambda g: grads.__setitem__("v", g))
    target_layer.register_forward_hook(fwd_hook)

    rows = []
    n_nomask = 0
    for i in range(len(ds)):
        samp = ds.samples[i]
        if not samp.get("mask_path"):
            n_nomask += 1
            continue
        x, label, org_id = ds[i]
        x = x.unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        acts.clear(); grads.clear()
        logit = model(x)                       # (1,)
        prob = torch.sigmoid(logit).item()
        logit.backward()

        a = acts["v"]; g = grads["v"]          # (1,C,h,w)
        w = g.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = F.relu((w * a).sum(dim=1, keepdim=True))            # (1,1,h,w)
        cam = F.interpolate(cam, size=TARGET_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        mask = load_mask(samp["mask_path"], TARGET_SIZE)
        denom = cam.sum()
        focus = float((cam * mask).sum() / denom) if denom > 0 else float("nan")

        lab = int(label.item()) if hasattr(label, "item") else int(label)
        pred = int(prob > 0.5)
        rows.append({
            "organoid_id": org_id,
            "true_label": "Acceptable" if lab == 1 else "Not Acceptable",
            "prob_acceptable": round(prob, 4),
            "pred": "Acceptable" if pred == 1 else "Not Acceptable",
            "correct": int(pred == lab),
            "confidence": round(abs(prob - 0.5), 4),
            "focus": round(focus, 4),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    print(f"\nWrote {len(rows)} organoids -> {args.out}  ({n_nomask} skipped: no mask)")
    if rows:
        fs = [r["focus"] for r in rows if r["focus"] == r["focus"]]
        print(f"focus: mean={np.mean(fs):.3f}  median={np.median(fs):.3f}  "
              f"range {min(fs):.3f}-{max(fs):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_gradcam_rotation_check.py — rotation-equivariance check for Grad-CAM.

For each selected organoid we render the day-N image at four orientations
(0°, 90°, 180°, 270°), run Grad-CAM on each, and overlay the heatmap in the
rotated frame. We also counter-rotate every CAM back to the 0° frame and
compute cosine similarity vs. the 0° CAM — a single "rotation consistency"
score per organoid.

Reading the output
------------------
- Heatmap travels WITH the organoid as you flip the image → model is
  attending to morphology (content-tracking). Good.
- Heatmap stays in the same image corner / edge regardless of rotation →
  model is attending to spurious position features (well edge, lighting
  artifact). Brittle.
- Consistency score: 1.0 = perfectly content-tracking,
                     0.0 = entirely position-tracking.
  EfficientNet isn't rotation-equivariant by architecture, so even good
  content-tracking attention won't be a perfect 1.0. Use ~0.7+ as "trust",
  <0.3 as "suspect."

Usage
-----
    python analysis/images/cnn_lstm/make_gradcam_rotation_check.py \\
        --label idor --day 30 --top-n 4 \\
        --selection-mode confident_correct

Outputs to --plots-dir:
    gradcam_rotation_<label>_Dy<day>_<mode>/
        <organoid>_rotation.png
        rotation_consistency_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Shared helpers (mirrors make_gradcam_base_effnet.py)
# ============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_day_image_path(rec, day, image_type="clipped"):
    for tp in rec.get("timepoints", []):
        if float(tp.get("mdl_day")) == float(day):
            p = tp.get("img_paths", {}).get(image_type)
            if p:
                return Path(p)
    return None


def load_rgb_image(path, preprocess):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    arr_display = arr / 255.0 if arr.max() > 1 else arr.copy()
    pil = Image.fromarray((arr_display * 255).astype(np.uint8))
    x = preprocess(pil).unsqueeze(0)
    return arr_display, x


def find_last_conv_layer(model):
    """Return (name, module) of the deepest Conv2d in the model."""
    last_name, last_module = None, None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_name, last_module = name, m
    return last_name, last_module


def overlay_cam(img, cam, alpha=0.4):
    """Return (heat_rgb, overlay_rgb) for plotting.

    If cam was computed at the model's input resolution (typically smaller
    than the native image), upsample it back to the image size so the
    overlay broadcast works.
    """
    if cam.shape[:2] != img.shape[:2]:
        from PIL import Image as _PILImage
        cam_pil = _PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((img.shape[1], img.shape[0]), _PILImage.BILINEAR)
        cam = np.asarray(cam_pil).astype(np.float32) / 255.0
    import matplotlib.cm as cm
    heat = cm.jet(cam)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heat
    overlay = np.clip(overlay, 0, 1)
    return heat, overlay


# ============================================================
# Selection — same logic + columns as make_gradcam_base_effnet.py
# ============================================================

def select_organoids(misses, args):
    """Same modes as make_gradcam_base_effnet.py."""
    if args.filter_label != "none":
        misses = misses[misses["true_label"] == args.filter_label].copy()

    if args.selection_mode == "missed":
        return misses.sort_values(
            ["miss_rate", "total_misses", "organoid_id"],
            ascending=[False, False, True]
        ).head(args.top_n)
    if args.selection_mode == "lowest":
        return misses.sort_values(
            ["total_misses", "miss_rate", "organoid_id"],
            ascending=[True, True, True]
        ).head(args.top_n)
    if args.selection_mode == "perfect":
        return misses[misses["total_misses"] == 0].sort_values(
            "organoid_id"
        ).head(args.top_n)
    if args.selection_mode in ("confident_correct", "confident_wrong",
                               "confident_any", "uncertain"):
        if "confidence" not in misses.columns:
            raise ValueError(
                f"Mode '{args.selection_mode}' needs a confidence column. "
                "Run the pre-pass first."
            )
        sub = misses.copy()
        if args.selection_mode == "confident_correct":
            sub = sub[sub["pred_correct"] == True]
        elif args.selection_mode == "confident_wrong":
            sub = sub[sub["pred_correct"] == False]
        # uncertain: ascending (lowest first); others: descending
        ascending = args.selection_mode == "uncertain"
        return sub.sort_values(
            ["confidence", "organoid_id"], ascending=[ascending, True]
        ).head(args.top_n)
    raise ValueError(f"Unknown selection mode: {args.selection_mode}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="idor")
    parser.add_argument("--day", type=float, default=30)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument(
        "--selection-mode", default="confident_correct",
        choices=["missed", "lowest", "perfect",
                 "confident_correct", "confident_wrong", "confident_any",
                 "uncertain"],
        help=(
            "'uncertain' = lowest confidence first (model was wavering near 0.5)."
        ),
    )
    parser.add_argument(
        "--filter-label", default="none",
        choices=["none", "Acceptable", "Not Acceptable"],
    )
    parser.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    parser.add_argument("--run-dir", type=Path,
                        default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    parser.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    parser.add_argument("--plots-dir", type=Path,
                        default=Path("/net/projects2/promega/project_data/amanda_test/model_plots"))
    args = parser.parse_args()

    label = args.label
    day_str = f"{args.day:g}"

    ckpt_path  = args.run_dir / label / "base_effnet" / f"day_{day_str}" / f"model_day_{day_str}.pth"
    test_json  = args.cohorts_dir / label / "series" / "test.json"
    misses_csv = args.plots_dir / f"misses_{label}.csv"

    out_dir = args.plots_dir / f"gradcam_rotation_{label}_Dy{day_str}_{args.selection_mode}"
    if args.filter_label != "none":
        out_dir = args.plots_dir / (
            f"gradcam_rotation_{label}_Dy{day_str}_{args.selection_mode}"
            f"_{args.filter_label.replace(' ', '_')}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] label = {label} | day = {day_str} | mode = {args.selection_mode}")
    print(f"[info] checkpoint = {ckpt_path}")
    print(f"[info] out dir = {out_dir}")
    for p in (ckpt_path, test_json, misses_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    sys.path.append(str(Path(".").resolve()))
    from analysis.images.cnn_lstm.train_base_model import BaselineEfficientNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    model = BaselineEfficientNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    conv_name, target_layer = find_last_conv_layer(model)
    print(f"[info] Grad-CAM target layer = {conv_name}")

    activations, gradients = {}, {}

    def forward_hook(module, inp, out):
        activations["value"] = out
        if out.requires_grad:
            def save_grad(grad):
                gradients["value"] = grad
            out.register_hook(save_grad)

    target_layer.register_forward_hook(forward_hook)

    # IMPORTANT: must match the trainer's eval pipeline. base_effnet was
    # trained at (H=384, W=512); native clipped images are 575x575.
    preprocess = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def make_gradcam(x):
        x = x.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        activations.clear(); gradients.clear()

        out = model(x)
        if out.ndim == 2 and out.shape[1] == 2:
            pred_idx = int(out.argmax(dim=1).item())
            score = out[0, pred_idx]
            prob_accept = torch.softmax(out, dim=1)[0, 1].item()
        else:
            logit = out.view(-1)[0]
            prob_accept = torch.sigmoid(logit).item()
            score = logit if prob_accept >= 0.5 else -logit

        score.backward()
        acts = activations["value"]
        grads = gradients["value"]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, prob_accept

    test_data = load_json(test_json)
    misses = pd.read_csv(misses_csv)

    # -------- Confidence pre-pass --------
    @torch.no_grad()
    def _prob_only(x):
        x = x.to(device)
        out = model(x)
        if out.ndim == 2 and out.shape[1] == 2:
            return torch.softmax(out, dim=1)[0, 1].item()
        return torch.sigmoid(out.view(-1)[0]).item()

    print("[info] computing P(Acceptable) for every test organoid (pre-pass)...")
    probs, correct = {}, {}
    label_to_int = {"Acceptable": 1, "Not Acceptable": 0}
    for oid in misses["organoid_id"]:
        rec = test_data.get(oid)
        if rec is None: continue
        img_path = get_day_image_path(rec, args.day, args.image_type)
        if img_path is None or not img_path.exists(): continue
        _, x = load_rgb_image(img_path, preprocess)
        p = _prob_only(x)
        probs[oid] = p
        true_int = label_to_int.get(rec.get("label"))
        if true_int is not None:
            pred_int = 1 if p >= 0.5 else 0
            correct[oid] = (pred_int == true_int)

    misses["prob_accept"]  = misses["organoid_id"].map(probs)
    misses["confidence"]   = (misses["prob_accept"] - 0.5).abs() * 2
    misses["pred_correct"] = misses["organoid_id"].map(correct)

    selected = select_organoids(misses, args)
    print(f"\n[info] selected organoids (top {args.top_n}):")
    debug_cols = ["organoid_id", "true_label", "n_votes_good", "n_votes_total",
                  "miss_rate", "prob_accept", "confidence", "pred_correct"]
    print(selected[debug_cols].to_string(index=False))
    if selected.empty:
        raise SystemExit("No organoids selected.")

    # -------- Rotation check per organoid --------
    summary_rows = []

    for _, row in selected.iterrows():
        oid = row["organoid_id"]
        rec = test_data.get(oid)
        if rec is None:
            continue
        img_path = get_day_image_path(rec, args.day, args.image_type)
        if img_path is None or not img_path.exists():
            print(f"[warn] missing image: {oid}"); continue

        # Original image (HWC float 0..1)
        orig_img, _ = load_rgb_image(img_path, preprocess)

        angles = [0, 90, 180, 270]
        cams_rotated_frame = []     # CAM in the rotated image's own coordinates
        cams_unrotated_frame = []   # CAM rotated back to 0° frame for similarity
        probs_at_angle = []

        for a in angles:
            # k = number of CCW 90-degree rotations to apply to the image
            k = a // 90
            rot_img = np.rot90(orig_img, k=k).copy()
            pil = Image.fromarray((rot_img * 255).astype(np.uint8))
            x = preprocess(pil).unsqueeze(0)
            cam, p = make_gradcam(x)
            probs_at_angle.append(p)
            cams_rotated_frame.append(cam)
            # Counter-rotate the CAM back to the original frame
            cam_back = np.rot90(cam, k=-k).copy()
            cams_unrotated_frame.append(cam_back)

        # Rotation consistency: cosine sim of each rotated-back CAM vs the 0° one
        c0 = cams_unrotated_frame[0].flatten()
        sims = []
        for c in cams_unrotated_frame[1:]:
            v = c.flatten()
            num = float(np.dot(c0, v))
            den = float(np.linalg.norm(c0) * np.linalg.norm(v))
            sims.append(num / den if den > 0 else 0.0)
        consistency = float(np.mean(sims)) if sims else 0.0

        # ----- Plot 2x4 grid: row 0 = rotated images, row 1 = overlay -----
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
        for i, a in enumerate(angles):
            rot_img = np.rot90(orig_img, k=a // 90).copy()
            _, overlay = overlay_cam(rot_img, cams_rotated_frame[i])
            axes[0, i].imshow(rot_img)
            axes[0, i].set_title(f"{a}°  P(Acc)={probs_at_angle[i]:.2f}", fontsize=10)
            axes[1, i].imshow(overlay)
            axes[1, i].set_title("Grad-CAM overlay", fontsize=9)
            for ax in (axes[0, i], axes[1, i]):
                ax.set_xticks([]); ax.set_yticks([])

        true_label = row["true_label"]
        votes = f"{int(row['n_votes_good'])}/{int(row['n_votes_total'])}"
        pred_ok = "✓" if bool(row.get("pred_correct", False)) else "✗"
        title = (
            f"{oid}   true={true_label} ({votes})   "
            f"pred {pred_ok}   confidence={float(row['confidence']):.2f}   "
            f"rotation_consistency={consistency:.2f}"
        )
        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = out_dir / f"{oid}_rotation.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {out_png}")

        summary_rows.append({
            "organoid_id": oid,
            "true_label": true_label,
            "votes": votes,
            "prob_accept_0deg": probs_at_angle[0],
            "prob_accept_90deg": probs_at_angle[1],
            "prob_accept_180deg": probs_at_angle[2],
            "prob_accept_270deg": probs_at_angle[3],
            "pred_correct": bool(row.get("pred_correct", False)),
            "confidence": float(row["confidence"]),
            "rotation_consistency": consistency,
        })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "rotation_consistency_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[done] summary: {summary_path}")
    print(summary_df[
        ["organoid_id", "pred_correct", "confidence", "rotation_consistency"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

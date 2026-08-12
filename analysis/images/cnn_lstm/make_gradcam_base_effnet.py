#!/usr/bin/env python3

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


def overlay_cam(img, cam, alpha=0.45):
    # If the CAM is at the model's input resolution (post-resize) and the
    # display image is native size, upsample the CAM so shapes match.
    if cam.shape[:2] != img.shape[:2]:
        from PIL import Image as _PILImage
        cam_pil = _PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((img.shape[1], img.shape[0]), _PILImage.BILINEAR)
        cam = np.asarray(cam_pil).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]
    overlay = (1 - alpha) * img + alpha * heat
    overlay = np.clip(overlay, 0, 1)
    return heat, overlay


def find_last_conv_layer(model):
    last_conv_name = None
    target_layer = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
            target_layer = module

    if target_layer is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")

    return last_conv_name, target_layer


def select_organoids(misses, args):
    """
    Rank test-set organoids for Grad-CAM inspection.

    Aggregation-based modes (use cross-variant miss patterns):
        missed   – most misclassified across variants
        lowest   – fewest misclassifications
        perfect  – never misclassified across variants

    Confidence-based modes (use THIS model's P(Acceptable) — require
    prob_accept and 'pred_correct' columns to have been added upstream):
        confident_correct  – model is sure AND right
        confident_wrong    – model is sure AND wrong (most diagnostic failures)
        confident_any      – model is sure (correct or not)
    """
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
                f"Mode '{args.selection_mode}' needs per-organoid probabilities. "
                "Confidence pre-pass was not run."
            )
        sub = misses.copy()
        if args.selection_mode == "confident_correct":
            sub = sub[sub["pred_correct"] == True]
        elif args.selection_mode == "confident_wrong":
            sub = sub[sub["pred_correct"] == False]
        # confident_any and uncertain keep all
        ascending = args.selection_mode == "uncertain"   # lowest first for uncertain
        return sub.sort_values(
            ["confidence", "organoid_id"], ascending=[ascending, True]
        ).head(args.top_n)

    raise ValueError(f"Unknown selection mode: {args.selection_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="idor_minvotes3", help="idor or idor_minvotes3")
    parser.add_argument("--day", type=float, default=30)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--selection-mode",
        default="missed",
        choices=["missed", "lowest", "perfect",
                 "confident_correct", "confident_wrong", "confident_any",
                 "uncertain"],
        help=(
            "Aggregation modes (cross-variant miss patterns): missed | lowest | perfect. "
            "Confidence modes (this model's P(Acceptable), distance from 0.5): "
            "confident_correct | confident_wrong | confident_any | uncertain. "
            "'uncertain' = lowest confidence first (model was wavering near 0.5)."
        ),
    )
    parser.add_argument(
        "--filter-label",
        default="none",
        choices=["none", "Acceptable", "Not Acceptable"]
    )
    parser.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs")
    )
    parser.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("/net/projects2/promega/project_data/amanda_test/model_plots")
    )
    args = parser.parse_args()

    label = args.label
    day_str = f"{args.day:g}"

    ckpt_path = (
        args.run_dir / label / "base_effnet" /
        f"day_{day_str}" / f"model_day_{day_str}.pth"
    )
    test_json = args.cohorts_dir / label / "series" / "test.json"
    misses_csv = args.plots_dir / f"misses_{label}.csv"

    out_dir = args.plots_dir / f"gradcam_{label}_base_Dy{day_str}_{args.selection_mode}"
    if args.filter_label != "none":
        out_dir = args.plots_dir / f"gradcam_{label}_base_Dy{day_str}_{args.selection_mode}_{args.filter_label.replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] label = {label}")
    print(f"[info] day = {day_str}")
    print(f"[info] selection mode = {args.selection_mode}")
    print(f"[info] checkpoint = {ckpt_path}")
    print(f"[info] test json = {test_json}")
    print(f"[info] misses csv = {misses_csv}")
    print(f"[info] out dir = {out_dir}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not test_json.exists():
        raise FileNotFoundError(f"Test JSON not found: {test_json}")
    if not misses_csv.exists():
        raise FileNotFoundError(f"Misses CSV not found: {misses_csv}")

    sys.path.append(str(Path(".").resolve()))
    from analysis.images.cnn_lstm.train_base_model import BaselineEfficientNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    model = BaselineEfficientNet().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # Grad-CAM needs gradients, but this does not retrain or update weights.
    for p in model.parameters():
        p.requires_grad_(True)

    conv_name, target_layer = find_last_conv_layer(model)
    print(f"[info] Grad-CAM target layer = {conv_name}")

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

        # Only register backward hook if grad is actually being tracked.
        # The pre-pass runs under torch.no_grad(), where out.requires_grad
        # is False and register_hook would raise.
        if out.requires_grad:
            def save_grad(grad):
                gradients["value"] = grad
            out.register_hook(save_grad)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    # IMPORTANT: must match the trainer's eval pipeline so probabilities are
    # comparable to what the model saw during training. base_effnet trains on
    # (H=384, W=512) inputs; native clipped images are 575x575.
    preprocess = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def make_gradcam(x):
        x = x.to(device)
        x.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        activations.clear()
        gradients.clear()

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

        if "value" not in activations:
            raise RuntimeError("Activation was not captured.")
        if "value" not in gradients:
            raise RuntimeError("Gradient was not captured.")

        acts = activations["value"]
        grads = gradients["value"]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, prob_accept

    test_data = load_json(test_json)
    misses = pd.read_csv(misses_csv)

    # ------------------------------------------------------------------
    # Pre-pass: compute P(Acceptable) for every organoid in misses df.
    # Adds three columns: prob_accept, confidence (|p - 0.5|*2), pred_correct.
    # Needed for the confident_* selection modes; harmless for the others.
    # ------------------------------------------------------------------
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
        if rec is None:
            continue
        img_path = get_day_image_path(rec, args.day, args.image_type)
        if img_path is None or not img_path.exists():
            continue
        _, x = load_rgb_image(img_path, preprocess)
        p = _prob_only(x)
        probs[oid] = p
        true_int = label_to_int.get(rec.get("label"))
        if true_int is not None:
            pred_int = 1 if p >= 0.5 else 0
            correct[oid] = (pred_int == true_int)

    misses["prob_accept"]  = misses["organoid_id"].map(probs)
    misses["confidence"]   = (misses["prob_accept"] - 0.5).abs() * 2     # 0..1
    misses["pred_correct"] = misses["organoid_id"].map(correct)
    print(f"[info] probs computed for {misses['prob_accept'].notna().sum()} / "
          f"{len(misses)} organoids")

    selected = select_organoids(misses, args)

    print("\n[info] selected organoids:")
    debug_cols = ["organoid_id", "true_label", "n_votes_good",
                  "n_votes_total", "miss_rate", "total_misses"]
    if args.selection_mode.startswith("confident"):
        debug_cols += ["prob_accept", "confidence", "pred_correct"]
    print(selected[debug_cols].to_string(index=False))

    if selected.empty:
        raise SystemExit("No organoids selected. Try --selection-mode lowest or --selection-mode missed.")

    fig_rows = []

    for _, row in selected.iterrows():
        oid = row["organoid_id"]

        if oid not in test_data:
            print(f"[warn] {oid} not found in test JSON")
            continue

        rec = test_data[oid]
        img_path = get_day_image_path(rec, args.day, args.image_type)

        if img_path is None or not img_path.exists():
            print(f"[warn] missing image for {oid} Dy{day_str}: {img_path}")
            continue

        img, x = load_rgb_image(img_path, preprocess)
        cam, prob_accept = make_gradcam(x)
        heat, overlay = overlay_cam(img, cam)

        true_label = row["true_label"]
        votes = f"{int(row['n_votes_good'])}/{int(row['n_votes_total'])}"
        miss_rate = float(row["miss_rate"])
        total_misses = int(row["total_misses"])

        indiv_path = out_dir / f"{oid}_Dy{day_str}_gradcam.png"

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{oid} | true={true_label} | votes={votes} | "
            f"miss_rate={miss_rate:.2f} | total_misses={total_misses} | P(Acceptable)={prob_accept:.2f}",
            fontsize=9
        )

        plt.tight_layout()
        fig.savefig(indiv_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig_rows.append(
            (oid, img, cam, overlay, true_label, votes, miss_rate, total_misses, prob_accept)
        )

    hook_handle.remove()

    if not fig_rows:
        raise SystemExit("No Grad-CAM rows generated.")

    n = len(fig_rows)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.0 * n), squeeze=False)

    for i, (oid, img, cam, overlay, true_label, votes, miss_rate, total_misses, prob_accept) in enumerate(fig_rows):
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 2].imshow(overlay)

        axes[i, 0].set_ylabel(
            f"{oid}\n{true_label}, {votes}\n"
            f"miss={miss_rate:.2f}, n={total_misses}\nP(Acc)={prob_accept:.2f}",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center"
        )
        axes[i, 0].yaxis.set_label_coords(-0.25, 0.5)

        for j in range(3):
            axes[i, j].axis("off")

    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Grad-CAM")
    axes[0, 2].set_title("Overlay")

    fig.suptitle(
        f"Grad-CAM — {label} base EffNet Dy{day_str} — {args.selection_mode}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    summary_path = out_dir / f"gradcam_summary_{label}_base_Dy{day_str}_{args.selection_mode}_top{len(fig_rows)}.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote individual overlays to: {out_dir}")
    print(f"[done] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Augmentation illustration: 3 days × (1 original + 12 augmented) panel.

Picks organoid BA1 96_1 A2 at Dy06, Dy20_5, Dy28 to show the same cell
across timepoints. Applies training augmentation 12 times per image.

Output: figures/augmentation_demo.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.augmentation_demo"
"""

import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from torchvision import transforms as T

warnings.filterwarnings("ignore")

IMG_HEIGHT = 384
IMG_WIDTH  = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
_FILL = [int(v * 255) for v in IMAGENET_MEAN]   # [123, 116, 103]

N_AUG  = 12
SEED   = 42

SAMPLES = [
    ("Dy06",   "BA1 96_1 A2",
     "/net/projects2/promega/2026_04_15_data/intermediate/mean_fill_clip"
     "/BA1_96_1_Dy06_A2_clipped_meanfill_auto_filled.png"),
    ("Dy20.5", "BA1 96_1 A2",
     "/net/projects2/promega/2026_04_15_data/intermediate/mean_fill_clip"
     "/BA1_96_1_Dy20.5_A2_clipped_meanfill_auto_filled.png"),
    ("Dy28",   "BA1 96_1 A2",
     "/net/projects2/promega/2026_04_15_data/intermediate/mean_fill_clip"
     "/BA1_96_1_Dy28_A2_clipped_meanfill_auto_filled.png"),
]

OUT_PATH = Path("figures/augmentation_demo.png")


def _resize_only():
    return T.Compose([T.Resize((IMG_HEIGHT, IMG_WIDTH))])


def _aug_transform(translate=(0.1, 0.1)):
    """Training augmentation without ToTensor/Normalize — returns PIL image."""
    return T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=180, translate=translate, fill=_FILL),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ])


def _to_np(pil_img):
    return np.array(pil_img)


def main():
    n_rows = len(SAMPLES)
    n_cols = 1 + N_AUG   # original + 12 augmented

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 1.7),
        gridspec_kw={"wspace": 0.03, "hspace": 0.12},
    )

    resize_tf = _resize_only()

    for row_i, (day_label, org_id, img_path) in enumerate(SAMPLES):
        pil_img = Image.open(img_path).convert("RGB")

        # Column 0: original (just resized)
        ax = axes[row_i, 0]
        ax.imshow(_to_np(resize_tf(pil_img)))
        ax.axis("off")
        if row_i == 0:
            ax.set_title("Original", fontsize=8, fontweight="bold", pad=3)

        # Row label — annotate outside the axes since axis("off") hides ylabel
        axes[row_i, 0].annotate(
            day_label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-8, 0), textcoords="offset points",
            fontsize=11, fontweight="bold", va="center", ha="right",
        )

        # Dy28: disable translation (organoid touches boundary)
        translate = None if day_label == "Dy28" else (0.1, 0.1)
        aug_tf = _aug_transform(translate=translate)

        # Columns 1–12: augmented
        random.seed(SEED)
        np.random.seed(SEED)
        for col_i in range(1, n_cols):
            # Each augmentation gets a different seed
            seed_i = SEED * 100 + row_i * 13 + col_i
            random.seed(seed_i)
            np.random.seed(seed_i)
            import torch; torch.manual_seed(seed_i)

            aug_img = aug_tf(pil_img)
            ax = axes[row_i, col_i]
            ax.imshow(_to_np(aug_img))
            ax.axis("off")
            if row_i == 0:
                ax.set_title(f"Aug {col_i}", fontsize=7, pad=3)

    # Column separator line after "Original"
    for row_i in range(n_rows):
        axes[row_i, 0].spines["right"].set_visible(True)
        axes[row_i, 0].spines["right"].set_linewidth(1.5)
        axes[row_i, 0].spines["right"].set_color("#888888")

    # Note about Dy28 translation
    fig.text(0.5, 0.01,
             "Dy28: translation disabled (organoid may touch image boundary).  "
             "Dy06/Dy20.5: rotation ±180°, translation ±10%, H-flip, colour jitter.",
             ha="center", va="bottom", fontsize=7.5, color="#555555",
             style="italic")

    fig.suptitle(
        "EfficientNet-B0 Training Augmentation  ·  Organoid BA1 96_1 A2",
        fontsize=11, fontweight="bold", y=1.01,
    )

    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

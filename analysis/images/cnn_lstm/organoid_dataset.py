"""
PyTorch Dataset for organoid time-series CNN-LSTM training.

Reads from the runtime ``OrganoidDataset`` (no materialized split JSONs).
Background pixels are mean-filled so the model focuses on organoid texture
rather than the surrounding well.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from pipeline.data_loader import (
    LABEL_TO_INT,
    OrganoidDataset,
    filters_for_mode,
    get_clipped_meanfill_image_path,
    get_clipped_meanfill_mask_path,
    get_day_float,
    get_survey_vote_counts,
)
from pipeline.splits import Splits

LABEL_DAY = "Dy30"

# All canonical timepoints in the protocol (float days).
CANONICAL_DAYS = [3.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0, 20.5, 24.0, 28.0, 30.0]


def _has_clipped_image(dataset: "OrganoidDataset", oid: str, max_day: float = 8.0) -> bool:
    """Return True if the organoid has at least one cm_source_image_abs record on a day ≤ max_day.

    Uses max_day=8.0 (the earliest training range) so organoids that only have
    clipped images from later days are excluded — they would produce empty sequences
    for the most restrictive temporal ablation window.
    """
    for day_id, rec in dataset.organoid_records(oid).items():
        day_float = get_day_float(day_id)
        if day_float is not None and day_float <= max_day:
            if get_clipped_meanfill_image_path(rec) is not None:
                return True
    return False


def make_canonical_splits(
    all_data_path: str = "data/all_data.json",
):
    """Build train/val/test using the canonical 2026-winter split and base filter.

    Uses the same organoids as the per-day EfficientNet model (base filter +
    Splits.canonical()).  Organoids that have no cm_source_image_abs images are
    dropped.  Incomplete time series are handled by padding in
    ``OrganoidTimeSeriesDataset.__getitem__``.

    Returns ``(dataset, train_ids, val_ids, test_ids)``.
    """
    dataset = OrganoidDataset(
        all_data_path,
        filters=filters_for_mode("base"),
        splits=Splits.canonical(),
    )
    train_ids = [oid for oid in dataset.get_split("train") if _has_clipped_image(dataset, oid)]
    val_ids   = [oid for oid in dataset.get_split("val")   if _has_clipped_image(dataset, oid)]
    test_ids  = [oid for oid in dataset.get_split("test")  if _has_clipped_image(dataset, oid)]
    print(
        f"Canonical split (base filter, clipped images): {len(dataset.organoid_ids)} organoids "
        f"-> train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return dataset, train_ids, val_ids, test_ids


def compute_global_mean(dataset: OrganoidDataset, organoid_ids, image_type="clipped"):
    """Compute mean RGB across foreground pixels in the given organoids."""
    print(f"Computing global mean from {len(organoid_ids)} organoids (foreground)...")
    all_means = []
    for oid in organoid_ids:
        for rec in dataset.organoid_records(oid).values():
            img_path = (
                get_clipped_meanfill_image_path(rec)
                if image_type == "clipped"
                else (rec.get("images") or {}).get("img_path")
            )
            mask_path = (
                get_clipped_meanfill_mask_path(rec)
                if image_type == "clipped"
                else (rec.get("images") or {}).get("mask_path")
            )
            if not img_path:
                continue
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if mask_path and Path(mask_path).exists():
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                fg = img[(mask > 127).astype(bool)]
                if len(fg) > 0:
                    all_means.append(fg.mean(axis=0))
    global_mean = np.mean(all_means, axis=0)
    print(f"Global mean RGB: {global_mean}")
    return global_mean


def compute_global_mean_from_ids(dataset: OrganoidDataset, organoid_ids, image_type="clipped"):
    """Compute mean RGB across all pixels in the given organoids' images."""
    print(f"Computing global mean from {len(organoid_ids)} organoids (all pixels)...")
    all_means = []
    for oid in organoid_ids:
        for rec in dataset.organoid_records(oid).values():
            img_path = (
                get_clipped_meanfill_image_path(rec)
                if image_type == "clipped"
                else (rec.get("images") or {}).get("img_path")
            )
            if not img_path:
                continue
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            all_means.append(img.reshape(-1, 3).mean(axis=0))
    global_mean = np.mean(all_means, axis=0) / 255.0
    print(f"Global mean RGB: {global_mean}")
    return global_mean


class OrganoidTimeSeriesDataset(Dataset):
    """Loads organoid image sequences from an ``OrganoidDataset``."""

    def __init__(
        self,
        organoid_ids,
        dataset: OrganoidDataset,
        transform=None,
        use_clipping_mask=False,
        global_mean=None,
        max_day=None,
        image_type="clipped",
    ):
        """
        Args:
            organoid_ids: list of organoid_id strings (must be present in ``dataset``).
            dataset:      OrganoidDataset built with ``filters_for_mode("base")``.
            image_type:   'clipped' (575x575 mean-fill) or 'std' (512x384 standard).
        """
        self.organoid_ids = list(organoid_ids)
        self.dataset = dataset
        self.transform = transform
        self.use_clipping_mask = use_clipping_mask
        self.global_mean = global_mean
        self.max_day = max_day
        self.image_type = image_type

    def __len__(self):
        return len(self.organoid_ids)

    def _image_path(self, record):
        if self.image_type == "clipped":
            return get_clipped_meanfill_image_path(record)
        return (record.get("images") or {}).get("img_path")

    def _mask_path(self, record):
        if self.image_type == "clipped":
            return get_clipped_meanfill_mask_path(record)
        return (record.get("images") or {}).get("mask_path")

    def apply_mean_fill(self, img, mask, blur_kernel=(15, 15), dilate_iterations=5):
        """Mean-fill the background using a dilated, feathered mask."""
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
        if blur_kernel is not None:
            mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        mask = mask.astype(np.float32) / 255.0
        if self.global_mean is not None:
            mean_rgb = (self.global_mean * 255.0)[None, None, :]
        else:
            mean_rgb = img.reshape(-1, 3).mean(axis=0)[None, None, :]
        return img * mask[:, :, None] + mean_rgb * (1.0 - mask[:, :, None])

    def __getitem__(self, idx):
        organoid_id = self.organoid_ids[idx]
        records = self.dataset.organoid_records(organoid_id)

        # Sort timepoints by mdl_day (3.0, 6.0, ..., 20.5, 24.0, 28.0, 30.0)
        timepoints = sorted(
            ((get_day_float(day) or 0.0, day, rec) for day, rec in records.items()),
            key=lambda t: t[0],
        )

        images = []
        days_used = []
        for mdl_day, _day_id, rec in timepoints:
            if self.max_day is not None and mdl_day > self.max_day:
                break

            img_path = self._image_path(rec)
            if img_path is None:
                continue
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img.astype(np.float32)

            if self.use_clipping_mask:
                mask_path = self._mask_path(rec)
                if mask_path and Path(mask_path).exists():
                    mask = imread(mask_path)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    img = self.apply_mean_fill(img, mask)

            if self.transform:
                from PIL import Image
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_pil = self.transform(img_pil)
                img = np.array(img_pil).astype(np.float32) / 255.0

            images.append(img)
            days_used.append(mdl_day)

        actual_T = len(images)

        # Determine padded length from canonical day schedule.
        if self.max_day is not None:
            max_T = sum(1 for d in CANONICAL_DAYS if d <= self.max_day)
        else:
            max_T = actual_T

        # Pad shorter series with zero frames so all samples in a batch are (max_T, C, H, W).
        if actual_T > 0 and max_T > actual_T:
            pad_frame = np.zeros_like(images[0])
            for _ in range(max_T - actual_T):
                images.append(pad_frame)

        sequence = np.stack(images)           # (max_T, H, W, C)
        sequence = np.transpose(sequence, (0, 3, 1, 2))  # (max_T, C, H, W)
        seq = torch.from_numpy(sequence).float()

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        seq = (seq - imagenet_mean) / imagenet_std

        # Label per LABEL_TO_INT: 1 = Not Acceptable, 0 = Acceptable (rule #9).
        label_str = self.dataset.organoid_label(organoid_id) or ""
        label = LABEL_TO_INT.get(label_str, 0)

        # Normalize days over the actual (non-padded) range, then zero-pad.
        actual_days = np.array(days_used, dtype=np.float32)
        dmin = actual_days.min() if actual_T > 0 else 0.0
        drng = max(actual_days.max() - dmin, 1e-8) if actual_T > 0 else 1.0
        days_arr = np.zeros(max_T, dtype=np.float32)
        days_arr[:actual_T] = (actual_days - dmin) / drng
        days_norm = torch.from_numpy(days_arr).float()

        # Boolean mask: True = real frame, False = padding (for attention masking).
        pad_mask = torch.zeros(max_T, dtype=torch.bool)
        pad_mask[:actual_T] = True

        # Sample weight from Dy30 vote agreement
        dy30_rec = records.get(LABEL_DAY)
        n_good, n_total = (0, 0) if dy30_rec is None else get_survey_vote_counts(dy30_rec)
        if n_total > 0:
            frac = n_good / n_total
            if frac >= 0.9 or frac <= 0.1:
                weight = 1.0
            elif frac >= 0.7 or frac <= 0.3:
                weight = 0.8
            else:
                weight = 0.6
        else:
            weight = 1.0

        return (
            seq,
            days_norm,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            organoid_id,
            pad_mask,
        )
def resolve_split_path(splits_dir, phase):
    """
    Given a directory and a phase ('train' | 'val' | 'test'), return the path
    to the matching split JSON. Supports both layouts so the trainers can read
    from either:
        new (cohort-style):  <splits_dir>/<phase>.json
        old (data_splits/):  <splits_dir>/<phase>_idor_series.json
    Cohort layout takes precedence when both exist.
    """
    d = Path(splits_dir)
    cohort_style = d / f"{phase}.json"
    legacy_style = d / f"{phase}_idor_series.json"
    if cohort_style.exists():
        return str(cohort_style)
    if legacy_style.exists():
        return str(legacy_style)
    raise FileNotFoundError(
        f"No split file for phase '{phase}' in {d}. Looked for "
        f"{cohort_style.name} and {legacy_style.name}."
    )


def load_split_from_json(split_path):
    """
    Load a pre-made split JSON produced by scripts/split_series_reproducible.py.

    Returns:
        organoid_ids  - list of organoid_id strings
        series_data   - dict {organoid_id: {label, n_votes_good, n_votes_total,
                                            base_well, genealogy_type, n_timepoints,
                                            timepoints: [{key, mdl_day, img_path,
                                                          mask_path, ...}]}}

    Usage:
        train_ids, train_data = load_split_from_json('data_splits/series_train.json')
        val_ids,   val_data   = load_split_from_json('data_splits/series_val.json')
        test_ids,  test_data  = load_split_from_json('data_splits/series_test.json')

        train_dataset = OrganoidTimeSeriesDataset(train_ids, train_data, ...)
    """
    with open(split_path) as f:
        series_data = json.load(f)

    organoid_ids = list(series_data.keys())

    labels = [series_data[oid]['label'] for oid in organoid_ids]
    acc    = labels.count('Acceptable')
    na     = labels.count('Not Acceptable')
    print(f"Loaded {len(organoid_ids)} organoids from {split_path} "
          f"({acc} Acceptable, {na} Not Acceptable)")

    return organoid_ids, series_data

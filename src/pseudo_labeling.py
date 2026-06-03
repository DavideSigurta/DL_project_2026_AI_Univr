"""
Pseudo-labeling utilities for E4.

Functions:
  - build_unlabeled_loader: DataLoader for unlabeled pool (returns image_id)
  - generate_pseudo_labels: predict + confidence filter → DataFrame
  - save_pseudo_label_csv: persist DataFrame + print stats
  - make_merged_csv: combine labeled + pseudo-labeled CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .augmentations import get_val_transforms
from .datasets import ISICDataset
from .utils import get_device


def build_unlabeled_loader(
    csv_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
) -> DataLoader:
    """Build DataLoader for unlabeled CSV.

    Uses val transforms (no random augmentation) and ``return_id=True``
    so ``generate_pseudo_labels`` can track which image received which label.
    """
    transforms = get_val_transforms(img_size)
    dataset = ISICDataset(csv_path, transforms=transforms, return_id=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.no_grad()
def generate_pseudo_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.9,
    source: str = "isic2020",
) -> pd.DataFrame:
    """Generate pseudo-labels from model predictions.

    Only keeps samples where confidence (max probability) >= threshold.

    Returns
    -------
    pd.DataFrame with columns ``image_id, filepath, target, source``.
    Empty DataFrame if no sample passes threshold.
    """
    model.eval()
    records: list[dict] = []

    for images, _targets, image_ids in loader:
        images = images.to(device)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

        for prob, img_id in zip(probs, image_ids):
            pred_label = 1 if prob >= 0.5 else 0
            confidence = prob if pred_label == 1 else 1.0 - prob
            if confidence >= threshold:
                records.append({
                    "image_id": img_id,
                    "target": pred_label,
                    "confidence": float(confidence),
                })

    if not records:
        return pd.DataFrame(columns=["image_id", "filepath", "target", "source"])

    df = pd.DataFrame(records)

    # Attach filepath from original CSV via loader dataset
    orig_csv: str = loader.dataset.csv_path  # type: ignore[attr-defined]
    orig_df = pd.read_csv(orig_csv)
    df = df.merge(orig_df[["image_id", "filepath"]], on="image_id", how="left")
    df["source"] = source

    return df[["image_id", "filepath", "target", "source"]]


def save_pseudo_label_csv(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save pseudo-label DataFrame to CSV and print statistics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(df)
    n_malignant = df["target"].sum() if n_total > 0 else 0
    n_benign = n_total - n_malignant

    df.to_csv(output_path, index=False)

    print(f"  Pseudo-labels saved: {output_path}")
    print(f"    Total  : {n_total}")
    print(f"    Malig  : {n_malignant} ({100*n_malignant/max(n_total,1):.1f}%)")
    print(f"    Benign : {n_benign} ({100*n_benign/max(n_total,1):.1f}%)")


def make_merged_csv(
    labeled_csv: str | Path,
    pseudo_csv: str | Path,
    output_path: str | Path,
) -> str:
    """Concatenate labeled CSV and pseudo-label CSV into a single training set.

    Both CSVs must have columns ``image_id, filepath, target, source``.
    """
    labeled = pd.read_csv(labeled_csv)
    pseudo = pd.read_csv(pseudo_csv)

    merged = pd.concat([labeled, pseudo], ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    n_labeled = len(labeled)
    n_pseudo = len(pseudo)
    n_total = len(merged)
    print(f"  Merged dataset: {output_path}")
    print(f"    Labeled      : {n_labeled}")
    print(f"    Pseudo-label : {n_pseudo}")
    print(f"    Total unique : {n_total}")
    return str(output_path)

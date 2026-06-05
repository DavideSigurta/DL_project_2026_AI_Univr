from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentations import (
    get_simclr_transforms,
    get_student_transforms,
    get_teacher_transforms,
    get_train_transforms,
    get_val_transforms,
)


def _read_image(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _apply_transforms(img: Image.Image, transforms):
    if transforms is None:
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    out = transforms(img)
    if isinstance(out, dict) and "image" in out:
        return out["image"]
    return out


class ISICDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transforms=None,
        return_id: bool = False,
        label_col: str = "target",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transforms = transforms
        self.return_id = return_id
        self.label_col = label_col

        required = {"image_id", "filepath"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_image(row["filepath"])
        img = _apply_transforms(img, self.transforms)

        if self.label_col not in row:
            raise ValueError(f"Label column '{self.label_col}' not found in CSV.")
        target = torch.tensor(float(row[self.label_col]), dtype=torch.float32)

        if self.return_id:
            return img, target, row["image_id"]
        return img, target


class UnlabeledDataset(Dataset):
    def __init__(self, csv_path: str | Path, transforms=None) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transforms = transforms

        required = {"image_id", "filepath"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_image(row["filepath"])
        return _apply_transforms(img, self.transforms)


class SimCLRDataset(Dataset):
    def __init__(self, csv_path: str | Path, simclr_transforms=None) -> None:
        self.base = UnlabeledDataset(csv_path, transforms=None)
        self.simclr_transforms = simclr_transforms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        row = self.base.df.iloc[idx]
        img = _read_image(row["filepath"])
        view1 = _apply_transforms(img, self.simclr_transforms)
        view2 = _apply_transforms(img, self.simclr_transforms)
        return view1, view2


class SemiSupervisedDataset(Dataset):
    def __init__(
        self,
        labeled_csv: str | Path,
        unlabeled_csv: str | Path,
        labeled_transforms=None,
        unlabeled_transforms: Optional[Tuple] = None,
        unlabeled_ratio: float = 3.0,
        label_col: str = "target",
    ) -> None:
        if unlabeled_transforms is None or len(unlabeled_transforms) != 2:
            raise ValueError(
                "SemiSupervisedDataset requires unlabeled_transforms=(student_tf, teacher_tf). "
                "Both transforms must be provided to ensure proper image normalization."
            )
        self.labeled = ISICDataset(
            labeled_csv,
            transforms=labeled_transforms,
            return_id=False,
            label_col=label_col,
        )
        self.unlabeled = UnlabeledDataset(unlabeled_csv, transforms=None)
        self.unlabeled_k = max(1, int(round(unlabeled_ratio)))
        self.unlabeled_transforms = unlabeled_transforms

    def __len__(self) -> int:
        return len(self.labeled)

    def __getitem__(self, idx: int):
        img_l, target = self.labeled[idx]
        student_tf, teacher_tf = self.unlabeled_transforms

        unlabeled_student = []
        unlabeled_teacher = []
        for _ in range(self.unlabeled_k):
            u_idx = np.random.randint(0, len(self.unlabeled))
            row = self.unlabeled.df.iloc[u_idx]
            img_u = _read_image(row["filepath"])
            img_u_s = _apply_transforms(img_u, student_tf)
            img_u_t = _apply_transforms(img_u, teacher_tf)
            unlabeled_student.append(img_u_s)
            unlabeled_teacher.append(img_u_t)

        return {
            "labeled": (img_l, target),
            "unlabeled": (torch.stack(unlabeled_student), torch.stack(unlabeled_teacher)),
        }


class DomainDataset(Dataset):
    def __init__(
        self,
        source_csv: str | Path,
        target_csv: str | Path,
        transforms=None,
        label_col: str = "target",
        include_target_labels: bool = False,
        max_target_samples: Optional[int] = None,
    ) -> None:
        src = pd.read_csv(source_csv)
        tgt = pd.read_csv(target_csv)

        # Subsample target to prevent OOM and excessive batch/epoch at low fractions
        if max_target_samples is not None and len(tgt) > max_target_samples:
            tgt = tgt.sample(n=max_target_samples, random_state=42)

        src["domain_label"] = 0
        tgt["domain_label"] = 1

        if not include_target_labels:
            tgt[label_col] = -1

        self.df = pd.concat([src, tgt], ignore_index=True)
        self.transforms = transforms
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_image(row["filepath"])
        img = _apply_transforms(img, self.transforms)
        class_label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        domain_label = torch.tensor(int(row["domain_label"]), dtype=torch.long)
        return img, class_label, domain_label


def _make_weighted_sampler(df: pd.DataFrame, label_col: str = "target") -> WeightedRandomSampler:
    counts = df[label_col].value_counts().to_dict()
    weights = df[label_col].map(lambda x: 1.0 / counts[x]).astype(np.float32).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_loaders(config: Dict) -> Dict[str, DataLoader]:
    data_cfg = config.get("data", {})
    task = data_cfg.get("task", "supervised")
    img_size = data_cfg.get("img_size", 224)
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)
    label_col = data_cfg.get("label_col", "target")

    if task == "supervised":
        train_tf = get_train_transforms(img_size)
        val_tf = get_val_transforms(img_size)

        train_csv = data_cfg["train_csv"]
        val_csv = data_cfg.get("val_csv")
        test_csv = data_cfg.get("test_csv")

        train_ds = ISICDataset(train_csv, transforms=train_tf, label_col=label_col)
        sampler = None
        if data_cfg.get("use_weighted_sampler", False):
            sampler = _make_weighted_sampler(train_ds.df, label_col=label_col)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
        )

        loaders = {"train": train_loader}
        if val_csv:
            val_ds = ISICDataset(val_csv, transforms=val_tf, label_col=label_col)
            loaders["val"] = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        if test_csv:
            test_ds = ISICDataset(test_csv, transforms=val_tf, label_col=label_col)
            loaders["test"] = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        return loaders

    if task == "simclr":
        train_csv = data_cfg["train_csv"]
        simclr_tf = get_simclr_transforms(img_size)
        train_ds = SimCLRDataset(train_csv, simclr_transforms=simclr_tf)
        return {
            "train": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )
        }

    if task == "mean_teacher":
        labeled_csv = data_cfg["labeled_csv"]
        unlabeled_csv = data_cfg["unlabeled_csv"]
        labeled_tf = get_student_transforms(img_size)
        student_tf = get_student_transforms(img_size)
        teacher_tf = get_teacher_transforms(img_size)
        train_ds = SemiSupervisedDataset(
            labeled_csv,
            unlabeled_csv,
            labeled_transforms=labeled_tf,
            unlabeled_transforms=(student_tf, teacher_tf),
            unlabeled_ratio=data_cfg.get("unlabeled_ratio", 3.0),
            label_col=label_col,
        )
        sampler = None
        if data_cfg.get("use_weighted_sampler", False):
            sampler = _make_weighted_sampler(train_ds.labeled.df, label_col=label_col)
        loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=True,
            )
        }
        val_csv = data_cfg.get("val_csv")
        if val_csv:
            val_ds = ISICDataset(val_csv, transforms=get_val_transforms(img_size), label_col=label_col)
            loaders["val"] = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        return loaders

    if task == "dann":
        source_csv = data_cfg["source_csv"]
        target_csv = data_cfg["target_csv"]
        train_ds = DomainDataset(
            source_csv,
            target_csv,
            transforms=get_train_transforms(img_size),
            label_col=label_col,
            include_target_labels=data_cfg.get("include_target_labels", False),
            max_target_samples=data_cfg.get("max_target_samples", None),
        )

        # Balanced domain sampler: each batch gets ~50% source + ~50% target
        # Weight = 1/count_per_domain so both domains have equal sampling probability
        balanced = data_cfg.get("balanced_domains", False)
        sampler = None
        shuffle = True
        if balanced:
            df = train_ds.df
            n_src = (df["domain_label"] == 0).sum()
            n_tgt = (df["domain_label"] == 1).sum()
            weights = np.where(df["domain_label"] == 0, 1.0 / n_src, 1.0 / n_tgt)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights.astype(np.float64)),
                num_samples=len(df),
                replacement=True,
            )
            shuffle = False  # sampler handles shuffling

        loaders = {
            "train": DataLoader(
                train_ds, batch_size=batch_size, shuffle=shuffle,
                sampler=sampler, num_workers=num_workers
            )
        }
        val_csv = data_cfg.get("val_csv")
        if val_csv:
            val_ds = ISICDataset(val_csv, transforms=get_val_transforms(img_size), label_col=label_col)
            loaders["val"] = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        return loaders

    raise ValueError(f"Unknown task '{task}' in config.data.task")

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import random

import numpy as np
import torch
import torch.nn as nn

from ..datasets import build_loaders
from ..models import ProjectionHead, SimCLRModel, build_backbone
from ..losses import NTXentLoss
from .supervised import run_experiment
from ..utils import (
    append_jsonl,
    get_device,
    init_run_dir,
    save_checkpoint,
    save_config,
    set_seed,
    setup_logging,
)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _build_ssl_loader(config: Dict) -> torch.utils.data.DataLoader:
    """Build SimCLR dataloader with seeded workers for reproducibility."""
    generator = torch.Generator()
    generator.manual_seed(config["experiment"].get("seed", 42))
    loaders = build_loaders(config)
    loader = loaders["train"]
    # Rebuild with worker seeding (build_loaders may not set these)
    loader = torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return loader


# ---------------------------------------------------------------------------
# SSL pretraining
# ---------------------------------------------------------------------------

def run_ssl_pretraining(config: Dict) -> str:
    """
    Run SimCLR pretraining. Returns path to best encoder checkpoint.
    Saves: encoder_best.pt, encoder_last.pt, encoder_epoch_N.pt every save_every epochs.
    """
    exp_cfg = config["experiment"]
    ssl_cfg = config["ssl"]
    model_cfg = config["model"]
    log_cfg = config.get("logging", {})

    set_seed(exp_cfg.get("seed", 42))
    device = get_device(exp_cfg.get("device"))
    run_dir = init_run_dir(exp_cfg.get("name", "simclr_pretrain"), base_dir=log_cfg.get("save_dir", "results/runs"))
    logger = setup_logging(run_dir, name="simclr")
    save_config(config, run_dir / "config.yaml")

    loader = _build_ssl_loader(config)

    encoder = build_backbone(
        arch=model_cfg["arch"],
        pretrained=model_cfg.get("pretrained", True),
        num_classes=0,
        features_only=True,
    ).to(device)

    proj_head = ProjectionHead(
        in_dim=encoder.feature_dim,
        hidden_dim=model_cfg.get("projection_hidden_dim", 512),
        out_dim=model_cfg.get("projection_dim", 128),
    ).to(device)

    model = SimCLRModel(encoder, proj_head).to(device)

    criterion = NTXentLoss(temperature=ssl_cfg.get("temperature", 0.07)).to(device)

    opt_name = ssl_cfg.get("optimizer", "adamw").lower()
    lr = ssl_cfg.get("lr", 3e-4)
    wd = ssl_cfg.get("weight_decay", 1e-4)
    if opt_name == "lars":
        try:
            from torch_optimizer import LARS
            optimizer = LARS(model.parameters(), lr=lr, weight_decay=wd)
        except ImportError:
            logger.warning("torch_optimizer not found, falling back to AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = ssl_cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    save_every = ssl_cfg.get("save_every", 10)
    save_best = ssl_cfg.get("save_best", True)

    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(run_dir) / "metrics.jsonl"

    best_loss = float("inf")
    best_encoder_path = str(ckpt_dir / "encoder_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for batch in loader:
            # build_loaders with task='simclr' returns (view1, view2) or ((view1, view2), label)
            if isinstance(batch, (list, tuple)) and isinstance(batch[0], (list, tuple)):
                (v1, v2), _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor) and batch[0].ndim == 4:
                v1, v2 = batch
            else:
                (v1, v2), _ = batch

            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(v1)
            z2 = model(v2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * v1.size(0)
            n += v1.size(0)

        avg_loss = total_loss / max(n, 1)
        scheduler.step()

        log_entry = {"epoch": epoch, "ssl_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
        append_jsonl(metrics_path, log_entry)
        logger.info(log_entry)

        # Save periodic checkpoint (encoder only)
        if epoch % save_every == 0:
            save_checkpoint(
                {"encoder": model.encoder.state_dict(), "epoch": epoch, "loss": avg_loss},
                ckpt_dir / f"encoder_epoch_{epoch}.pt",
            )

        # Save best
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {"encoder": model.encoder.state_dict(), "epoch": epoch, "loss": avg_loss},
                best_encoder_path,
            )

    # Always save last
    save_checkpoint(
        {"encoder": model.encoder.state_dict(), "epoch": epochs, "loss": avg_loss},
        ckpt_dir / "encoder_last.pt",
    )

    logger.info(f"Pretraining done. Best loss={best_loss:.4f}. Encoder saved to {best_encoder_path}")
    return best_encoder_path


# ---------------------------------------------------------------------------
# SSL fine-tuning
# ---------------------------------------------------------------------------

def run_ssl_finetune(
    base_config: Dict,
    ssl_ckpt_path: str,
    fraction: Optional[float] = None,
) -> Dict[str, float]:
    """
    Fine-tune an SSL-pretrained encoder on a labeled subset.
    Delegates to supervised.run_experiment with ssl_checkpoint injected.
    fraction: if provided, overrides data.train_csv to the corresponding subset CSV.
    """
    config = copy.deepcopy(base_config)
    config.setdefault("data", {})
    config.setdefault("experiment", {})

    ft_cfg = config.get("finetune", {})
    ft_data_cfg = config.get("finetune_data", {})
    if ft_cfg:
        config["training"] = copy.deepcopy(ft_cfg)
    else:
        config.setdefault("training", {})

    # Point to label-fraction CSV if requested
    if fraction is not None:
        if fraction >= 1.0:
            config["data"]["train_csv"] = "data/processed/isic2018/train.csv"
        else:
            pct = int(round(fraction * 100))
            config["data"]["train_csv"] = f"data/processed/isic2018/subsets/train_{pct:02d}pct.csv"
        config["data"]["val_csv"] = "data/processed/isic2018/val.csv"
        config["data"]["test_csv"] = "data/processed/isic2018/test.csv"
        config["data"]["use_weighted_sampler"] = True
        # Adjust freeze epochs based on fraction
        if fraction <= 0.05:
            config["training"]["freeze_backbone_epochs"] = ft_cfg.get("freeze_backbone_epochs_low", 5)
        else:
            config["training"]["freeze_backbone_epochs"] = ft_cfg.get("freeze_backbone_epochs_high", 0)
        config["experiment"]["name"] = f"simclr_ft_{fraction:.2f}"

    config["ssl_checkpoint"] = ssl_ckpt_path
    config["data"]["task"] = "supervised"
    if "batch_size" in ft_data_cfg:
        config["data"]["batch_size"] = ft_data_cfg["batch_size"]
    if "num_workers" in ft_data_cfg:
        config["data"]["num_workers"] = ft_data_cfg["num_workers"]

    return run_experiment(config)
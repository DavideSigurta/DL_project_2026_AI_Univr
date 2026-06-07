from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

import torch
from torch import nn

from ..datasets import build_loaders
from ..losses import build_loss
from ..models import build_backbone
from ..utils import (
    append_jsonl,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
    evaluate as evaluate_shared,
    get_device,
    get_monitor_value,
    init_run_dir,
    save_checkpoint,
    save_config,
    set_backbone_trainable,
    set_seed,
    setup_logging,
    split_params,
)


def train_one_epoch(model, loader, optimizer, criterion, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return {"loss": total_loss / max(n, 1)}


def run_experiment(config: Dict) -> Dict[str, float]:
    exp_cfg = config.get("experiment", {})
    set_seed(exp_cfg.get("seed", 42))

    device = get_device(exp_cfg.get("device"))
    run_dir = init_run_dir(exp_cfg.get("name", "supervised"))
    logger = setup_logging(run_dir, name="supervised")
    save_config(config, run_dir / "config.yaml")

    loaders = build_loaders(config)
    model = build_backbone(
        arch=config["model"]["arch"],
        pretrained=config["model"].get("pretrained", True),
        num_classes=config["model"].get("num_classes", 1),
    ).to(device)

    # Load SSL encoder weights if provided
    ssl_ckpt_path = config.get("ssl_checkpoint")
    if ssl_ckpt_path:
        ckpt = torch.load(ssl_ckpt_path, map_location=device)
        encoder_state = ckpt.get("encoder", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        logger.info(f"Loaded SSL checkpoint from {ssl_ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")

    backbone_params, head_params = split_params(model)
    optim_cfg = config.get("training", {})
    lr_backbone = optim_cfg.get("lr_backbone", optim_cfg.get("lr", 1e-4))
    lr_head = optim_cfg.get("lr_head", optim_cfg.get("lr", 1e-4))

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

    optimizer = build_optimizer(param_groups, config)
    scheduler = build_scheduler(optimizer, config)

    criterion = build_loss(config).to(device)
    early, monitor_name, monitor_mode = build_early_stopping(config)

    epochs = optim_cfg.get("epochs", 1)
    freeze_epochs = optim_cfg.get("freeze_backbone_epochs", 0)

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_val_auc = -float("inf")
    metrics_path = Path(run_dir) / "metrics.jsonl"

    for epoch in range(1, epochs + 1):
        set_backbone_trainable(model, trainable=epoch > freeze_epochs)
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)

        log_entry = {"epoch": epoch, "train_loss": train_metrics["loss"]}
        if "val" in loaders:
            val_metrics = evaluate_shared(model, loaders["val"], criterion, device)
            log_entry.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_auc": val_metrics["auc_roc"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]

            monitor_value = get_monitor_value(val_metrics, monitor_name)
            log_entry["val_monitor"] = monitor_value

            improved = monitor_value < best_metric if monitor_mode == "min" else monitor_value > best_metric
            if improved:
                best_metric = monitor_value
                save_checkpoint({"model": model.state_dict(), "epoch": epoch}, run_dir / "checkpoints/best.pt")

            if early.step(monitor_value):
                logger.info("Early stopping triggered.")
                append_jsonl(metrics_path, log_entry)
                break

        append_jsonl(metrics_path, log_entry)
        logger.info(log_entry)

        save_checkpoint({"model": model.state_dict(), "epoch": epoch}, run_dir / "checkpoints/last.pt")
        if scheduler is not None:
            scheduler.step()

    return {"run_dir": str(run_dir), "best_val_auc": best_val_auc, "best_monitor_value": best_metric}

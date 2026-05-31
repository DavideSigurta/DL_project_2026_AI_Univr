from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

import torch
from torch import nn

from ..datasets import build_loaders
from ..losses import FocalLoss, WeightedBCELoss
from ..metrics import compute_metrics
from ..models import build_backbone
from ..utils import (
    EarlyStopping,
    append_jsonl,
    get_device,
    init_run_dir,
    save_checkpoint,
    save_config,
    set_seed,
    setup_logging,
)


def _split_params(model: nn.Module):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ["classifier", "fc", "head"]):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ["classifier", "fc", "head"]):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def _build_loss(cfg: Dict) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "bce")
    if loss_type == "focal":
        return FocalLoss(gamma=loss_cfg.get("gamma", 2.0), alpha=loss_cfg.get("alpha", None))
    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=loss_cfg.get("pos_weight", 1.0))
    return nn.BCEWithLogitsLoss()


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


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_targets = []
    all_probs = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images).view(-1)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    y_true = torch.from_numpy(np.concatenate(all_targets)).flatten().numpy()
    y_pred = torch.from_numpy(np.concatenate(all_probs)).flatten().numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


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

    backbone_params, head_params = _split_params(model)
    optimizer_cfg = config.get("training", {})
    optimizer_name = optimizer_cfg.get("optimizer", "adamw").lower()
    lr_backbone = optimizer_cfg.get("lr_backbone", optimizer_cfg.get("lr", 1e-4))
    lr_head = optimizer_cfg.get("lr_head", optimizer_cfg.get("lr", 1e-4))
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    scheduler = None
    if optimizer_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimizer_cfg.get("epochs", 1)
        )

    criterion = _build_loss(config).to(device)
    early_cfg = config.get("early_stopping", {})
    monitor_name = early_cfg.get("monitor", "val_auc")
    monitor_mode = early_cfg.get("mode")
    if monitor_mode is None:
        monitor_mode = "min" if str(monitor_name).lower().endswith("loss") else "max"
    early = EarlyStopping(
        patience=early_cfg.get("patience", 10),
        mode=monitor_mode,
        min_delta=early_cfg.get("min_delta", 0.0),
    )

    epochs = optimizer_cfg.get("epochs", 1)
    freeze_epochs = optimizer_cfg.get("freeze_backbone_epochs", 0)

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_val_auc = -float("inf")
    metrics_path = Path(run_dir) / "metrics.jsonl"

    for epoch in range(1, epochs + 1):
        _set_backbone_trainable(model, trainable=epoch > freeze_epochs)
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)

        log_entry = {"epoch": epoch, "train_loss": train_metrics["loss"]}
        if "val" in loaders:
            val_metrics = evaluate(model, loaders["val"], criterion, device)
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

            monitor_map = {
                "val_auc": val_metrics["auc_roc"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
            }
            monitor_value = monitor_map.get(monitor_name, val_metrics["auc_roc"])
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

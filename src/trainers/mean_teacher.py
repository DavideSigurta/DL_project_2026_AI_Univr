from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import build_loaders
from ..losses import build_loss
from ..models import build_backbone
from ..utils import (
    EarlyStopping,
    append_jsonl,
    evaluate as evaluate_shared,
    get_device,
    init_run_dir,
    save_checkpoint,
    save_config,
    set_backbone_trainable,
    set_seed,
    setup_logging,
    split_params,
)


def update_ema(student: nn.Module, teacher: nn.Module, alpha: float = 0.999) -> None:
    """EMA update: teacher = alpha * teacher + (1 - alpha) * student."""
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """MSE on sigmoid probabilities between student and teacher."""
    student_probs = torch.sigmoid(student_logits).view(-1)
    teacher_probs = torch.sigmoid(teacher_logits).view(-1)
    return F.mse_loss(student_probs, teacher_probs.detach())


def rampup_weight(epoch: int, rampup_epochs: int, max_weight: float = 10.0) -> float:
    """Sigmoid ramp-up: 0 -> max_weight over rampup_epochs."""
    if rampup_epochs <= 0:
        return max_weight
    if epoch >= rampup_epochs:
        return max_weight
    phase = epoch / max(rampup_epochs, 1)
    return float(max_weight * torch.sigmoid(torch.tensor(12.0 * (phase - 0.5))).item())


def train_one_epoch_mt(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    rampup_epochs: int,
    max_cons_weight: float,
    ema_decay: float,
) -> Dict[str, float]:
    """Single Mean Teacher epoch."""
    student.train()
    teacher.train()

    total_sup_loss = 0.0
    total_cons_loss = 0.0
    total_loss = 0.0
    n_labeled = 0
    n_unlabeled = 0

    cons_w = rampup_weight(epoch, rampup_epochs, max_cons_weight)

    for batch in loader:
        (img_l, targets) = batch["labeled"]
        (student_views, teacher_views) = batch["unlabeled"]

        # student_views/teacher_views shape after collation: [B, k, C, H, W]
        # Reshape to [B*k, C, H, W] for single-image forward pass
        student_views = student_views.view(-1, *student_views.shape[2:])
        teacher_views = teacher_views.view(-1, *teacher_views.shape[2:])

        img_l = img_l.to(device)
        targets = targets.to(device)
        student_views = student_views.to(device)
        teacher_views = teacher_views.to(device)

        # Labeled branch: supervised loss
        logits_l = student(img_l).view(-1)
        sup_loss = criterion(logits_l, targets)

        # Unlabeled branch: consistency loss
        stu_logits_u = student(student_views).view(-1)
        with torch.no_grad():
            tea_logits_u = teacher(teacher_views).view(-1)
        cons_loss_val = consistency_loss(stu_logits_u, tea_logits_u)

        loss = sup_loss + cons_w * cons_loss_val

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # EMA update
        update_ema(student, teacher, alpha=ema_decay)

        total_sup_loss += sup_loss.item() * img_l.size(0)
        total_cons_loss += cons_loss_val.item() * student_views.size(0)
        total_loss += loss.item() * img_l.size(0)
        n_labeled += img_l.size(0)
        n_unlabeled += student_views.size(0)

    return {
        "sup_loss": total_sup_loss / max(n_labeled, 1),
        "cons_loss": total_cons_loss / max(n_unlabeled, 1),
        "loss": total_loss / max(n_labeled, 1),
        "cons_weight": cons_w,
    }


def run_mean_teacher(config: Dict) -> Dict[str, float]:
    """Run Mean Teacher training for semi-supervised learning.

    Expects config with:
        data.task = "mean_teacher"
        semisup.ema_decay, semisup.consistency_weight, semisup.rampup_epochs
        training.freeze_backbone_epochs
    """
    exp_cfg = config.get("experiment", {})
    set_seed(exp_cfg.get("seed", 42))

    device = get_device(exp_cfg.get("device"))
    run_dir = init_run_dir(exp_cfg.get("name", "mean_teacher"))
    logger = setup_logging(run_dir, name="mean_teacher")
    save_config(config, run_dir / "config.yaml")

    loaders = build_loaders(config)

    # Build student + teacher (same architecture, teacher = EMA of student)
    student = build_backbone(
        arch=config["model"]["arch"],
        pretrained=config["model"].get("pretrained", True),
        num_classes=config["model"].get("num_classes", 1),
    ).to(device)

    teacher = build_backbone(
        arch=config["model"]["arch"],
        pretrained=config["model"].get("pretrained", True),
        num_classes=config["model"].get("num_classes", 1),
    ).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizer: differential LR backbone / head
    backbone_params, head_params = split_params(student)
    optim_cfg = config.get("training", {})
    lr_backbone = optim_cfg.get("lr_backbone", optim_cfg.get("lr", 1e-4))
    lr_head = optim_cfg.get("lr_head", optim_cfg.get("lr", 1e-4))
    weight_decay = optim_cfg.get("weight_decay", 0.0)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    optimizer_name = optim_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    scheduler = None
    if optim_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optim_cfg.get("epochs", 1)
        )

    criterion = build_loss(config).to(device)

    # Semi-supervised hyperparams
    semi_cfg = config.get("semisup", {})
    ema_decay = semi_cfg.get("ema_decay", 0.999)
    max_cons_weight = semi_cfg.get("consistency_weight", 10.0)
    rampup_epochs = semi_cfg.get("rampup_epochs", 10)

    # Early stopping
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

    epochs = optim_cfg.get("epochs", 50)
    freeze_epochs = optim_cfg.get("freeze_backbone_epochs", 0)

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_val_auc = -float("inf")
    metrics_path = Path(run_dir) / "metrics.jsonl"

    for epoch in range(1, epochs + 1):
        set_backbone_trainable(student, trainable=epoch > freeze_epochs)

        train_metrics = train_one_epoch_mt(
            student, teacher, loaders["train"], optimizer, criterion,
            device, epoch, rampup_epochs, max_cons_weight, ema_decay,
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_sup_loss": train_metrics["sup_loss"],
            "train_cons_loss": train_metrics["cons_loss"],
            "cons_weight": train_metrics["cons_weight"],
        }

        if "val" in loaders:
            val_metrics = evaluate_shared(student, loaders["val"], criterion, device)
            log_entry.update({
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc_roc"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            })
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
                save_checkpoint({"model": student.state_dict(), "epoch": epoch}, run_dir / "checkpoints/best.pt")

            if early.step(monitor_value):
                logger.info("Early stopping triggered.")
                append_jsonl(metrics_path, log_entry)
                break

        append_jsonl(metrics_path, log_entry)
        logger.info(log_entry)

        save_checkpoint({"model": student.state_dict(), "epoch": epoch}, run_dir / "checkpoints/last.pt")
        if scheduler is not None:
            scheduler.step()

    return {"run_dir": str(run_dir), "best_val_auc": best_val_auc, "best_monitor_value": best_metric}

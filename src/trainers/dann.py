from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import build_loaders
from ..losses import CORLoss, build_loss
from ..models import build_backbone, build_dann_model
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
    set_seed,
    setup_logging,
)


def _ganin_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    """Ganin progressive schedule: 0 -> lambda_max over total_epochs.

    λ(p) = λ_max * (2 / (1 + exp(-10 * p)) - 1),  p = epoch / total_epochs
    """
    p = epoch / max(total_epochs, 1)
    return float(lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


def _linear_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    """Linear schedule: 0 -> lambda_max over total_epochs."""
    return float(lambda_max * min(epoch / max(total_epochs, 1), 1.0))


def _build_schedule(config: Dict) -> callable:
    da_cfg = config.get("da", {})
    schedule_type = da_cfg.get("lambda_schedule", "dann")
    lambda_max = da_cfg.get("lambda_max", 1.0)

    if schedule_type == "linear":
        return lambda e, t: _linear_lambda(e, t, lambda_max)
    return lambda e, t: _ganin_lambda(e, t, lambda_max)


def _make_eval_state_dict(
    backbone: nn.Module,
    task_head: nn.Module,
    classifier_key: str = "fc",
) -> Dict[str, torch.Tensor]:
    """Combine backbone + task_head into single state dict compatible with build_backbone(num_classes=1).

    build_backbone returns timm model with classifier named {classifier_key}.
    build_dann_model returns separate backbone (num_classes=0, no head) + task_head.
    This function merges them for evaluate_on_csv compatibility.

    **Assumption:** ``classifier_key`` defaults to ``"fc"`` (ResNet convention).
    If you swap the backbone architecture (e.g. EfficientNet → ``"classifier.1"``,
    ViT → ``"head"``), update this argument accordingly. No validation is performed
    on the resulting state dict keys, so a mismatch will silently fail in
    ``load_state_dict``.
    """
    state = {}
    for k, v in backbone.state_dict().items():
        state[k] = v
    for k, v in task_head.state_dict().items():
        state[f"{classifier_key}.{k}"] = v
    return state


def train_one_epoch_dann(
    backbone: nn.Module,
    task_head: nn.Module,
    domain_classifier: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    device: torch.device,
    lambda_val: float,
    use_coral: bool = False,
    coral_criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Single DANN epoch."""
    backbone.train()
    task_head.train()
    domain_classifier.train()

    total_cls_loss = 0.0
    total_dom_loss = 0.0
    total_loss = 0.0
    n_source = 0
    n_total = 0

    for batch in loader:
        images, class_labels, domain_labels = batch
        images = images.to(device)
        class_labels = class_labels.to(device)
        domain_labels = domain_labels.to(device)

        # Forward
        features = backbone(images)
        domain_logits = domain_classifier(features)

        # ── Domain loss (all samples) ────────────────────────────────────
        dom_loss = F.cross_entropy(domain_logits, domain_labels)

        # ── Task loss (source only: domain_label == 0) ───────────────────
        source_mask = domain_labels == 0
        n_source_batch = source_mask.sum().item()

        cls_loss = torch.tensor(0.0, device=device)
        if n_source_batch > 0:
            src_features = features[source_mask]
            src_class_labels = class_labels[source_mask]
            class_logits = task_head(src_features).view(-1)
            cls_loss = cls_criterion(class_logits, src_class_labels)

        # ── CORAL (optional) ─────────────────────────────────────────────
        coral_loss = torch.tensor(0.0, device=device)
        if use_coral and coral_criterion is not None and n_source_batch > 0:
            target_mask = domain_labels == 1
            n_target_batch = target_mask.sum().item()
            if n_target_batch > 0:
                tgt_features = features[target_mask]
                coral_loss = coral_criterion(src_features, tgt_features)

        # Combined loss
        loss = cls_loss + lambda_val * dom_loss + coral_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item() * max(n_source_batch, 1)
        total_dom_loss += dom_loss.item() * images.size(0)
        total_loss += loss.item() * images.size(0)
        n_source += n_source_batch
        n_total += images.size(0)

    return {
        "cls_loss": total_cls_loss / max(n_source, 1),
        "dom_loss": total_dom_loss / max(n_total, 1),
        "loss": total_loss / max(n_total, 1),
        "lambda": lambda_val,
    }


def run_dann(config: Dict) -> Dict[str, float]:
    """Run DANN training for domain adaptation.

    Expects config with:
        data.task = "dann"
        da.lambda_schedule, da.lambda_max, da.use_coral
        training.lr_backbone, lr_head, lr_domain
    """
    exp_cfg = config.get("experiment", {})
    set_seed(exp_cfg.get("seed", 42))

    device = get_device(exp_cfg.get("device"))
    run_dir = init_run_dir(exp_cfg.get("name", "dann"))
    logger = setup_logging(run_dir, name="dann")
    save_config(config, run_dir / "config.yaml")

    loaders = build_loaders(config)

    # Build DANN model: backbone + task_head + domain_classifier
    dann_models = build_dann_model(
        backbone_arch=config["model"]["arch"],
        num_classes=config["model"].get("num_classes", 1),
        pretrained=config["model"].get("pretrained", True),
    )
    backbone = dann_models["backbone"].to(device)
    task_head = dann_models["task_head"].to(device)
    domain_classifier = dann_models["domain_classifier"].to(device)

    # Optimizer: 3 param groups
    optim_cfg = config.get("training", {})
    lr_backbone = optim_cfg.get("lr_backbone", optim_cfg.get("lr", 1e-4))
    lr_head = optim_cfg.get("lr_head", optim_cfg.get("lr", 1e-4))
    lr_domain = optim_cfg.get("lr_domain", 1e-3)
    weight_decay = optim_cfg.get("weight_decay", 0.0)

    param_groups = [
        {"params": backbone.parameters(), "lr": lr_backbone},
        {"params": task_head.parameters(), "lr": lr_head},
        {"params": domain_classifier.parameters(), "lr": lr_domain},
    ]
    optimizer = build_optimizer(param_groups, config)
    scheduler = build_scheduler(optimizer, config)

    # Losses
    cls_criterion = build_loss(config).to(device)

    # Domain loss is built-in CE in train_one_epoch_dann

    # CORAL (optional)
    da_cfg = config.get("da", {})
    use_coral = da_cfg.get("use_coral", False)
    coral_criterion = CORLoss().to(device) if use_coral else None

    # Lambda schedule
    lambda_fn = _build_schedule(config)
    total_epochs = optim_cfg.get("epochs", 50)

    # Early stopping
    early, monitor_name, monitor_mode = build_early_stopping(config)

    epochs = optim_cfg.get("epochs", 50)
    freeze_epochs = optim_cfg.get("freeze_backbone_epochs", 0)

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_val_auc = -float("inf")
    metrics_path = Path(run_dir) / "metrics.jsonl"

    # Build eval model (backbone + task_head combined) for validation
    # Use timm model with classifier to match evaluate() expectations
    eval_model = build_backbone(
        arch=config["model"]["arch"],
        pretrained=False,
        num_classes=config["model"].get("num_classes", 1),
    ).to(device)

    for epoch in range(1, epochs + 1):
        # Freeze backbone control
        if epoch <= freeze_epochs:
            for p in backbone.parameters():
                p.requires_grad = False
        else:
            for p in backbone.parameters():
                p.requires_grad = True

        lambda_val = lambda_fn(epoch, total_epochs)

        train_metrics = train_one_epoch_dann(
            backbone, task_head, domain_classifier, loaders["train"],
            optimizer, cls_criterion, device, lambda_val,
            use_coral=use_coral, coral_criterion=coral_criterion,
        )

        # Sync eval_model weights (backbone + task_head)
        eval_state = _make_eval_state_dict(backbone, task_head)
        eval_model.load_state_dict(eval_state)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_dom_loss": train_metrics["dom_loss"],
            "lambda": train_metrics["lambda"],
        }

        if "val" in loaders:
            val_metrics = evaluate_shared(eval_model, loaders["val"], cls_criterion, device)
            log_entry.update({
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc_roc"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            })
            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]

            monitor_value = get_monitor_value(val_metrics, monitor_name)
            log_entry["val_monitor"] = monitor_value

            improved = monitor_value < best_metric if monitor_mode == "min" else monitor_value > best_metric
            if improved:
                best_metric = monitor_value
                # Save combined state dict for evaluate_on_csv compatibility
                save_checkpoint({
                    "model": _make_eval_state_dict(backbone, task_head),
                    "backbone": backbone.state_dict(),
                    "task_head": task_head.state_dict(),
                    "domain_classifier": domain_classifier.state_dict(),
                    "epoch": epoch,
                }, run_dir / "checkpoints/best.pt")

            if early.step(monitor_value):
                logger.info("Early stopping triggered.")
                append_jsonl(metrics_path, log_entry)
                break

        append_jsonl(metrics_path, log_entry)
        logger.info(log_entry)

        # Save last checkpoint (eval format)
        save_checkpoint({
            "model": _make_eval_state_dict(backbone, task_head),
            "backbone": backbone.state_dict(),
            "task_head": task_head.state_dict(),
            "domain_classifier": domain_classifier.state_dict(),
            "epoch": epoch,
        }, run_dir / "checkpoints/last.pt")

        if scheduler is not None:
            scheduler.step()

    return {"run_dir": str(run_dir), "best_val_auc": best_val_auc, "best_monitor_value": best_metric}

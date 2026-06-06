from __future__ import annotations

from typing import Optional

import logging
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # squeeze last dim if 2D (e.g. [N, 1] -> [N]), but NOT if 1D (preserve [N] and [1])
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1)
        targets = targets.float()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=logits.device))
        return criterion(logits, targets)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)

        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)

        positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(sim.device)
        labels = positives

        loss = F.cross_entropy(sim, labels)
        return loss


def build_loss(config: dict) -> nn.Module:
    """Build loss function from config section.

    Supports: focal, weighted_bce, bce (default).
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "bce")
    if loss_type == "focal":
        return FocalLoss(gamma=loss_cfg.get("gamma", 2.0), alpha=loss_cfg.get("alpha", None))
    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=loss_cfg.get("pos_weight", 1.0))
    logger.warning("Unknown loss.type '%s', falling back to BCEWithLogitsLoss", loss_type)
    return nn.BCEWithLogitsLoss()


class CORLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        if source_feat.size(0) < 2 or target_feat.size(0) < 2:
            return torch.tensor(0.0, device=source_feat.device)
        d = source_feat.size(1)
        src = source_feat - source_feat.mean(dim=0, keepdim=True)
        tgt = target_feat - target_feat.mean(dim=0, keepdim=True)
        cov_src = (src.T @ src) / (source_feat.size(0) - 1)
        cov_tgt = (tgt.T @ tgt) / (target_feat.size(0) - 1)
        loss = ((cov_src - cov_tgt) ** 2).sum() / (4.0 * d * d)
        return loss

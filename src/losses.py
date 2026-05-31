from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logits = logits.squeeze()
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
        logits = logits.squeeze()
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


class ConsistencyLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(teacher_logits)
        return F.mse_loss(p_s, p_t, reduction=self.reduction)


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q: float = 0.7, num_classes: int = 2) -> None:
        super().__init__()
        if not (0.0 < q <= 1.0):
            raise ValueError("q must be in (0, 1].")
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1 or logits.size(-1) == 1:
            probs_pos = torch.sigmoid(logits.view(-1, 1))
            probs = torch.cat([1.0 - probs_pos, probs_pos], dim=1)
            targets = targets.long()
        else:
            probs = F.softmax(logits, dim=1)
            targets = targets.long()

        one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        p_t = (probs * one_hot).sum(dim=1)
        loss = (1.0 - p_t.pow(self.q)) / self.q
        return loss.mean()


class CORLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        d = source_feat.size(1)
        src = source_feat - source_feat.mean(dim=0, keepdim=True)
        tgt = target_feat - target_feat.mean(dim=0, keepdim=True)
        cov_src = (src.T @ src) / (source_feat.size(0) - 1)
        cov_tgt = (tgt.T @ tgt) / (target_feat.size(0) - 1)
        loss = ((cov_src - cov_tgt) ** 2).sum() / (4.0 * d * d)
        return loss

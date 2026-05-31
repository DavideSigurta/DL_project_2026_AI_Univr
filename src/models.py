from __future__ import annotations

from typing import Dict

import timm
import torch
from torch import nn


def build_backbone(arch: str, pretrained: bool, num_classes: int, features_only: bool = False) -> nn.Module:
    if features_only:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
    else:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    if hasattr(model, "num_features"):
        model.feature_dim = model.num_features  # type: ignore[attr-defined]
    return model


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """Encoder + projection head as named submodules.
    Only encoder.state_dict() is saved in SSL checkpoints; proj_head is discarded after pretraining.
    """

    def __init__(self, encoder: nn.Module, proj_head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.proj_head = proj_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_head(self.encoder(x))


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFn.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, alpha: float = 1.0) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grl(x)
        return self.net(x)


def build_dann_model(
    backbone_arch: str,
    num_classes: int,
    pretrained: bool = True,
) -> Dict[str, nn.Module]:
    backbone = timm.create_model(backbone_arch, pretrained=pretrained, num_classes=0)
    feat_dim = backbone.num_features if hasattr(backbone, "num_features") else None
    if feat_dim is None:
        raise ValueError("Could not infer feature dimension from backbone.")

    task_head = nn.Linear(feat_dim, num_classes)
    domain_classifier = DomainClassifier(feat_dim)
    return {"backbone": backbone, "task_head": task_head, "domain_classifier": domain_classifier}
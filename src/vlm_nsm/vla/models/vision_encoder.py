"""Vision encoder for VLA — ResNet-based multi-view feature extractor.

Supports:
  - ResNet18/34 backbones with ImageNet pretrained weights (default).
  - Shared weights for multi-view inputs (agentview + frontview).
  - Configurable output dimension.
  - Optional from-scratch training (freeze=False, pretrained=False).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class MultiViewEncoder(nn.Module):
    """Encodes multiple camera views into a single feature vector.

    Uses shared ResNet weights for all views, then concatenates features.

    Args:
        backbone: ResNet variant ("resnet18" or "resnet34").
        pretrained: Use ImageNet pretrained weights.
        freeze_bn: Freeze batch norm layers during training.
        output_dim: Feature dimension per view.
        n_views: Number of camera views.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_bn: bool = True,
        output_dim: int = 512,
        n_views: int = 2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.n_views = n_views
        self.output_dim = output_dim * n_views

        # Build backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            self.encoder = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            self.encoder = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace final fc with output_dim
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(in_features, output_dim)

        # Freeze batch norm
        if freeze_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multiple views.

        Args:
            views: (B, n_views, 3, H, W) or (B, 3, H, W) for single view.

        Returns:
            (B, output_dim * n_views) feature vector.
        """
        if views.dim() == 4:
            # Single view
            return self.encoder(views)

        # Multi-view: process each view through shared encoder
        B, V, C, H, W = views.shape
        features = []
        for v in range(V):
            feat = self.encoder(views[:, v, :, :, :])
            features.append(feat)
        return torch.cat(features, dim=-1)


# ── M0: VLA BC Model (ResNet18 + MLP) ──────────────────────────────────────

class VLABCModel(nn.Module):
    """M0 baseline: ResNet encoder + MLP action predictor.

    This is the simplest VLA model — purely for pipeline validation.
    Does NOT use diffusion. Predicts a single action via MSE regression.

    Inputs:
      - agentview + frontview RGB (128x128)
      - proprioception (7-dim: eef_xyz + gripper + zeros)
      - symbolic condition (sym_dim, e.g. 31-42)

    Output:
      - action (4-dim: dx, dy, dz, gripper)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_bn: bool = True,
        vision_output_dim: int = 256,
        proprio_dim: int = 7,
        symbolic_dim: int = 31,
        hidden_dims: tuple[int, ...] = (512, 256),
        action_dim: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_encoder = MultiViewEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_bn=freeze_bn,
            output_dim=vision_output_dim,
            n_views=2,
        )
        vision_feat_dim = vision_output_dim * 2  # 2 views

        # Proprio MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
        )

        # Symbolic condition MLP
        self.symbolic_mlp = nn.Sequential(
            nn.Linear(symbolic_dim, 128),
            nn.ReLU(inplace=True),
        )

        # Fusion + head
        fusion_dim = vision_feat_dim + 64 + 128
        layers = []
        in_dim = fusion_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.head = nn.Sequential(*layers)

        self._action_dim = action_dim

    def forward(
        self,
        agentview: torch.Tensor,
        frontview: torch.Tensor,
        proprio: torch.Tensor,
        symbolic_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict action from multimodal inputs.

        Args:
            agentview: (B, 3, H, W)
            frontview: (B, 3, H, W)
            proprio: (B, 7)
            symbolic_cond: (B, sym_dim)

        Returns:
            action: (B, 4)
        """
        # Stack views for shared encoder
        views = torch.stack([agentview, frontview], dim=1)  # (B, 2, 3, H, W)
        vis_feat = self.vision_encoder(views)                # (B, vision_dim*2)

        prop_feat = self.proprio_mlp(proprio)                # (B, 64)
        sym_feat = self.symbolic_mlp(symbolic_cond)          # (B, 128)

        fused = torch.cat([vis_feat, prop_feat, sym_feat], dim=-1)
        return self.head(fused)

    @property
    def action_dim(self) -> int:
        return self._action_dim

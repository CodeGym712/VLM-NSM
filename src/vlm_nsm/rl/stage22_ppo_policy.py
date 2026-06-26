"""Stage 22-A3: PPO policy with Stage 20-Z warm start.

Actor: LSTM encoder from Stage 20-Z, Gaussian policy head.
Critic: Separate LSTM value network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOPolicy(nn.Module):
    """Actor-Critic policy for PPO with LSTM encoder.

    Actor: encodes feature history → Gaussian action distribution
    Critic: encodes feature history → scalar value
    """

    def __init__(
        self,
        feature_dim: int = 110,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 2,
        history_len: int = 4,
        init_log_std: float = -1.5,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.action_scale = action_scale

        # Shared encoder
        self.encoder = nn.LSTM(
            feature_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.1,
        )

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # Critic head
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, feature_dim) or (H, feature_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, H, D)
        _, (h, _) = self.encoder(x)
        return h[-1]  # (B, hidden_dim)

    def get_action_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic action mean."""
        latent = self.encode(x)
        return self.actor_mean(latent) * self.action_scale

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """State value."""
        latent = self.encode(x)
        return self.critic(latent).squeeze(-1)

    def forward(self, x: torch.Tensor, deterministic: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action, log_prob, value.

        Returns:
            action: (B, action_dim) or (action_dim,)
            log_prob: (B,) or scalar
            value: (B,) or scalar
        """
        latent = self.encode(x)
        mean = self.actor_mean(latent) * self.action_scale
        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(latent).squeeze(-1)

        # Clamp actions
        action = torch.clamp(action, -self.action_scale, self.action_scale)
        return action, log_prob, value

    def evaluate_action(
        self, x: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for a given action."""
        latent = self.encode(x)
        mean = self.actor_mean(latent) * self.action_scale
        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return log_prob, entropy, value


def load_warm_start(checkpoint_path: str, policy: PPOPolicy, device: torch.device) -> bool:
    """Load Stage 20-Z actor weights into PPO policy.

    Maps Stage 20-Z LSTM + action head to PPO encoder + actor_mean.
    Stage 20-Z checkpoint keys:
        rnn.* → encoder.* (LSTM)
        head.0.* → actor_mean.0.* (LayerNorm)
        head.1.* → actor_mean.1.* (first Linear, 7→256)

    Returns True if warm start succeeded, False if partial.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
    except Exception:
        return False

    # Manual mapping for known checkpoint structure
    mapping = {}
    for k in state_dict:
        if k.startswith("rnn."):
            mapping[k] = "encoder." + k[4:]  # rnn.* → encoder.*
        elif k == "head.0.weight":
            mapping[k] = "actor_mean.0.weight"  # LayerNorm weight
        elif k == "head.0.bias":
            mapping[k] = "actor_mean.0.bias"  # LayerNorm bias
        elif k == "head.1.weight":
            mapping[k] = "actor_mean.1.weight"  # first Linear weight (7, 256)
        elif k == "head.1.bias":
            mapping[k] = "actor_mean.1.bias"  # first Linear bias (7,)
        # gripper_magnitude_head.* and holding_head.* are ignored (aux heads)

    loaded = 0
    skipped = 0
    policy_params = dict(policy.named_parameters())

    for ckpt_key, policy_key in mapping.items():
        if policy_key in policy_params and ckpt_key in state_dict:
            ckpt_val = state_dict[ckpt_key]
            policy_val = policy_params[policy_key]
            if policy_val.shape == ckpt_val.shape:
                policy_val.data.copy_(ckpt_val.data)
                loaded += 1
            else:
                print(f"  Shape mismatch: {ckpt_key}({list(ckpt_val.shape)}) → {policy_key}({list(policy_val.shape)})")
                skipped += 1
        else:
            skipped += 1

    print(f"Warm start: loaded {loaded}/{len(mapping)} params, skipped {skipped}")
    return loaded > 0

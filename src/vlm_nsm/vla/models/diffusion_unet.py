"""Conditional 1D UNet for Diffusion Policy action denoising.

A lightweight 1D convolutional UNet that takes:
  - noise_action_chunk: (B, horizon, action_dim)  — noisy action sequence
  - condition: (B, cond_dim)                       — observation embedding
  - timestep: (B,)                                  — diffusion timestep

And outputs:
  - denoised_action_chunk: (B, horizon, action_dim)

Design follows the Diffusion Policy paper (Chi et al., 2023):
  - FiLM conditioning for observation embedding
  - Sinusoidal timestep embedding
  - 1D convolutions over action horizon (temporal dimension)
  - Skip connections (U-Net style)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (as in DDPM/Transformers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block for 1D convolutions."""

    def __init__(self, in_dim: int, cond_dim: int, out_dim: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(min(8, out_dim), out_dim)
        # FiLM: condition → scale + shift
        self.film = nn.Linear(cond_dim, out_dim * 2)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        x = self.conv(x)
        scale_shift = self.film(cond)[:, :, None]  # (B, 2*C, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return self.act(x)


class ConditionalUnet1D(nn.Module):
    """1D Conditional UNet for action chunk denoising.

    Args:
        action_dim: Dimension of each action (e.g., 4 for dx,dy,dz,gripper).
        horizon: Action chunk length (e.g., 16).
        cond_dim: Observation condition dimension.
        hidden_dims: Channel dimensions for each UNet level.
        time_dim: Timestep embedding dimension.
    """

    def __init__(
        self,
        action_dim: int = 4,
        horizon: int = 16,
        cond_dim: int = 1216,
        hidden_dims: tuple[int, ...] = (256, 512, 1024),
        time_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.Mish(),
        )

        # Combined conditioning (obs + time)
        combined_dim = time_dim + time_dim

        # Encoder
        in_dim = action_dim
        self.encoders = nn.ModuleList()
        for h_dim in hidden_dims:
            self.encoders.append(FiLMBlock(in_dim, combined_dim, h_dim))
            in_dim = h_dim

        # Bottleneck
        self.bottleneck = FiLMBlock(hidden_dims[-1], combined_dim, hidden_dims[-1])

        # Decoder (with skip connections)
        self.decoders = nn.ModuleList()
        rev_dims = list(reversed(hidden_dims[:-1]))
        for i, h_dim in enumerate(rev_dims):
            skip_dim = hidden_dims[-(i + 2)]
            self.decoders.append(
                FiLMBlock(hidden_dims[-(i + 1)] + skip_dim, combined_dim, h_dim)
            )

        # Final projection
        self.final_conv = nn.Sequential(
            nn.Conv1d(rev_dims[-1] if rev_dims else hidden_dims[0], action_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise action chunk.

        Args:
            x: (B, horizon, action_dim) — noisy action chunk.
            cond: (B, cond_dim) — observation embedding.
            timestep: (B,) — diffusion timestep [0, T-1].

        Returns:
            (B, horizon, action_dim) — predicted noise (or denoised action).
        """
        # Transpose: (B, H, A) → (B, A, H) for 1D conv over time
        x = x.transpose(1, 2)

        # Encode timestep and condition
        t_emb = self.time_mlp(timestep)          # (B, time_dim)
        c_emb = self.cond_proj(cond)              # (B, time_dim)
        combined = torch.cat([t_emb, c_emb], dim=-1)  # (B, combined_dim)

        # Encoder with skip connections
        skips = []
        for enc in self.encoders:
            x = enc(x, combined)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x, combined)

        # Decoder
        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            x = torch.cat([x, skip], dim=1)
            x = dec(x, combined)

        # Final projection
        x = self.final_conv(x)

        # Transpose back: (B, A, H) → (B, H, A)
        return x.transpose(1, 2)

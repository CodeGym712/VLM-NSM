"""Diffusion Policy for VLA action generation.

Wraps ConditionalUnet1D with DDPM noise schedule and DDIM sampling.
Combines vision encoder, proprio MLP, and symbolic MLP with the diffusion head.
"""

from __future__ import annotations

import torch
from torch import nn

from vlm_nsm.vla.models.vision_encoder import MultiViewEncoder
from vlm_nsm.vla.models.diffusion_unet import ConditionalUnet1D


class DDPMScheduler:
    """DDPM noise scheduler with cosine schedule."""

    def __init__(self, num_train_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = "cosine"):
        self.num_train_steps = num_train_steps

        if schedule == "cosine":
            betas = self._cosine_beta_schedule(num_train_steps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_train_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _cosine_beta_schedule(self, steps: int, s: float = 0.008) -> torch.Tensor:
        t = torch.linspace(0, steps, steps + 1)
        ft = torch.cos((t / steps + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = ft / ft[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, max=0.999)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_cum)*x0 + sqrt(1-alpha_cum)*noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x0.device)[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t]
        # Reshape for broadcasting: (B,) → (B, 1, 1)
        sqrt_alpha = sqrt_alpha[:, None, None]
        sqrt_one_minus = sqrt_one_minus[:, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


class DDIMScheduler:
    """DDIM sampler for fast inference."""

    def __init__(self, num_train_steps: int = 100, num_inference_steps: int = 16,
                 eta: float = 0.0):
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # DDIM step schedule: subsampled timesteps from T-1 down to 0
        # NOTE: arange(0, 100, step) may produce step+1 values (17 for ratio=6).
        # This is intentional — the extra step at t=0 ensures full denoising.
        step_ratio = num_train_steps // num_inference_steps
        self.timesteps = torch.arange(0, num_train_steps, step_ratio).flip(0)

    def step(self, model_output, t_idx, x_t, scheduler: DDPMScheduler):
        """Single DDIM denoising step."""
        t = self.timesteps[t_idx]
        t_next = self.timesteps[t_idx + 1] if t_idx + 1 < len(self.timesteps) else -1

        alpha_cum = scheduler.alphas_cumprod[t]
        alpha_cum_next = (scheduler.alphas_cumprod[t_next]
                          if t_next >= 0 else torch.tensor(1.0))

        # Predicted x0
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cum)
        pred_x0 = (x_t - sqrt_one_minus_alpha * model_output) / torch.sqrt(alpha_cum)

        # Direction to x_t
        pred_dir = torch.sqrt(1.0 - alpha_cum_next) * model_output

        # x_{t-1}
        x_prev = torch.sqrt(alpha_cum_next) * pred_x0 + pred_dir

        # Add noise for stochastic sampling (eta > 0)
        if self.eta > 0 and t_idx > 0:
            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_cum_next) / (1.0 - alpha_cum) *
                (1.0 - alpha_cum / alpha_cum_next)
            )
            x_prev = x_prev + sigma * torch.randn_like(x_prev)

        return x_prev


class DiffusionPolicy(nn.Module):
    """VLA Diffusion Policy.

    Combines:
      - MultiViewEncoder (ResNet, shared weights)
      - Proprio MLP
      - Symbolic condition MLP
      - Conditional 1D UNet (diffusion head)
      - DDPM/DDIM schedulers
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_bn: bool = True,
        vision_output_dim: int = 256,
        proprio_dim: int = 7,
        symbolic_dim: int = 42,
        action_dim: int = 4,
        horizon: int = 16,
        cond_dim: int = 1216,
        unet_dims: tuple[int, ...] = (256, 512, 1024),
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps

        # Encoder
        self.vision_encoder = MultiViewEncoder(
            backbone=backbone, pretrained=pretrained, freeze_bn=freeze_bn,
            output_dim=vision_output_dim, n_views=2,
        )
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64), nn.ReLU(inplace=True))
        self.symbolic_mlp = nn.Sequential(
            nn.Linear(symbolic_dim, 128), nn.ReLU(inplace=True))

        # Compute actual cond_dim from encoder architecture
        actual_cond_dim = (vision_output_dim * 2) + 64 + 128  # vis*2 + proprio + symbolic
        self.cond_dim = actual_cond_dim

        # Diffusion UNet
        self.unet = ConditionalUnet1D(
            action_dim=action_dim, horizon=horizon,
            cond_dim=actual_cond_dim, hidden_dims=unet_dims,
        )

        # Schedulers
        self.noise_scheduler = DDPMScheduler(num_train_steps=num_diffusion_steps)
        self.inference_scheduler = DDIMScheduler(
            num_train_steps=num_diffusion_steps,
            num_inference_steps=num_inference_steps,
        )

    def encode_obs(
        self,
        agentview: torch.Tensor,
        frontview: torch.Tensor,
        proprio: torch.Tensor,
        symbolic_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Encode multimodal observation into a single condition vector."""
        views = torch.stack([agentview, frontview], dim=1)
        vis_feat = self.vision_encoder(views)
        prop_feat = self.proprio_mlp(proprio)
        sym_feat = self.symbolic_mlp(symbolic_cond)
        return torch.cat([vis_feat, prop_feat, sym_feat], dim=-1)

    def forward(
        self,
        agentview: torch.Tensor,
        frontview: torch.Tensor,
        proprio: torch.Tensor,
        symbolic_cond: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward: predict noise added to action_chunk.

        Args:
            action_chunk: (B, horizon, 4) — clean action chunk.
            Returns: (B, horizon, 4) — predicted noise, and loss-ready.
        """
        B = action_chunk.shape[0]
        cond = self.encode_obs(agentview, frontview, proprio, symbolic_cond)

        # Sample noise and timesteps
        noise = torch.randn_like(action_chunk)
        timesteps = torch.randint(0, self.num_diffusion_steps, (B,),
                                  device=action_chunk.device)

        # Add noise
        noisy = self.noise_scheduler.add_noise(action_chunk, noise, timesteps)

        # Predict noise
        noise_pred = self.unet(noisy, cond, timesteps)

        # Loss is MSE between predicted and actual noise
        return noise_pred, noise

    def sample(
        self,
        agentview: torch.Tensor,
        frontview: torch.Tensor,
        proprio: torch.Tensor,
        symbolic_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: generate action chunk via DDIM sampling.

        Returns: (B, horizon, action_dim) — denoised action chunk.
        """
        B = agentview.shape[0]
        device = agentview.device
        cond = self.encode_obs(agentview, frontview, proprio, symbolic_cond)

        # Start from random noise
        x_t = torch.randn(B, self.horizon, self.action_dim, device=device)

        # DDIM denoising loop
        for t_idx in range(len(self.inference_scheduler.timesteps)):
            t = self.inference_scheduler.timesteps[t_idx]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(x_t, cond, t_batch)
            x_t = self.inference_scheduler.step(noise_pred, t_idx, x_t,
                                                self.noise_scheduler)

        return x_t

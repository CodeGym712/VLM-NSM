"""Tests for Diffusion Policy and 1D UNet."""

import numpy as np
import pytest
import torch

from vlm_nsm.vla.models.diffusion_unet import ConditionalUnet1D
from vlm_nsm.vla.models.diffusion_policy import DiffusionPolicy, DDPMScheduler, DDIMScheduler


class TestDDPMScheduler:
    def test_add_noise_shape(self):
        sched = DDPMScheduler(num_train_steps=100)
        x0 = torch.randn(4, 16, 4)
        noise = torch.randn(4, 16, 4)
        t = torch.randint(0, 100, (4,))
        xt = sched.add_noise(x0, noise, t)
        assert xt.shape == (4, 16, 4)

    def test_noise_level(self):
        """At t=0, x_t ≈ x0. At t=99, x_t is noisy."""
        sched = DDPMScheduler(num_train_steps=100)
        B = 32
        x0 = torch.ones(B, 4, 4)
        noise = torch.randn(B, 4, 4)
        t0 = torch.zeros(B, dtype=torch.long)
        t99 = torch.full((B,), 99, dtype=torch.long)
        xt0 = sched.add_noise(x0, noise, t0)
        xt99 = sched.add_noise(x0, noise, t99)
        # At t=0, cosine schedule adds small noise (sqrt_alpha≈0.9997)
        # MSE between xt0 and x0 should be very small
        mse_t0 = torch.nn.functional.mse_loss(xt0, x0)
        assert mse_t0 < 0.01, f"MSE at t=0 too high: {mse_t0:.4f}"
        # At t=99, should be noisy → variance >> 0
        assert torch.var(xt99, correction=0) > 0.1


class TestDDIMScheduler:
    def test_inference_shape(self):
        train_sched = DDPMScheduler(num_train_steps=100)
        infer_sched = DDIMScheduler(num_train_steps=100, num_inference_steps=16)
        x_t = torch.randn(2, 8, 4)
        model_out = torch.randn(2, 8, 4)
        x_prev = infer_sched.step(model_out, 0, x_t, train_sched)
        assert x_prev.shape == (2, 8, 4)
        assert torch.all(torch.isfinite(x_prev))


class TestConditionalUnet1D:
    def test_forward_shape(self):
        unet = ConditionalUnet1D(action_dim=4, horizon=16, cond_dim=256)
        x = torch.randn(2, 16, 4)
        cond = torch.randn(2, 256)
        t = torch.randint(0, 100, (2,))
        out = unet(x, cond, t)
        assert out.shape == (2, 16, 4)
        assert torch.all(torch.isfinite(out))

    def test_overfit_small(self):
        """1D UNet should overfit a small batch."""
        unet = ConditionalUnet1D(action_dim=4, horizon=8, cond_dim=64,
                                 hidden_dims=(128, 256))
        x = torch.randn(8, 8, 4)
        cond = torch.randn(8, 64)
        target = torch.randn(8, 8, 4)

        opt = torch.optim.Adam(unet.parameters(), lr=1e-3)
        for _ in range(100):
            t = torch.randint(0, 100, (8,))
            pred = unet(x, cond, t)
            loss = torch.nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < 0.5  # should decrease significantly


class TestDiffusionPolicy:
    @pytest.fixture
    def policy(self):
        return DiffusionPolicy(
            backbone="resnet18", pretrained=False,
            vision_output_dim=64, symbolic_dim=31,
            action_dim=4, horizon=8,
            unet_dims=(128, 256),
            num_diffusion_steps=20,
            num_inference_steps=4,
        )

    def test_training_forward(self, policy):
        B = 2
        av = torch.randn(B, 3, 128, 128)
        fv = torch.randn(B, 3, 128, 128)
        prop = torch.randn(B, 7)
        sym = torch.randn(B, 31)
        action = torch.randn(B, 8, 4)

        noise_pred, noise = policy(av, fv, prop, sym, action)
        assert noise_pred.shape == (B, 8, 4)
        assert noise.shape == (B, 8, 4)
        assert torch.all(torch.isfinite(noise_pred))

    def test_inference(self, policy):
        B = 2
        av = torch.randn(B, 3, 128, 128)
        fv = torch.randn(B, 3, 128, 128)
        prop = torch.randn(B, 7)
        sym = torch.randn(B, 31)

        with torch.no_grad():
            action = policy.sample(av, fv, prop, sym)
        assert action.shape == (B, 8, 4)
        assert torch.all(torch.isfinite(action))

    def test_train_one_step(self, policy):
        """Training loop should reduce loss."""
        av = torch.randn(4, 3, 128, 128)
        fv = torch.randn(4, 3, 128, 128)
        prop = torch.randn(4, 7)
        sym = torch.randn(4, 31)
        action = torch.randn(4, 8, 4)

        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        losses = []
        for _ in range(10):
            noise_pred, noise = policy(av, fv, prop, sym, action)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        assert losses[-1] < losses[0] * 0.9  # loss should decrease

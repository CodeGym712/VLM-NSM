"""Tests for VLA vision encoder and M0 model."""

import numpy as np
import pytest
import torch

from vlm_nsm.vla.models.vision_encoder import MultiViewEncoder, VLABCModel


class TestMultiViewEncoder:
    """Tests for MultiViewEncoder."""

    @pytest.fixture
    def dummy_views(self):
        """(B, 2, 3, 128, 128) tensor."""
        return torch.randn(4, 2, 3, 128, 128)

    @pytest.fixture
    def dummy_single(self):
        """(B, 3, 128, 128) tensor."""
        return torch.randn(4, 3, 128, 128)

    def test_resnet18_pretrained(self, dummy_views):
        encoder = MultiViewEncoder(backbone="resnet18", pretrained=True,
                                   output_dim=256, n_views=2)
        out = encoder(dummy_views)
        assert out.shape == (4, 512)  # 256 * 2 views

    def test_resnet18_from_scratch(self, dummy_views):
        encoder = MultiViewEncoder(backbone="resnet18", pretrained=False,
                                   output_dim=256, n_views=2)
        out = encoder(dummy_views)
        assert out.shape == (4, 512)

    def test_single_view(self, dummy_single):
        encoder = MultiViewEncoder(backbone="resnet18", pretrained=False,
                                   output_dim=256, n_views=1)
        out = encoder(dummy_single)
        assert out.shape == (4, 256)

    def test_output_dim_configurable(self, dummy_views):
        encoder = MultiViewEncoder(backbone="resnet18", pretrained=False,
                                   output_dim=128, n_views=2)
        out = encoder(dummy_views)
        assert out.shape == (4, 256)  # 128 * 2

    def test_freeze_bn(self, dummy_views):
        encoder = MultiViewEncoder(backbone="resnet18", pretrained=True,
                                   freeze_bn=True)
        encoder.eval()
        with torch.no_grad():
            out1 = encoder(dummy_views)
            out2 = encoder(dummy_views)
        assert torch.allclose(out1, out2)  # deterministic


class TestVLABCModel:
    """Tests for M0 VLA BC model."""

    @pytest.fixture
    def model(self):
        return VLABCModel(
            backbone="resnet18",
            pretrained=False,
            vision_output_dim=128,
            symbolic_dim=31,
            hidden_dims=(256, 128),
            action_dim=4,
        )

    def test_forward_shape(self, model):
        B = 4
        av = torch.randn(B, 3, 128, 128)
        fv = torch.randn(B, 3, 128, 128)
        prop = torch.randn(B, 7)
        sym = torch.randn(B, 31)

        out = model(av, fv, prop, sym)
        assert out.shape == (B, 4)

    def test_forward_values_finite(self, model):
        av = torch.randn(2, 3, 128, 128)
        fv = torch.randn(2, 3, 128, 128)
        prop = torch.randn(2, 7)
        sym = torch.randn(2, 31)

        out = model(av, fv, prop, sym)
        assert torch.all(torch.isfinite(out))

    def test_training_mode(self, model):
        model.train()
        av = torch.randn(2, 3, 128, 128)
        fv = torch.randn(2, 3, 128, 128)
        prop = torch.randn(2, 7)
        sym = torch.randn(2, 31)
        target = torch.randn(2, 4)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        pred = model(av, fv, prop, sym)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_overfit_small_batch(self, model):
        """Model should be able to overfit a tiny batch."""
        B = 4
        av = torch.randn(B, 3, 128, 128)
        fv = torch.randn(B, 3, 128, 128)
        prop = torch.randn(B, 7)
        sym = torch.randn(B, 31)
        target = torch.randn(B, 4)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        for epoch in range(50):
            model.train()
            pred = model(av, fv, prop, sym)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = float(loss)

        final_loss = float(loss)
        # Should drop significantly
        assert final_loss < initial_loss * 0.5, \
            f"Overfit failed: {initial_loss:.4f} → {final_loss:.4f}"

    def test_action_dim_property(self, model):
        assert model.action_dim == 4

    def test_different_symbolic_dim(self):
        model = VLABCModel(
            backbone="resnet18", pretrained=False,
            vision_output_dim=128, symbolic_dim=42,
            action_dim=4,
        )
        av = torch.randn(2, 3, 128, 128)
        fv = torch.randn(2, 3, 128, 128)
        prop = torch.randn(2, 7)
        sym = torch.randn(2, 42)
        out = model(av, fv, prop, sym)
        assert out.shape == (2, 4)

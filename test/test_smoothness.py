"""Test smoothness loss computation."""

import pytest
import torch

from utils.losses import lut_smoothness_loss
from utils.transforms import identity_lut


class TestLUTSmoothnessLoss:
    """Tests for LUT smoothness loss computation."""

    def test_identity_lut_low_smoothness(self):
        """Identity LUT should have low smoothness loss (smooth transitions)."""
        lut = identity_lut(resolution=16)
        loss = lut_smoothness_loss(lut)

        # Identity LUT has smooth gradients, so loss should be small
        assert loss.item() < 0.01, "Identity LUT should have low smoothness loss"

    def test_random_lut_high_smoothness(self):
        """Random LUT should have higher smoothness loss (rough transitions)."""
        random_lut = torch.rand(16, 16, 16, 3)
        identity = identity_lut(resolution=16)

        loss_random = lut_smoothness_loss(random_lut)
        loss_identity = lut_smoothness_loss(identity)

        # Random LUT should have rougher gradients than identity
        assert loss_random > loss_identity

    def test_constant_lut_zero_smoothness(self):
        """Constant LUT should have near-zero smoothness loss (no variation)."""
        constant_lut = torch.ones(16, 16, 16, 3) * 0.5
        loss = lut_smoothness_loss(constant_lut)

        # Constant LUT has no gradients, so loss should be ~0
        assert loss.item() < 1e-6, "Constant LUT should have near-zero smoothness loss"

    def test_gradient_flow(self):
        """Test that gradients flow through smoothness loss."""
        lut = identity_lut(resolution=16)
        lut.requires_grad = True

        loss = lut_smoothness_loss(lut)
        loss.backward()

        assert lut.grad is not None, "Gradients should exist"
        assert lut.grad.abs().mean() > 0, "Gradients should be non-zero"

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_different_lut_sizes(self, size):
        """Test smoothness loss works with different LUT sizes."""
        lut = identity_lut(resolution=size)
        loss = lut_smoothness_loss(lut)

        # Loss should be computable and small for identity LUTs
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
        assert loss.item() < 0.01  # Identity should be smooth

    def test_grayscale_lut_smoothness(self):
        """Test smoothness loss works with grayscale (single-channel) LUTs."""
        grayscale_lut = identity_lut(resolution=16, grayscale=True)
        loss = lut_smoothness_loss(grayscale_lut)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

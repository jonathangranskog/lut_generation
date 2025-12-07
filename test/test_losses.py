"""
Tests for loss functions in utils/losses.py
"""

import pytest
import torch
from utils.losses import (
    image_smoothness_loss,
    image_regularization_loss,
    black_level_preservation_loss,
    lut_smoothness_loss,
    compute_losses,
)
from utils.constants import REC709_LUMA_R, REC709_LUMA_G, REC709_LUMA_B


class TestImageSmoothnessLoss:
    """Test image smoothness loss function."""

    def test_smooth_image_low_loss(self):
        """Smooth images should have low smoothness loss."""
        # Create a smooth gradient image
        batch_size, channels, height, width = 2, 3, 64, 64
        x = torch.linspace(0, 1, width).view(1, 1, 1, width)
        y = torch.linspace(0, 1, height).view(1, 1, height, 1)
        smooth_image = (x + y) / 2
        smooth_image = smooth_image.expand(batch_size, channels, height, width)

        loss = image_smoothness_loss(smooth_image)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert loss.item() < 0.01, "Smooth image should have very low loss"

    def test_noisy_image_higher_loss(self):
        """Noisy images should have higher smoothness loss than smooth ones."""
        batch_size, channels, height, width = 2, 3, 64, 64

        # Smooth image
        smooth_image = torch.ones(batch_size, channels, height, width) * 0.5

        # Noisy image
        noisy_image = torch.rand(batch_size, channels, height, width)

        smooth_loss = image_smoothness_loss(smooth_image)
        noisy_loss = image_smoothness_loss(noisy_image)

        assert noisy_loss > smooth_loss, "Noisy image should have higher loss"

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        images = torch.rand(2, 3, 64, 64, requires_grad=True)

        loss = image_smoothness_loss(images)
        loss.backward()

        assert images.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(images.grad).any(), "Gradients should not be NaN"

    def test_output_shape(self):
        """Test that output is a scalar."""
        images = torch.rand(2, 3, 64, 64)
        loss = image_smoothness_loss(images)

        assert loss.ndim == 0, "Loss should be a scalar"

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("image_size", [32, 64, 128])
    def test_different_sizes(self, batch_size, image_size):
        """Test with different batch sizes and image sizes."""
        images = torch.rand(batch_size, 3, image_size, image_size)
        loss = image_smoothness_loss(images)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss).any(), "Loss should not be NaN"


class TestImageRegularizationLoss:
    """Test image regularization loss (MSE between original and transformed)."""

    def test_identical_images_zero_loss(self):
        """Identical images should have zero loss."""
        images = torch.rand(2, 3, 64, 64)

        loss = image_regularization_loss(images, images)

        assert torch.isclose(
            loss, torch.tensor(0.0), atol=1e-6
        ), "Identical images should have zero loss"

    def test_different_images_positive_loss(self):
        """Different images should have positive loss."""
        original = torch.rand(2, 3, 64, 64)
        transformed = torch.rand(2, 3, 64, 64)

        loss = image_regularization_loss(transformed, original)

        assert loss.item() > 0, "Different images should have positive loss"

    def test_gradient_flow(self):
        """Test that gradients flow to transformed images."""
        original = torch.rand(2, 3, 64, 64)
        transformed = torch.rand(2, 3, 64, 64, requires_grad=True)

        loss = image_regularization_loss(transformed, original)
        loss.backward()

        assert transformed.grad is not None, "Gradients should flow to transformed"
        assert not torch.isnan(transformed.grad).any(), "Gradients should not be NaN"

    def test_loss_magnitude(self):
        """Test that loss increases with larger differences."""
        original = torch.ones(2, 3, 64, 64) * 0.5

        # Small difference
        transformed_small = original + 0.01
        loss_small = image_regularization_loss(transformed_small, original)

        # Large difference
        transformed_large = original + 0.1
        loss_large = image_regularization_loss(transformed_large, original)

        assert (
            loss_large > loss_small
        ), "Larger differences should produce larger loss"


class TestBlackLevelPreservationLoss:
    """Test black level preservation loss function."""

    def test_preserved_blacks_zero_loss(self):
        """When dark pixels remain dark, loss should be zero."""
        batch_size, height, width = 2, 64, 64

        # Create images with dark regions
        original = torch.zeros(batch_size, 3, height, width)
        original[:, :, :32, :] = 0.005  # Dark region (below threshold)
        original[:, :, 32:, :] = 0.5  # Bright region

        # Transformed keeps dark regions dark
        transformed = original.clone()

        loss = black_level_preservation_loss(transformed, original, threshold=0.01)

        assert torch.isclose(
            loss, torch.tensor(0.0), atol=1e-6
        ), "Preserved blacks should have zero loss"

    def test_lifted_blacks_positive_loss(self):
        """When dark pixels are lifted, loss should be positive."""
        batch_size, height, width = 2, 64, 64

        # Create images with dark regions
        original = torch.zeros(batch_size, 3, height, width)
        original[:, :, :32, :] = 0.005  # Dark region (below threshold)
        original[:, :, 32:, :] = 0.5  # Bright region

        # Transformed lifts dark regions
        transformed = original.clone()
        transformed[:, :, :32, :] = 0.1  # Lifted blacks

        loss = black_level_preservation_loss(transformed, original, threshold=0.01)

        assert loss.item() > 0, "Lifted blacks should produce positive loss"

    def test_lowered_blacks_zero_loss(self):
        """Lowering blacks (making darker) should not be penalized."""
        batch_size, height, width = 2, 64, 64

        # Create images with dark regions
        original = torch.zeros(batch_size, 3, height, width)
        original[:, :, :32, :] = 0.005  # Dark region
        original[:, :, 32:, :] = 0.5  # Bright region

        # Transformed makes blacks even darker
        transformed = original.clone()
        transformed[:, :, :32, :] = 0.001  # Even darker

        loss = black_level_preservation_loss(transformed, original, threshold=0.01)

        assert torch.isclose(
            loss, torch.tensor(0.0), atol=1e-6
        ), "Making blacks darker should not be penalized"

    def test_threshold_effect(self):
        """Test that threshold parameter affects which pixels are considered dark."""
        batch_size, height, width = 2, 64, 64

        # Create image with pixels at 0.015 luminance
        original = torch.ones(batch_size, 3, height, width) * 0.015
        transformed = torch.ones(batch_size, 3, height, width) * 0.1

        # With threshold=0.01, pixels at 0.015 are NOT dark (no penalty)
        loss_high_threshold = black_level_preservation_loss(
            transformed, original, threshold=0.01
        )

        # With threshold=0.02, pixels at 0.015 ARE dark (penalty)
        loss_low_threshold = black_level_preservation_loss(
            transformed, original, threshold=0.02
        )

        assert (
            loss_low_threshold > loss_high_threshold
        ), "Higher threshold should capture more pixels as dark"

    def test_luma_calculation(self):
        """Test that luminance is calculated correctly using Rec. 709."""
        batch_size, height, width = 2, 64, 64

        # Create pure red, green, and blue images
        red = torch.zeros(batch_size, 3, height, width)
        red[:, 0, :, :] = 1.0  # Red channel

        green = torch.zeros(batch_size, 3, height, width)
        green[:, 1, :, :] = 1.0  # Green channel

        blue = torch.zeros(batch_size, 3, height, width)
        blue[:, 2, :, :] = 1.0  # Blue channel

        # Dark original (below threshold)
        dark_original = torch.zeros(batch_size, 3, height, width)

        # Green should contribute most to luminance (highest Rec. 709 coefficient)
        loss_red = black_level_preservation_loss(red, dark_original, threshold=0.1)
        loss_green = black_level_preservation_loss(green, dark_original, threshold=0.1)
        loss_blue = black_level_preservation_loss(blue, dark_original, threshold=0.1)

        # Green has highest coefficient, so should produce highest loss
        assert (
            loss_green > loss_red and loss_green > loss_blue
        ), "Green should contribute most to luminance"

    def test_gradient_flow(self):
        """Test gradient flow through the loss."""
        original = torch.rand(2, 3, 64, 64)
        transformed = torch.rand(2, 3, 64, 64, requires_grad=True)

        loss = black_level_preservation_loss(transformed, original)
        loss.backward()

        assert transformed.grad is not None, "Gradients should flow"
        assert not torch.isnan(transformed.grad).any(), "Gradients should not be NaN"

    def test_no_dark_pixels(self):
        """Test with no dark pixels (should handle gracefully)."""
        # All bright pixels
        original = torch.ones(2, 3, 64, 64) * 0.5
        transformed = torch.ones(2, 3, 64, 64) * 0.6

        loss = black_level_preservation_loss(original, transformed, threshold=0.01)

        # Should not crash, loss should be very small (epsilon handling)
        assert not torch.isnan(loss), "Should handle no dark pixels gracefully"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestLUTSmoothnessLoss:
    """Test LUT smoothness loss (already tested in test_smoothness.py, but include here for completeness)."""

    def test_identity_lut_low_loss(self):
        """Identity LUT should have low smoothness loss."""
        from utils import identity_lut

        lut = identity_lut(resolution=32)
        loss = lut_smoothness_loss(lut)

        assert loss.item() < 0.01, "Identity LUT should have very low loss"

    def test_random_lut_higher_loss(self):
        """Random LUT should have higher loss than identity."""
        from utils import identity_lut

        identity = identity_lut(resolution=16)
        random_lut = torch.rand(16, 16, 16, 3)

        loss_identity = lut_smoothness_loss(identity)
        loss_random = lut_smoothness_loss(random_lut)

        assert loss_random > loss_identity, "Random LUT should have higher loss"

    def test_gradient_flow(self):
        """Test gradient flow."""
        lut = torch.rand(16, 16, 16, 3, requires_grad=True)

        loss = lut_smoothness_loss(lut)
        loss.backward()

        assert lut.grad is not None, "Gradients should flow"
        assert not torch.isnan(lut.grad).any(), "Gradients should not be NaN"


class TestComputeLosses:
    """Test the compute_losses orchestration function."""

    def test_primary_loss_only(self):
        """Test with only primary loss enabled."""

        def mock_loss_fn(transformed, original):
            return torch.tensor(1.0)

        transformed = torch.rand(2, 3, 64, 64)
        original = torch.rand(2, 3, 64, 64)
        lut = torch.rand(16, 16, 16, 3)

        loss, components = compute_losses(
            loss_fn=mock_loss_fn,
            transformed_images=transformed,
            original_images=original,
            lut_tensor=lut,
            image_text_weight=2.0,
            image_smoothness=0.0,
            image_regularization=0.0,
            black_preservation=0.0,
            lut_smoothness=0.0,
        )

        assert torch.isclose(
            loss, torch.tensor(2.0)
        ), "Loss should be weighted primary loss"
        assert "primary" in components, "Should have primary loss component"
        assert len(components) == 1, "Should only have primary component"

    def test_all_losses_combined(self):
        """Test with all loss components enabled."""

        def mock_loss_fn(transformed, original):
            return torch.tensor(1.0)

        transformed = torch.rand(2, 3, 64, 64)
        original = torch.rand(2, 3, 64, 64)
        lut = torch.rand(16, 16, 16, 3)

        loss, components = compute_losses(
            loss_fn=mock_loss_fn,
            transformed_images=transformed,
            original_images=original,
            lut_tensor=lut,
            image_text_weight=1.0,
            image_smoothness=0.1,
            image_regularization=0.2,
            black_preservation=0.3,
            lut_smoothness=0.4,
        )

        # Should have all components
        assert "primary" in components
        assert "img_smooth" in components
        assert "img_reg" in components
        assert "black" in components
        assert "lut_smooth" in components
        assert len(components) == 5

        # Total loss should be sum of weighted components
        expected_loss = (
            1.0 * components["primary"]
            + 0.1 * components["img_smooth"]
            + 0.2 * components["img_reg"]
            + 0.3 * components["black"]
            + 0.4 * components["lut_smooth"]
        )
        assert torch.isclose(
            loss, expected_loss, atol=1e-6
        ), "Total loss should be sum of weighted components"

    def test_selective_losses(self):
        """Test with selective loss components."""

        def mock_loss_fn(transformed, original):
            return torch.tensor(1.0)

        transformed = torch.rand(2, 3, 64, 64)
        original = torch.rand(2, 3, 64, 64)
        lut = torch.rand(16, 16, 16, 3)

        # Enable only image smoothness and LUT smoothness
        loss, components = compute_losses(
            loss_fn=mock_loss_fn,
            transformed_images=transformed,
            original_images=original,
            lut_tensor=lut,
            image_text_weight=1.0,
            image_smoothness=0.5,
            image_regularization=0.0,
            black_preservation=0.0,
            lut_smoothness=0.5,
        )

        assert "primary" in components
        assert "img_smooth" in components
        assert "lut_smooth" in components
        assert "img_reg" not in components, "Disabled loss should not be in components"
        assert "black" not in components, "Disabled loss should not be in components"
        assert len(components) == 3

    def test_gradient_flow(self):
        """Test that gradients flow through combined losses."""

        def mock_loss_fn(transformed, original):
            return transformed.mean()

        transformed = torch.rand(2, 3, 64, 64, requires_grad=True)
        original = torch.rand(2, 3, 64, 64)
        lut = torch.rand(16, 16, 16, 3, requires_grad=True)

        loss, components = compute_losses(
            loss_fn=mock_loss_fn,
            transformed_images=transformed,
            original_images=original,
            lut_tensor=lut,
            image_text_weight=1.0,
            image_smoothness=0.1,
            image_regularization=0.1,
            black_preservation=0.1,
            lut_smoothness=0.1,
        )

        loss.backward()

        assert transformed.grad is not None, "Gradients should flow to images"
        assert lut.grad is not None, "Gradients should flow to LUT"
        assert not torch.isnan(transformed.grad).any()
        assert not torch.isnan(lut.grad).any()

    def test_zero_weights_no_computation(self):
        """Test that zero weights don't add components to dict."""

        def mock_loss_fn(transformed, original):
            return torch.tensor(1.0)

        transformed = torch.rand(2, 3, 64, 64)
        original = torch.rand(2, 3, 64, 64)
        lut = torch.rand(16, 16, 16, 3)

        loss, components = compute_losses(
            loss_fn=mock_loss_fn,
            transformed_images=transformed,
            original_images=original,
            lut_tensor=lut,
            image_text_weight=1.0,
            image_smoothness=0.0,
            image_regularization=0.0,
            black_preservation=0.0,
            lut_smoothness=0.0,
        )

        # Only primary should be present
        assert set(components.keys()) == {
            "primary"
        }, "Zero-weighted losses should not be computed"

    @pytest.mark.parametrize(
        "weights",
        [
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.1, 0.0, 0.0, 0.0),
            (1.0, 0.1, 0.2, 0.0, 0.0),
            (1.0, 0.1, 0.2, 0.3, 0.0),
            (1.0, 0.1, 0.2, 0.3, 0.4),
        ],
    )
    def test_various_weight_combinations(self, weights):
        """Test with various weight combinations."""

        def mock_loss_fn(transformed, original):
            return torch.tensor(1.0)

        transformed = torch.rand(2, 3, 64, 64)
        original = torch.rand(2, 3, 64, 64)
        lut = torch.rand(16, 16, 16, 3)

        (
            image_text_weight,
            image_smoothness,
            image_regularization,
            black_preservation,
            lut_smoothness_weight,
        ) = weights

        loss, components = compute_losses(
            loss_fn=mock_loss_fn,
            transformed_images=transformed,
            original_images=original,
            lut_tensor=lut,
            image_text_weight=image_text_weight,
            image_smoothness=image_smoothness,
            image_regularization=image_regularization,
            black_preservation=black_preservation,
            lut_smoothness=lut_smoothness_weight,
        )

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert "primary" in components, "Primary loss should always be present"

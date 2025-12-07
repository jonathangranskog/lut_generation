"""Tests for LUT loading, saving, and application."""

import pytest
import torch

from utils.io import read_cube_file, write_cube_file
from utils.transforms import apply_lut, identity_lut


@pytest.fixture
def red_shifted_lut():
    """Create a LUT that adds a red shift to images."""
    lut = identity_lut(resolution=16)
    # Add red shift: increase red channel by 0.15, clamp to [0, 1]
    lut[..., 0] = torch.clamp(lut[..., 0] + 0.15, 0, 1)
    return lut


class TestLUTIO:
    """Tests for LUT file I/O operations."""

    def test_write_and_read_roundtrip(
        self, red_shifted_lut, temp_cube_file, domain_default
    ):
        """Test that saving and loading a LUT preserves its values."""
        # Write the LUT
        write_cube_file(
            str(temp_cube_file),
            red_shifted_lut,
            domain_default["min"],
            domain_default["max"],
            title="Red Shifted LUT Test",
        )

        # Read it back
        lut_reloaded, domain_min, domain_max = read_cube_file(str(temp_cube_file))

        # Check domain values
        assert domain_min == domain_default["min"]
        assert domain_max == domain_default["max"]

        # Check LUT values (allow small float precision differences)
        torch.testing.assert_close(
            red_shifted_lut, lut_reloaded, rtol=1e-5, atol=1e-5
        )

    def test_write_with_custom_domain(self, identity_lut_16, temp_cube_file):
        """Test LUT writing with custom domain values."""
        custom_min = [0.1, 0.1, 0.1]
        custom_max = [0.9, 0.9, 0.9]

        write_cube_file(
            str(temp_cube_file),
            identity_lut_16,
            custom_min,
            custom_max,
            title="Custom Domain Test",
        )

        _, domain_min, domain_max = read_cube_file(str(temp_cube_file))

        assert domain_min == custom_min
        assert domain_max == custom_max

    def test_grayscale_lut_writing(self, temp_cube_file):
        """Test writing a grayscale LUT (single channel)."""
        # Create a single-channel LUT
        grayscale_lut = torch.linspace(0, 1, 16**3).reshape(16, 16, 16, 1)

        write_cube_file(
            str(temp_cube_file),
            grayscale_lut,
            title="Grayscale LUT",
            grayscale=True,
        )

        # Read it back
        lut_reloaded, _, _ = read_cube_file(str(temp_cube_file))

        # Should be replicated to 3 channels
        assert lut_reloaded.shape == (16, 16, 16, 3)
        # All channels should be identical
        torch.testing.assert_close(lut_reloaded[..., 0], lut_reloaded[..., 1])
        torch.testing.assert_close(lut_reloaded[..., 1], lut_reloaded[..., 2])


class TestLUTApplication:
    """Tests for LUT application to images."""

    def test_apply_preserves_shape(self, gradient_image, identity_lut_16):
        """Test that applying a LUT preserves image shape."""
        result = apply_lut(gradient_image, identity_lut_16)
        assert result.shape == gradient_image.shape

    def test_identity_lut_unchanged(self, gradient_image, identity_lut_16):
        """Test that identity LUT doesn't change the image."""
        result = apply_lut(gradient_image, identity_lut_16)
        torch.testing.assert_close(gradient_image, result, rtol=1e-4, atol=1e-4)

    def test_red_shift_increases_red_channel(self, gradient_image, red_shifted_lut):
        """Test that red-shifted LUT increases red channel values."""
        result = apply_lut(gradient_image, red_shifted_lut)

        # Red channel should increase on average
        red_increase = (result[0] - gradient_image[0]).mean()
        assert red_increase > 0.1, "Red channel should increase significantly"

    def test_batch_processing(self, gradient_image, identity_lut_16):
        """Test that LUT application works with batched images."""
        batch = gradient_image.unsqueeze(0).repeat(4, 1, 1, 1)  # (4, 3, H, W)
        result = apply_lut(batch, identity_lut_16)

        assert result.shape == batch.shape
        # All images in batch should be processed identically
        for i in range(1, 4):
            torch.testing.assert_close(result[0], result[i])

    def test_gradient_flow(self, gradient_image, identity_lut_16):
        """Test that gradients flow through LUT application."""
        lut = identity_lut_16.clone()
        lut.requires_grad = True

        result = apply_lut(gradient_image, lut)
        loss = result.sum()
        loss.backward()

        assert lut.grad is not None
        assert lut.grad.abs().sum() > 0


class TestLUTCreation:
    """Tests for LUT creation utilities."""

    def test_identity_lut_shape(self):
        """Test that identity LUT has correct shape."""
        for size in [8, 16, 32]:
            lut = identity_lut(resolution=size)
            assert lut.shape == (size, size, size, 3)

    def test_identity_lut_values(self):
        """Test that identity LUT contains correct values."""
        lut = identity_lut(resolution=16)

        # Check corners
        assert torch.allclose(lut[0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(lut[15, 15, 15], torch.tensor([1.0, 1.0, 1.0]))

        # Check value range
        assert lut.min() >= 0.0
        assert lut.max() <= 1.0

    def test_grayscale_identity_lut(self):
        """Test grayscale identity LUT creation."""
        lut = identity_lut(resolution=16, grayscale=True)
        assert lut.shape == (16, 16, 16, 1)

        # Should contain luminance values
        assert lut.min() >= 0.0
        assert lut.max() <= 1.0


class TestEndToEnd:
    """End-to-end tests combining multiple operations."""

    def test_full_lut_pipeline(
        self, gradient_image, red_shifted_lut, temp_cube_file, domain_default
    ):
        """Test complete pipeline: create LUT -> save -> load -> apply."""
        # Apply original LUT
        result_original = apply_lut(
            gradient_image, red_shifted_lut, domain_default["min"], domain_default["max"]
        )

        # Save LUT
        write_cube_file(
            str(temp_cube_file),
            red_shifted_lut,
            domain_default["min"],
            domain_default["max"],
            title="Pipeline Test",
        )

        # Load LUT
        lut_loaded, domain_min, domain_max = read_cube_file(str(temp_cube_file))

        # Apply loaded LUT
        result_loaded = apply_lut(gradient_image, lut_loaded, domain_min, domain_max)

        # Results should be nearly identical
        torch.testing.assert_close(result_original, result_loaded, rtol=1e-5, atol=1e-5)

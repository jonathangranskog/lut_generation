"""
Tests for grayscale LUT functionality.
"""

import pytest
import torch
from utils import identity_lut, apply_lut, write_cube_file, read_cube_file


@pytest.fixture
def grayscale_lut_16():
    """Create a 16x16x16 grayscale identity LUT."""
    return identity_lut(16, grayscale=True)


@pytest.fixture
def regular_lut_16():
    """Create a 16x16x16 regular (RGB) identity LUT."""
    return identity_lut(16, grayscale=False)


class TestGrayscaleLUTCreation:
    """Test grayscale LUT creation."""

    def test_grayscale_lut_shape(self, grayscale_lut_16):
        """Grayscale LUT should have shape (N, N, N, 1)."""
        lut_size = 16
        assert grayscale_lut_16.shape == (lut_size, lut_size, lut_size, 1)

    def test_regular_lut_shape(self, regular_lut_16):
        """Regular LUT should have shape (N, N, N, 3)."""
        lut_size = 16
        assert regular_lut_16.shape == (lut_size, lut_size, lut_size, 3)

    @pytest.mark.parametrize("lut_size", [8, 16, 32, 64])
    def test_grayscale_lut_different_sizes(self, lut_size):
        """Test grayscale LUT creation with different sizes."""
        grayscale_lut = identity_lut(lut_size, grayscale=True)
        assert grayscale_lut.shape == (lut_size, lut_size, lut_size, 1)

    def test_grayscale_values_consistent(self, grayscale_lut_16):
        """Values in grayscale LUT should be consistent across all positions."""
        # For identity LUT, grayscale value should match the luminance
        # of the RGB identity values at each position
        assert grayscale_lut_16.min() >= 0.0
        assert grayscale_lut_16.max() <= 1.0


class TestGrayscaleLUTApplication:
    """Test applying grayscale LUTs to images."""

    def test_apply_preserves_shape(self, grayscale_lut_16):
        """Applying grayscale LUT should preserve image shape."""
        test_image = torch.rand(3, 64, 64)
        transformed = apply_lut(test_image, grayscale_lut_16)
        assert transformed.shape == test_image.shape

    def test_output_is_grayscale(self, grayscale_lut_16):
        """Output of grayscale LUT should have R=G=B."""
        test_image = torch.rand(3, 64, 64)
        transformed = apply_lut(test_image, grayscale_lut_16)

        # All channels should be equal (R=G=B)
        diff_rg = torch.abs(transformed[0] - transformed[1]).max().item()
        diff_gb = torch.abs(transformed[1] - transformed[2]).max().item()

        assert diff_rg < 1e-5, "Red and green channels should be equal"
        assert diff_gb < 1e-5, "Green and blue channels should be equal"

    def test_batch_processing(self, grayscale_lut_16):
        """Grayscale LUT should work with batched images."""
        batch_images = torch.rand(4, 3, 32, 32)
        batch_transformed = apply_lut(batch_images, grayscale_lut_16)

        assert batch_transformed.shape == batch_images.shape

        # Verify all outputs are grayscale
        for i in range(batch_images.shape[0]):
            diff_rg = torch.abs(
                batch_transformed[i, 0] - batch_transformed[i, 1]
            ).max()
            diff_gb = torch.abs(
                batch_transformed[i, 1] - batch_transformed[i, 2]
            ).max()
            assert diff_rg < 1e-5
            assert diff_gb < 1e-5

    def test_gradient_flow(self, grayscale_lut_16):
        """Gradients should flow through grayscale LUT application."""
        test_image = torch.rand(3, 64, 64, requires_grad=True)
        transformed = apply_lut(test_image, grayscale_lut_16)
        loss = transformed.mean()
        loss.backward()

        assert test_image.grad is not None
        assert not torch.isnan(test_image.grad).any()

    @pytest.mark.parametrize("image_size", [32, 64, 128, 256])
    def test_different_image_sizes(self, grayscale_lut_16, image_size):
        """Grayscale LUT should work with different image sizes."""
        test_image = torch.rand(3, image_size, image_size)
        transformed = apply_lut(test_image, grayscale_lut_16)

        assert transformed.shape == test_image.shape
        # Verify output is grayscale
        diff_rg = torch.abs(transformed[0] - transformed[1]).max()
        assert diff_rg < 1e-5


class TestGrayscaleLUTIO:
    """Test saving and loading grayscale LUTs."""

    def test_save_and_load_roundtrip(self, grayscale_lut_16, temp_cube_file):
        """Grayscale LUT should survive save/load roundtrip."""
        lut_size = 16

        # Save grayscale LUT
        write_cube_file(
            temp_cube_file, grayscale_lut_16, grayscale=True, title="Test Grayscale"
        )

        # Load it back
        loaded_lut, domain_min, domain_max = read_cube_file(temp_cube_file)

        # When saved with grayscale=True, single channel is replicated to 3
        assert loaded_lut.shape == (lut_size, lut_size, lut_size, 3)

        # All channels should be equal
        diff_01 = torch.abs(loaded_lut[..., 0] - loaded_lut[..., 1]).max().item()
        diff_12 = torch.abs(loaded_lut[..., 1] - loaded_lut[..., 2]).max().item()

        assert diff_01 < 1e-5, "Channels 0 and 1 should be equal"
        assert diff_12 < 1e-5, "Channels 1 and 2 should be equal"

    def test_loaded_grayscale_lut_works(self, grayscale_lut_16, temp_cube_file):
        """Loaded grayscale LUT should produce grayscale output."""
        # Save and load
        write_cube_file(
            temp_cube_file, grayscale_lut_16, grayscale=True, title="Test"
        )
        loaded_lut, _, _ = read_cube_file(temp_cube_file)

        # Apply to test image
        test_image = torch.rand(3, 64, 64)
        transformed = apply_lut(test_image, loaded_lut)

        # Output should be grayscale
        diff_rg = torch.abs(transformed[0] - transformed[1]).max()
        diff_gb = torch.abs(transformed[1] - transformed[2]).max()

        assert diff_rg < 1e-5
        assert diff_gb < 1e-5

    def test_grayscale_title_in_file(self, grayscale_lut_16, temp_cube_file):
        """Saved grayscale LUT file should contain correct title."""
        title = "My Grayscale LUT"
        write_cube_file(temp_cube_file, grayscale_lut_16, grayscale=True, title=title)

        # Read file and check title is present
        with open(temp_cube_file, "r") as f:
            content = f.read()
            assert title in content


class TestGrayscaleVsRegular:
    """Test differences between grayscale and regular LUTs."""

    def test_shape_difference(self):
        """Grayscale and regular LUTs should have different channel dimensions."""
        lut_size = 16
        grayscale = identity_lut(lut_size, grayscale=True)
        regular = identity_lut(lut_size, grayscale=False)

        assert grayscale.shape[-1] == 1
        assert regular.shape[-1] == 3

    def test_grayscale_produces_grayscale_output(self):
        """Grayscale LUT should produce grayscale images, regular LUT should not."""
        lut_size = 16
        grayscale_lut = identity_lut(lut_size, grayscale=True)
        regular_lut = identity_lut(lut_size, grayscale=False)

        # Create a colorful test image (R, G, B all different)
        test_image = torch.stack(
            [
                torch.ones(64, 64) * 0.8,  # Red channel
                torch.ones(64, 64) * 0.5,  # Green channel
                torch.ones(64, 64) * 0.2,  # Blue channel
            ]
        )

        # Apply grayscale LUT
        gray_output = apply_lut(test_image, grayscale_lut)
        diff_rg = torch.abs(gray_output[0] - gray_output[1]).max()
        diff_gb = torch.abs(gray_output[1] - gray_output[2]).max()
        assert diff_rg < 1e-5 and diff_gb < 1e-5, "Grayscale output should have R=G=B"

        # Apply regular LUT (identity should preserve differences)
        regular_output = apply_lut(test_image, regular_lut)
        diff_rg = torch.abs(regular_output[0] - regular_output[1]).max()
        diff_gb = torch.abs(regular_output[1] - regular_output[2]).max()
        assert diff_rg > 0.1 or diff_gb > 0.1, "Regular output should preserve colors"

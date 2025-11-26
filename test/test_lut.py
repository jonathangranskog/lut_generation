"""
Tests for LUT loading, saving, and application.
"""

import os
import tempfile
import torch

# Add parent directory to path to import lut module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import read_cube_file, write_cube_file
from utils.transforms import apply_lut, identity_lut


def create_gradient_image(width=256, height=256):
    """
    Create a synthetic gradient test image with RGB gradients.

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W) with values in [0, 1]
    """
    # Create horizontal gradient for R
    r = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)

    # Create vertical gradient for G
    g = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)

    # Create diagonal gradient for B
    x = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)
    y = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)
    b = (x + y) / 2.0

    # Stack into (3, H, W) tensor
    image = torch.stack([r, g, b], dim=0)

    return image


def create_red_shifted_lut(resolution=16, red_shift=0.1):
    """
    Create a LUT that adds a red shift to images.

    Args:
        resolution: Size of the LUT cube
        red_shift: Amount to add to the red channel (default 0.1)

    Returns:
        torch.Tensor: LUT tensor of shape (resolution, resolution, resolution, 3)
    """
    # Start with identity LUT
    lut = identity_lut(resolution=resolution)

    # Add red shift: increase red channel, clamp to [0, 1]
    lut[..., 0] = torch.clamp(lut[..., 0] + red_shift, 0, 1)

    return lut


def test_lut_roundtrip():
    """
    Test that saving a LUT and loading it again produces the same result
    when applied to a synthetic gradient image.
    """
    # Create synthetic test data
    image_tensor = create_gradient_image(width=128, height=128)
    lut_tensor_original = create_red_shifted_lut(resolution=16, red_shift=0.15)
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]

    # Apply the original LUT
    result_original = apply_lut(
        image_tensor, lut_tensor_original, domain_min, domain_max
    )

    # Save the LUT to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cube", delete=False
    ) as tmp_file:
        tmp_lut_path = tmp_file.name

    try:
        # Write the LUT
        write_cube_file(
            tmp_lut_path,
            lut_tensor_original,
            domain_min,
            domain_max,
            title="Red Shifted LUT Test",
        )

        # Load the saved LUT
        lut_tensor_reloaded, domain_min_reloaded, domain_max_reloaded = read_cube_file(
            tmp_lut_path
        )

        # Apply the reloaded LUT
        result_reloaded = apply_lut(
            image_tensor, lut_tensor_reloaded, domain_min_reloaded, domain_max_reloaded
        )

        # Compare the results
        # Check that domain values match
        assert domain_min == domain_min_reloaded, "Domain min values don't match"
        assert domain_max == domain_max_reloaded, "Domain max values don't match"

        # Check that LUT tensors are very close (allow for small floating point differences)
        lut_diff = torch.abs(lut_tensor_original - lut_tensor_reloaded).max().item()
        assert lut_diff < 1e-5, f"LUT tensors differ by {lut_diff}"

        # Check that the applied results are very close
        result_diff = torch.abs(result_original - result_reloaded).max().item()
        assert result_diff < 1e-5, f"Applied results differ by {result_diff}"

        # Verify that red shift was applied
        # The result should have more red than the original image
        red_increase = (result_original[0] - image_tensor[0]).mean().item()
        assert red_increase > 0, (
            f"Expected red channel to increase, but it changed by {red_increase}"
        )

        print("✓ LUT roundtrip test passed!")
        print(f"  LUT size: {lut_tensor_original.shape}")
        print(f"  Max LUT difference: {lut_diff:.2e}")
        print(f"  Max result difference: {result_diff:.2e}")
        print(f"  Average red channel increase: {red_increase:.4f}")

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_lut_path):
            os.remove(tmp_lut_path)


def test_synthetic_lut_shape():
    """Test that synthetic LUT has the correct shape."""
    lut_tensor = create_red_shifted_lut(resolution=32, red_shift=0.1)

    # Check that it's 4D
    assert lut_tensor.ndim == 4, f"Expected 4D tensor, got {lut_tensor.ndim}D"

    # Check that it's cubic
    assert lut_tensor.shape[0] == lut_tensor.shape[1] == lut_tensor.shape[2], (
        f"LUT is not cubic: {lut_tensor.shape}"
    )

    # Check that it has 3 channels
    assert lut_tensor.shape[3] == 3, f"Expected 3 channels, got {lut_tensor.shape[3]}"

    # Check that red channel is shifted
    identity = identity_lut(resolution=32)
    red_diff = (lut_tensor[..., 0] - identity[..., 0]).mean().item()
    assert red_diff > 0, "Red channel should be increased"

    print("✓ Synthetic LUT shape test passed!")
    print(f"  LUT shape: {lut_tensor.shape}")
    print(f"  Average red shift: {red_diff:.4f}")


def test_gradient_image_shape():
    """Test that synthetic gradient image has correct properties."""
    image = create_gradient_image(width=128, height=128)

    # Check shape
    assert image.shape == (3, 128, 128), f"Expected (3, 128, 128), got {image.shape}"

    # Check value range
    assert image.min() >= 0.0, "Image values should be >= 0"
    assert image.max() <= 1.0, "Image values should be <= 1"

    # Check that it has actual gradients (not uniform)
    assert image[0].std() > 0.1, "Red channel should have variation"
    assert image[1].std() > 0.1, "Green channel should have variation"
    assert image[2].std() > 0.1, "Blue channel should have variation"

    print("✓ Gradient image test passed!")
    print(f"  Image shape: {image.shape}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    print(
        f"  Channel std: R={image[0].std():.3f}, G={image[1].std():.3f}, B={image[2].std():.3f}"
    )


if __name__ == "__main__":
    # Run tests
    print("Running LUT tests with synthetic data...\n")

    test_gradient_image_shape()
    print()
    test_synthetic_lut_shape()
    print()
    test_lut_roundtrip()

    print("\n✓ All tests passed!")

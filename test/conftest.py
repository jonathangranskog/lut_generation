"""Pytest configuration and shared fixtures for LUT generation tests."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cube_file(temp_dir):
    """Provide a temporary .cube file path."""
    return temp_dir / "test.cube"


@pytest.fixture
def gradient_image():
    """
    Create a synthetic gradient test image with RGB gradients.

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W) with values in [0, 1]
    """
    width, height = 128, 128

    # Create horizontal gradient for R
    r = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)

    # Create vertical gradient for G
    g = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)

    # Create diagonal gradient for B
    x = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)
    y = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)
    b = (x + y) / 2.0

    # Stack into (3, H, W) tensor
    return torch.stack([r, g, b], dim=0)


@pytest.fixture
def identity_lut_16():
    """Provide a 16x16x16 identity LUT."""
    from utils.transforms import identity_lut

    return identity_lut(resolution=16)


@pytest.fixture
def sample_image_folder(temp_dir):
    """Create a folder with sample test images."""
    images_dir = temp_dir / "images"
    images_dir.mkdir()

    # Create a few simple test images
    for i in range(3):
        img = Image.new("RGB", (256, 256), color=(i * 80, 100, 150))
        img.save(images_dir / f"test_image_{i}.jpg")

    return images_dir


@pytest.fixture
def domain_default():
    """Provide default domain values."""
    return {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}

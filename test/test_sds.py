"""
Tests for SDS (Score Distillation Sampling) loss with DeepFloyd IF.

Note: These tests require HuggingFace authentication and license acceptance.
To run:
1. huggingface-cli login
2. Accept license at: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0
"""

import pytest
import torch


def test_sds_constants():
    """Test that SDS constants are properly defined."""
    from utils.constants import (
        DEEPFLOYD_IMAGE_SIZE,
        DEEPFLOYD_UNET_SIZE,
        DEEPFLOYD_STAGE1_MODEL,
        DEEPFLOYD_STAGE1_MODEL_MEDIUM,
    )

    # UNet operates at 64x64
    assert DEEPFLOYD_UNET_SIZE == 64
    # Dataset crops at higher resolution (256x256) for context
    assert DEEPFLOYD_IMAGE_SIZE == 256
    assert "DeepFloyd" in DEEPFLOYD_STAGE1_MODEL
    assert "DeepFloyd" in DEEPFLOYD_STAGE1_MODEL_MEDIUM


def test_sds_preprocess_images():
    """Test image preprocessing for SDS (no model required)."""
    # Create a mock preprocessing function to test the logic
    from utils.constants import DEEPFLOYD_UNET_SIZE

    def preprocess_images(images: torch.Tensor) -> torch.Tensor:
        """Standalone preprocessing logic for testing."""
        # Resize to IF Stage I UNet resolution (64x64)
        if images.shape[-2:] != (DEEPFLOYD_UNET_SIZE, DEEPFLOYD_UNET_SIZE):
            images = torch.nn.functional.interpolate(
                images,
                size=(DEEPFLOYD_UNET_SIZE, DEEPFLOYD_UNET_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        # Convert from [0, 1] to [-1, 1]
        images = images * 2.0 - 1.0
        return images

    # Test with various input sizes (including typical dataset crop size of 256x256)
    test_cases = [
        (1, 3, 256, 256),  # Typical dataset crop size
        (2, 3, 512, 512),
        (4, 3, 64, 64),   # Already at UNet size
        (1, 3, 128, 128),
    ]

    for shape in test_cases:
        images = torch.rand(shape)
        processed = preprocess_images(images)

        # Check output shape is UNet size (64x64)
        assert processed.shape == (shape[0], 3, DEEPFLOYD_UNET_SIZE, DEEPFLOYD_UNET_SIZE)

        # Check output range is [-1, 1]
        assert processed.min() >= -1.0
        assert processed.max() <= 1.0


def test_sds_gradient_flow_mock():
    """Test gradient flow through a mock SDS-like loss computation."""
    # This tests the gradient flow logic without loading the actual model

    batch_size = 2
    image_size = 64

    # Create input images with gradient tracking
    images = torch.rand(batch_size, 3, image_size, image_size, requires_grad=True)

    # Simulate preprocessing
    x0 = images * 2.0 - 1.0

    # Simulate noise addition
    noise = torch.randn_like(x0)
    alpha_t = 0.5
    noisy_images = alpha_t**0.5 * x0 + (1 - alpha_t) ** 0.5 * noise

    # Simulate noise prediction (detached - no gradient through model)
    noise_pred = torch.randn_like(x0).detach()

    # SDS gradient computation
    w = 1.0 - alpha_t
    grad = w * (noise_pred - noise)
    target = (noisy_images - grad).detach()

    # Compute loss
    loss = 0.5 * torch.nn.functional.mse_loss(noisy_images, target)

    # Test gradient flow
    loss.backward()

    assert images.grad is not None, "Gradients should flow back to input images"
    assert images.grad.shape == images.shape
    assert not torch.isnan(images.grad).any(), "Gradients should not be NaN"
    assert not torch.isinf(images.grad).any(), "Gradients should not be infinite"


def test_sds_timestep_sampling():
    """Test timestep sampling logic."""
    min_timestep = 20
    max_timestep = 980
    batch_size = 10

    # Sample timesteps
    timesteps = torch.randint(min_timestep, max_timestep, (batch_size,))

    # Verify range
    assert (timesteps >= min_timestep).all()
    assert (timesteps < max_timestep).all()
    assert timesteps.shape == (batch_size,)


def test_sds_cfg_logic():
    """Test classifier-free guidance logic."""
    batch_size = 2
    channels = 3
    size = 64

    # Simulate noise predictions
    noise_pred_uncond = torch.randn(batch_size, channels, size, size)
    noise_pred_cond = torch.randn(batch_size, channels, size, size)

    guidance_scale = 20.0

    # Apply CFG
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    # Verify shape preserved
    assert noise_pred.shape == noise_pred_uncond.shape

    # Verify CFG effect (result should differ from both)
    assert not torch.allclose(noise_pred, noise_pred_uncond)
    assert not torch.allclose(noise_pred, noise_pred_cond)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for DeepFloyd IF model loading",
)
@pytest.mark.skip(reason="Requires HuggingFace authentication and license acceptance")
def test_sds_loss_full():
    """Full integration test with actual model (requires auth)."""
    from models.sds import SDSLoss

    device = "cuda"
    sds_loss = SDSLoss(
        prompt="warm golden hour sunlight",
        device=device,
        use_medium_model=True,
    )

    # Create test images
    batch_size = 2
    images = torch.rand(batch_size, 3, 256, 256, device=device, requires_grad=True)

    # Compute loss
    loss = sds_loss(images)

    # Verify loss is scalar
    assert loss.ndim == 0

    # Test gradient flow
    loss.backward()
    assert images.grad is not None
    assert not torch.isnan(images.grad).any()

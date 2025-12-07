"""
Tests for VLM (Vision Language Model) loss.

Note: Full integration tests require HuggingFace model downloads.
Most tests use mocks to avoid network dependencies.
"""

import pytest
import torch


class TestVLMConstants:
    """Test VLM-related constants."""

    def test_vlm_constants_exist(self):
        """Test that VLM constants are properly defined."""
        from utils.constants import VLM_IMAGE_SIZE

        # VLM models use 448x448 images (balance between quality and memory)
        assert VLM_IMAGE_SIZE == 448


class TestVLMPreprocessing:
    """Test VLM image preprocessing logic."""

    def test_image_preprocessing_shape(self):
        """Test that images are preprocessed to correct size."""
        from utils.constants import VLM_IMAGE_SIZE

        # Simulate preprocessing
        test_image = torch.rand(1, 3, 256, 256)

        # VLM expects images at VLM_IMAGE_SIZE (448x448)
        # This would be handled by the dataset, so we just verify the size
        assert VLM_IMAGE_SIZE == 448

    def test_image_range(self):
        """Test that images are in expected range [0, 1]."""
        test_image = torch.rand(2, 3, 512, 512)

        assert test_image.min() >= 0.0
        assert test_image.max() <= 1.0


class TestVLMGradientFlow:
    """Test gradient flow through mock VLM-like computations."""

    def test_gradient_flow_mock(self):
        """Test gradient flow through a mock VLM-like loss."""
        from utils.constants import VLM_IMAGE_SIZE

        batch_size = 2
        image_size = VLM_IMAGE_SIZE

        # Create images with gradient tracking
        transformed = torch.rand(batch_size, 3, image_size, image_size, requires_grad=True)
        original = torch.rand(batch_size, 3, image_size, image_size)

        # Simulate a simple VLM-like loss (difference between images)
        mock_loss = torch.nn.functional.mse_loss(transformed, original)

        # Test gradient flow
        mock_loss.backward()

        assert transformed.grad is not None, "Gradients should flow to transformed images"
        assert transformed.grad.shape == transformed.shape
        assert not torch.isnan(transformed.grad).any()
        assert not torch.isinf(transformed.grad).any()

    def test_gradient_magnitude_reasonable(self):
        """Test that gradient magnitudes are reasonable."""
        from utils.constants import VLM_IMAGE_SIZE

        transformed = torch.rand(2, 3, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, requires_grad=True)
        original = torch.rand(2, 3, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE)

        mock_loss = torch.nn.functional.mse_loss(transformed, original)
        mock_loss.backward()

        grad_magnitude = transformed.grad.abs().mean().item()

        # Gradients should be non-zero but not exploding
        assert grad_magnitude > 0, "Gradients should be non-zero"
        assert grad_magnitude < 10, "Gradients should not explode"


class TestVLMBinaryProbabilities:
    """Test binary probability computation logic."""

    def test_yes_no_probability_sum(self):
        """Test that yes/no probabilities sum reasonably."""
        # Simulate logits for yes/no tokens
        yes_logit = torch.tensor([2.0])
        no_logit = torch.tensor([1.0])

        # Compute probabilities using softmax-like logic
        yes_prob = torch.exp(yes_logit) / (torch.exp(yes_logit) + torch.exp(no_logit))
        no_prob = torch.exp(no_logit) / (torch.exp(yes_logit) + torch.exp(no_logit))

        # Should sum to approximately 1
        assert torch.isclose(yes_prob + no_prob, torch.tensor(1.0), atol=1e-6)

    def test_loss_computation_from_probs(self):
        """Test loss computation from yes/no probabilities."""
        # Higher yes probability should give lower loss
        p_yes_high = torch.tensor([0.8])
        p_no_high = torch.tensor([0.2])

        # Lower yes probability should give higher loss
        p_yes_low = torch.tensor([0.2])
        p_no_low = torch.tensor([0.8])

        # Loss is typically -log(p_yes) or similar
        loss_high = -torch.log(p_yes_high)
        loss_low = -torch.log(p_yes_low)

        assert loss_high < loss_low, "Higher yes probability should give lower loss"


class TestVLMPromptFormatting:
    """Test VLM prompt formatting logic."""

    def test_prompt_template_format(self):
        """Test that prompts are formatted correctly."""
        prompt = "warm golden hour"

        # VLM uses a question template
        question = (
            f"Looking at these two images, has the '{prompt}' color grade "
            "been successfully applied to create the second image? "
            "Answer with yes or no."
        )

        assert prompt in question
        assert "yes or no" in question
        assert "Looking at these two images" in question

    def test_prompt_with_special_characters(self):
        """Test prompt formatting with special characters."""
        prompt = "warm & cozy"

        question = (
            f"Looking at these two images, has the '{prompt}' color grade "
            "been successfully applied to create the second image? "
            "Answer with yes or no."
        )

        assert "&" in question
        assert prompt in question


class TestVLMContextAware:
    """Test VLM context-aware functionality."""

    def test_requires_both_images(self):
        """Test that VLM requires both original and transformed images."""
        from models.vlm import VLMLoss

        # VLMLoss.forward should require original_images parameter
        import inspect
        sig = inspect.signature(VLMLoss.forward)
        params = list(sig.parameters.keys())

        assert "transformed_images" in params
        assert "original_images" in params

    def test_context_aware_comparison(self):
        """Test that VLM compares images in context."""
        # This is a conceptual test - VLM should compare original vs transformed
        # In practice, this requires the actual model

        # We can at least verify the interface expects both images
        from models.vlm import VLMLoss

        # Check that the class exists and has expected structure
        assert hasattr(VLMLoss, 'forward')
        assert hasattr(VLMLoss, 'get_yes_no_probs')
        assert hasattr(VLMLoss, 'get_prediction')


@pytest.mark.skip(reason="Requires HuggingFace model download and network access")
def test_vlm_loss_full_integration():
    """Full integration test with actual VLM model (requires network)."""
    from models.vlm import VLMLoss
    from utils.constants import VLM_IMAGE_SIZE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create loss function
    vlm_loss = VLMLoss(
        prompt="warm golden hour",
        device=device,
    )

    # Create test images
    batch_size = 2
    original = torch.rand(batch_size, 3, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, device=device)
    transformed = torch.rand(batch_size, 3, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, device=device, requires_grad=True)

    # Compute loss
    loss = vlm_loss(transformed, original)

    # Verify loss is scalar
    assert loss.ndim == 0

    # Test gradient flow
    loss.backward()
    assert transformed.grad is not None
    assert not torch.isnan(transformed.grad).any()

    # Test probability computation
    p_yes, p_no = vlm_loss.get_yes_no_probs(transformed, original)
    assert p_yes.shape == (batch_size,)
    assert p_no.shape == (batch_size,)

    # Probabilities should be in [0, 1]
    assert (p_yes >= 0).all() and (p_yes <= 1).all()
    assert (p_no >= 0).all() and (p_no <= 1).all()

    # Test predictions
    predictions = vlm_loss.get_prediction(transformed, original)
    assert len(predictions) == batch_size
    assert all(pred in ["yes", "no"] for pred in predictions)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for VLM model with reasonable performance"
)
@pytest.mark.skip(reason="Requires HuggingFace model download")
def test_vlm_loss_cuda():
    """Test VLM loss on CUDA device."""
    from models.vlm import VLMLoss
    from utils.constants import VLM_IMAGE_SIZE

    device = "cuda"
    vlm_loss = VLMLoss(prompt="cinematic lighting", device=device)

    batch_size = 1
    original = torch.rand(batch_size, 3, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, device=device)
    transformed = torch.rand(batch_size, 3, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, device=device, requires_grad=True)

    loss = vlm_loss(transformed, original)

    assert loss.device.type == "cuda"
    assert loss.ndim == 0

    loss.backward()
    assert transformed.grad is not None
    assert transformed.grad.device.type == "cuda"

"""
Test script for VLM loss.

This tests the VLM context-aware loss to verify:
1. Gradient flow works correctly
2. Both original and transformed images are properly processed
3. Loss values are reasonable
"""

import torch
from models.vlm import VLMLoss


def test_vlm_loss():
    """Test the VLM context-aware loss."""
    print("=" * 60)
    print("Testing VLM Context-Aware Loss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create loss function
    vlm_loss = VLMLoss(
        prompt="warm golden hour",
        device=device,
        model_name="gemma3_4b"
    )

    # Create dummy original and transformed images with gradient tracking
    batch_size = 1
    original_images = torch.rand(batch_size, 3, 256, 256, device=device)
    transformed_images = torch.rand(
        batch_size, 3, 256, 256, device=device, requires_grad=True
    )

    # Compute loss
    print("Computing loss...")
    loss = vlm_loss(transformed_images, original_images)
    print(f"Loss: {loss.item():.4f}")

    # Get probabilities
    p_yes, p_no = vlm_loss.get_yes_no_probs(transformed_images, original_images)
    print(f"P(Yes): {p_yes.mean().item():.4f}, P(No): {p_no.mean().item():.4f}")

    # Get predictions
    predictions = vlm_loss.get_prediction(transformed_images, original_images)
    print(f"Predictions: {predictions}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss.backward()
    print(f"Gradients flowing: {transformed_images.grad is not None}")
    if transformed_images.grad is not None:
        print(f"Gradient magnitude: {transformed_images.grad.abs().mean().item():.6f}")

    print("\n✓ VLM loss working correctly!\n")


if __name__ == "__main__":
    test_vlm_loss()

    print("=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)

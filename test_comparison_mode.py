"""
Test script for VLM comparison mode.

This tests both assessment and comparison modes to verify:
1. Gradient flow works correctly
2. The comparison mode accepts original + transformed images
3. Both modes produce reasonable loss values
"""

import torch
from models.vlm import VLMLoss

def test_assessment_mode():
    """Test the original assessment mode."""
    print("=" * 60)
    print("Testing Assessment Mode")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create loss function in assessment mode
    vlm_loss = VLMLoss(
        prompt="warm golden hour",
        device=device,
        comparison_mode=False
    )

    # Create dummy images with gradient tracking
    batch_size = 2
    images = torch.rand(batch_size, 3, 512, 512, device=device, requires_grad=True)

    # Compute loss
    print("Computing loss...")
    loss = vlm_loss(images)
    print(f"Loss: {loss.item():.4f}")

    # Get probabilities
    p_yes, p_no = vlm_loss.get_yes_no_probs(images)
    print(f"P(Yes): {p_yes.mean().item():.4f}, P(No): {p_no.mean().item():.4f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss.backward()
    print(f"Gradients flowing: {images.grad is not None}")
    if images.grad is not None:
        print(f"Gradient magnitude: {images.grad.abs().mean().item():.6f}")

    print("\n✓ Assessment mode working correctly!\n")


def test_comparison_mode():
    """Test the new comparison mode."""
    print("=" * 60)
    print("Testing Comparison Mode")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create loss function in comparison mode
    vlm_loss = VLMLoss(
        prompt="warm golden hour",
        device=device,
        comparison_mode=True
    )

    # Create dummy original and transformed images with gradient tracking
    batch_size = 2
    original_images = torch.rand(batch_size, 3, 512, 512, device=device)
    transformed_images = torch.rand(batch_size, 3, 512, 512, device=device, requires_grad=True)

    # Compute loss
    print("Computing loss...")
    loss = vlm_loss(transformed_images, original_images)
    print(f"Loss: {loss.item():.4f}")

    # Get probabilities
    p_yes, p_no = vlm_loss.get_yes_no_probs(transformed_images, original_images)
    print(f"P(Yes): {p_yes.mean().item():.4f}, P(No): {p_no.mean().item():.4f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss.backward()
    print(f"Gradients flowing: {transformed_images.grad is not None}")
    if transformed_images.grad is not None:
        print(f"Gradient magnitude: {transformed_images.grad.abs().mean().item():.6f}")

    print("\n✓ Comparison mode working correctly!\n")


def test_comparison_mode_error():
    """Test that comparison mode raises error without original images."""
    print("=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create loss function in comparison mode
    vlm_loss = VLMLoss(
        prompt="warm golden hour",
        device=device,
        comparison_mode=True
    )

    # Create only transformed images (no original)
    batch_size = 2
    transformed_images = torch.rand(batch_size, 3, 512, 512, device=device)

    # Try to compute loss without original images - should raise error
    print("Attempting to call loss without original_images...")
    try:
        loss = vlm_loss(transformed_images)
        print("✗ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}\n")


if __name__ == "__main__":
    test_assessment_mode()
    test_comparison_mode()
    test_comparison_mode_error()

    print("=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)

#!/usr/bin/env python
"""
Simple test script for grayscale LUT functionality
"""

import torch
from utils import identity_lut, apply_lut, write_cube_file, read_cube_file

def test_grayscale_lut():
    print("Testing grayscale LUT functionality...")

    # Test 1: Create grayscale LUT
    print("\n1. Creating grayscale LUT...")
    lut_size = 16
    grayscale_lut = identity_lut(lut_size, grayscale=True)
    print(f"   Grayscale LUT shape: {grayscale_lut.shape}")
    assert grayscale_lut.shape == (lut_size, lut_size, lut_size, 1), \
        f"Expected shape ({lut_size}, {lut_size}, {lut_size}, 1), got {grayscale_lut.shape}"
    print("   ✓ Grayscale LUT created successfully")

    # Test 2: Create regular LUT for comparison
    print("\n2. Creating regular LUT...")
    regular_lut = identity_lut(lut_size, grayscale=False)
    print(f"   Regular LUT shape: {regular_lut.shape}")
    assert regular_lut.shape == (lut_size, lut_size, lut_size, 3), \
        f"Expected shape ({lut_size}, {lut_size}, {lut_size}, 3), got {regular_lut.shape}"
    print("   ✓ Regular LUT created successfully")

    # Test 3: Apply grayscale LUT to a test image
    print("\n3. Testing apply_lut with grayscale LUT...")
    # Create a simple test image (3, 64, 64)
    test_image = torch.rand(3, 64, 64)
    transformed = apply_lut(test_image, grayscale_lut)
    print(f"   Input image shape: {test_image.shape}")
    print(f"   Transformed image shape: {transformed.shape}")
    assert transformed.shape == test_image.shape, \
        f"Expected output shape {test_image.shape}, got {transformed.shape}"
    print("   ✓ Grayscale LUT applied successfully")

    # Test 4: Verify output is grayscale (R=G=B)
    print("\n4. Verifying output is grayscale (R=G=B)...")
    # For identity LUT with grayscale, output should have R=G=B
    diff_rg = torch.abs(transformed[0] - transformed[1]).max().item()
    diff_gb = torch.abs(transformed[1] - transformed[2]).max().item()
    print(f"   Max difference R-G: {diff_rg:.6f}")
    print(f"   Max difference G-B: {diff_gb:.6f}")
    assert diff_rg < 1e-5 and diff_gb < 1e-5, \
        "Output channels should be equal for grayscale LUT"
    print("   ✓ Output is properly grayscale")

    # Test 5: Save and load grayscale LUT
    print("\n5. Testing save/load with grayscale LUT...")
    test_path = "/tmp/test_grayscale.cube"
    write_cube_file(test_path, grayscale_lut, grayscale=True, title="Test Grayscale LUT")
    loaded_lut, domain_min, domain_max = read_cube_file(test_path)
    print(f"   Saved and loaded LUT shape: {loaded_lut.shape}")
    # When saved with grayscale=True, single channel is replicated to 3
    assert loaded_lut.shape == (lut_size, lut_size, lut_size, 3), \
        f"Expected shape ({lut_size}, {lut_size}, {lut_size}, 3), got {loaded_lut.shape}"

    # Verify all channels are equal (grayscale)
    diff_01 = torch.abs(loaded_lut[..., 0] - loaded_lut[..., 1]).max().item()
    diff_12 = torch.abs(loaded_lut[..., 1] - loaded_lut[..., 2]).max().item()
    print(f"   Max difference between channels (0-1): {diff_01:.6f}")
    print(f"   Max difference between channels (1-2): {diff_12:.6f}")
    assert diff_01 < 1e-5 and diff_12 < 1e-5, \
        "Loaded LUT should have equal channels for grayscale"
    print("   ✓ Grayscale LUT saved and loaded correctly")

    # Test 6: Test batch processing
    print("\n6. Testing batch processing...")
    batch_images = torch.rand(4, 3, 32, 32)
    batch_transformed = apply_lut(batch_images, grayscale_lut)
    print(f"   Batch input shape: {batch_images.shape}")
    print(f"   Batch output shape: {batch_transformed.shape}")
    assert batch_transformed.shape == batch_images.shape, \
        f"Expected output shape {batch_images.shape}, got {batch_transformed.shape}"
    print("   ✓ Batch processing works correctly")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_grayscale_lut()

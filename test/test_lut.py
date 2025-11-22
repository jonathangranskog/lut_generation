"""
Tests for LUT loading, saving, and application.
"""

import os
import tempfile
import torch
import numpy as np
from PIL import Image

# Add parent directory to path to import lut module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lut import read_cube_file, write_cube_file, apply_lut


def test_lut_roundtrip():
    """
    Test that loading a .cube file, saving it, and loading it again
    produces the same result when applied to an image.
    """
    # Paths to test files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lut_path = os.path.join(project_root, "Aladdin .cube")
    image_path = os.path.join(project_root, "macbeth.jpeg")
    
    # Verify test files exist
    assert os.path.exists(lut_path), f"LUT file not found: {lut_path}"
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    
    # Load the original LUT
    lut_tensor_original, domain_min, domain_max = read_cube_file(lut_path)
    
    # Load the test image
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    image_tensor = image_tensor.float() / 255.0
    
    # Apply the original LUT
    result_original = apply_lut(image_tensor, lut_tensor_original, domain_min, domain_max)
    
    # Save the LUT to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as tmp_file:
        tmp_lut_path = tmp_file.name
    
    try:
        # Write the LUT
        write_cube_file(
            tmp_lut_path,
            lut_tensor_original,
            domain_min,
            domain_max,
            title="Aladdin LUT Test"
        )
        
        # Load the saved LUT
        lut_tensor_reloaded, domain_min_reloaded, domain_max_reloaded = read_cube_file(tmp_lut_path)
        
        # Apply the reloaded LUT
        result_reloaded = apply_lut(image_tensor, lut_tensor_reloaded, domain_min_reloaded, domain_max_reloaded)
        
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
        
        print(f"✓ LUT roundtrip test passed!")
        print(f"  LUT size: {lut_tensor_original.shape}")
        print(f"  Max LUT difference: {lut_diff:.2e}")
        print(f"  Max result difference: {result_diff:.2e}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_lut_path):
            os.remove(tmp_lut_path)


def test_lut_tensor_shape():
    """Test that loaded LUT has the correct shape."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lut_path = os.path.join(project_root, "Aladdin .cube")
    
    lut_tensor, domain_min, domain_max = read_cube_file(lut_path)
    
    # Check that it's 4D
    assert lut_tensor.ndim == 4, f"Expected 4D tensor, got {lut_tensor.ndim}D"
    
    # Check that it's cubic
    assert lut_tensor.shape[0] == lut_tensor.shape[1] == lut_tensor.shape[2], \
        f"LUT is not cubic: {lut_tensor.shape}"
    
    # Check that it has 3 channels
    assert lut_tensor.shape[3] == 3, f"Expected 3 channels, got {lut_tensor.shape[3]}"
    
    # Check domain values are lists of 3 floats
    assert len(domain_min) == 3, "domain_min should have 3 values"
    assert len(domain_max) == 3, "domain_max should have 3 values"
    
    print(f"✓ LUT shape test passed!")
    print(f"  LUT shape: {lut_tensor.shape}")
    print(f"  Domain min: {domain_min}")
    print(f"  Domain max: {domain_max}")


if __name__ == "__main__":
    # Run tests
    print("Running LUT tests...\n")
    
    test_lut_tensor_shape()
    print()
    test_lut_roundtrip()
    
    print("\n✓ All tests passed!")


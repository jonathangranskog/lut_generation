import re
from typing import Tuple

import torch


def read_cube_file(lut_path: str) -> Tuple[torch.Tensor, list, list]:
    """
    Load a .cube LUT file and return as a PyTorch tensor

    Args:
        lut_path: Path to the .cube file

    Returns:
        Tuple of (lut_tensor, domain_min, domain_max)
        lut_tensor: 3D tensor of shape (size, size, size, 3) for RGB lookup
        domain_min: List of 3 floats for minimum domain values
        domain_max: List of 3 floats for maximum domain values
    """
    with open(lut_path, "r") as f:
        lines = f.readlines()

    # Parse header information
    lut_size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]

    # Find where the actual LUT data starts
    data_start_idx = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip comments and empty lines
        if line.startswith("#") or not line:
            continue

        # Parse LUT_3D_SIZE
        if line.startswith("LUT_3D_SIZE"):
            lut_size = int(line.split()[1])

        # Parse DOMAIN_MIN (optional)
        elif line.startswith("DOMAIN_MIN"):
            domain_min = [float(x) for x in line.split()[1:4]]

        # Parse DOMAIN_MAX (optional)
        elif line.startswith("DOMAIN_MAX"):
            domain_max = [float(x) for x in line.split()[1:4]]

        # Check if this line looks like RGB data (3 floats)
        elif re.match(r"^[\d\.\-\s]+$", line) and len(line.split()) == 3:
            data_start_idx = i
            break

    if lut_size is None:
        raise ValueError("LUT_3D_SIZE not found in cube file")

    # Parse the RGB data
    lut_data = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith("#"):
            try:
                r, g, b = map(float, line.split())
                lut_data.append([r, g, b])
            except ValueError:
                continue  # Skip invalid lines

    # Verify we have the right amount of data
    expected_entries = lut_size**3
    if len(lut_data) != expected_entries:
        raise ValueError(f"Expected {expected_entries} entries, got {len(lut_data)}")

    # Convert to tensor and reshape
    lut_tensor = torch.tensor(lut_data, dtype=torch.float32)

    # Reshape to 3D cube indexed as [B][G][R] (blue varies slowest, red varies fastest)
    # Output values remain in RGB order as stored in the file
    lut_cube = lut_tensor.reshape(lut_size, lut_size, lut_size, 3)

    return lut_cube, domain_min, domain_max


def apply_lut(
    image: torch.Tensor,
    lut_tensor: torch.Tensor,
    domain_min: list = [0.0, 0.0, 0.0],
    domain_max: list = [1.0, 1.0, 1.0],
) -> torch.Tensor:
    """
    Apply a 3D LUT using PyTorch's grid_sample for trilinear interpolation

    Args:
        image: Input image tensor, either (C, H, W) or (H, W, C) format
        lut_tensor: LUT tensor of shape (size, size, size, 3)
        domain_min: Minimum domain values for scaling (default [0.0, 0.0, 0.0])
        domain_max: Maximum domain values for scaling (default [1.0, 1.0, 1.0])

    Returns:
        LUT-applied image in the same format as input
    """
    # Ensure image is in (H, W, C) format
    if image.shape[0] == 3:  # (C, H, W)
        x = image.permute(1, 2, 0)
    else:
        x = image

    # Apply domain scaling if provided
    assert len(domain_min) == 3, "Domain min must be a 3-element list"
    assert len(domain_max) == 3, "Domain max must be a 3-element list"
    domain_min_t = torch.tensor(domain_min).to(x.device)
    domain_max_t = torch.tensor(domain_max).to(x.device)
    domain_scaled = (x - domain_min_t) / (domain_max_t - domain_min_t)

    # Delta-based approach for HDR extrapolation:
    # Sample LUT at clamped coordinates, compute delta from identity,
    # then apply delta to unclamped input to preserve HDR values
    clamped_coords = torch.clamp(domain_scaled, 0, 1)

    # Prepare for grid_sample: need (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
    # LUT is indexed as [B][G][R] (cube file format), so we maintain that order
    # permute(3, 0, 1, 2) transforms (B, G, R, 3) -> (3, B, G, R)
    lut = lut_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(x.device)  # (1, 3, B, G, R)

    # Convert RGB coordinates to BGR for LUT indexing (matches GLSL shader's color.bgr)
    clamped_coords_bgr = clamped_coords.flip(-1)

    # Image coordinates need to be in [-1, 1] range for grid_sample
    # Scale from [0, 1] to [-1, 1]
    coords = clamped_coords_bgr * 2.0 - 1.0

    # Reshape coordinates to (1, H*W, 1, 1, 3) for sampling
    H, W = domain_scaled.shape[:2]
    coords = coords.view(1, H * W, 1, 1, 3)

    # Sample the LUT with trilinear interpolation
    lut_sampled = torch.nn.functional.grid_sample(
        lut, coords, mode="bilinear", padding_mode="border", align_corners=False
    )

    # Reshape LUT output back to (H, W, 3) from (3, H * W)
    lut_sampled = lut_sampled.squeeze().permute(1, 0).view(H, W, 3)

    # Flip the result to match the original color space
    result = lut_sampled.flip(-1)

    # Return in original format
    if image.shape[0] == 3:
        return result.permute(2, 0, 1)
    else:
        return result


def identity_lut(resolution: int = 32) -> torch.Tensor:
    """
    Create identity LUT using meshgrid.
    Uses BGR indexing order to match cube file format.
    At position [b,g,r], outputs RGB value [r,g,b] to preserve original color.
    """
    coords = torch.linspace(0, 1, resolution)
    # Create identity LUT: position [b,g,r] outputs [r,g,b]
    b, g, r = torch.meshgrid(coords, coords, coords, indexing="ij")
    identity_lut = torch.stack([r, g, b], dim=-1)
    return identity_lut

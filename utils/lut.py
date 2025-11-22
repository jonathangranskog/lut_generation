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
        image: Input image tensor in one of the following formats:
               - (C, H, W): Single image, channels first
               - (H, W, C): Single image, channels last
               - (B, C, H, W): Batch of images, channels first
               - (B, H, W, C): Batch of images, channels last
        lut_tensor: LUT tensor of shape (size, size, size, 3)
        domain_min: Minimum domain values for scaling (default [0.0, 0.0, 0.0])
        domain_max: Maximum domain values for scaling (default [1.0, 1.0, 1.0])

    Returns:
        LUT-applied image(s) in the same format as input
    """
    is_batched = image.ndim == 4

    # Normalize to (B, H, W, C) format
    if is_batched:
        if image.shape[1] == 3:  # (B, C, H, W)
            x = image.permute(0, 2, 3, 1)
            channels_first = True
        else:  # (B, H, W, C)
            x = image
            channels_first = False
    else:
        # Add batch dimension for single image
        if image.shape[0] == 3:  # (C, H, W)
            x = image.permute(1, 2, 0).unsqueeze(0)
            channels_first = True
        else:  # (H, W, C)
            x = image.unsqueeze(0)
            channels_first = False

    B, H, W, C = x.shape

    # Apply domain scaling if provided
    assert len(domain_min) == 3, "Domain min must be a 3-element list"
    assert len(domain_max) == 3, "Domain max must be a 3-element list"
    domain_min_t = torch.tensor(domain_min).to(x.device)
    domain_max_t = torch.tensor(domain_max).to(x.device)
    domain_scaled = (x - domain_min_t) / (domain_max_t - domain_min_t)

    # Clamp coordinates for LUT lookup
    clamped_coords = torch.clamp(domain_scaled, 0, 1)

    # Prepare for grid_sample: need (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
    # LUT is indexed as [B][G][R] (cube file format), so we maintain that order
    # permute(3, 0, 1, 2) transforms (B, G, R, 3) -> (3, B, G, R)
    lut = lut_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(x.device)  # (1, 3, B, G, R)
    # Expand LUT for batch size
    lut = lut.expand(B, -1, -1, -1, -1)  # (B, 3, B, G, R)

    # Convert RGB coordinates to BGR for LUT indexing (matches GLSL shader's color.bgr)
    clamped_coords_bgr = clamped_coords.flip(-1)

    # Image coordinates need to be in [-1, 1] range for grid_sample
    # Scale from [0, 1] to [-1, 1]
    coords = clamped_coords_bgr * 2.0 - 1.0

    # Reshape coordinates to (B, H*W, 1, 1, 3) for sampling
    coords = coords.view(B, H * W, 1, 1, 3)

    # Sample the LUT with trilinear interpolation
    lut_sampled = torch.nn.functional.grid_sample(
        lut, coords, mode="bilinear", padding_mode="border", align_corners=False
    )

    # Reshape LUT output back to (B, H, W, 3)
    lut_sampled = lut_sampled.view(B, 3, H, W).permute(0, 2, 3, 1)

    # Flip the result to match the original color space
    result = lut_sampled.flip(-1)

    # Return in original format
    if not is_batched:
        result = result.squeeze(0)  # Remove batch dimension
        if channels_first:
            return result.permute(2, 0, 1)
        else:
            return result
    else:
        if channels_first:
            return result.permute(0, 3, 1, 2)
        else:
            return result


def write_cube_file(
    lut_path: str,
    lut_tensor: torch.Tensor,
    domain_min: list = [0.0, 0.0, 0.0],
    domain_max: list = [1.0, 1.0, 1.0],
    title: str = "Generated LUT",
) -> None:
    """
    Save a PyTorch LUT tensor to a .cube file

    Args:
        lut_path: Path where the .cube file will be saved
        lut_tensor: LUT tensor of shape (size, size, size, 3)
        domain_min: Minimum domain values (default [0.0, 0.0, 0.0])
        domain_max: Maximum domain values (default [1.0, 1.0, 1.0])
        title: Optional title for the LUT file
    """
    assert lut_tensor.ndim == 4, "LUT tensor must be 4D (size, size, size, 3)"
    assert lut_tensor.shape[-1] == 3, "LUT tensor must have 3 channels (RGB)"
    assert len(domain_min) == 3, "Domain min must be a 3-element list"
    assert len(domain_max) == 3, "Domain max must be a 3-element list"

    lut_size = lut_tensor.shape[0]

    # Verify cube shape
    assert lut_tensor.shape[0] == lut_tensor.shape[1] == lut_tensor.shape[2], (
        "LUT must be cubic (same size in all dimensions)"
    )

    with open(lut_path, "w") as f:
        # Write header
        f.write(f"# {title}\n")
        f.write("# Generated with PyTorch LUT tools\n")
        f.write(f"LUT_3D_SIZE {lut_size}\n")

        # Write domain info if not default
        if domain_min != [0.0, 0.0, 0.0]:
            f.write(
                f"DOMAIN_MIN {domain_min[0]:.6f} {domain_min[1]:.6f} {domain_min[2]:.6f}\n"
            )
        if domain_max != [1.0, 1.0, 1.0]:
            f.write(
                f"DOMAIN_MAX {domain_max[0]:.6f} {domain_max[1]:.6f} {domain_max[2]:.6f}\n"
            )

        f.write("\n")

        # Flatten LUT data in BGR indexing order (blue varies slowest, red fastest)
        # The tensor is already in [B][G][R] order from our format
        lut_data = lut_tensor.reshape(-1, 3).cpu()

        # Write RGB values, one per line
        for rgb in lut_data:
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")


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


def image_smoothness_loss(images: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation (smoothness) loss on images.

    Penalizes abrupt changes between adjacent pixels, preventing banding
    and other discontinuities in the output images.

    Args:
        images: Batch of images, shape (B, C, H, W)

    Returns:
        Scalar smoothness loss (mean squared differences across spatial dims)
    """
    # Compute differences along height (vertical)
    diff_h = images[:, :, 1:, :] - images[:, :, :-1, :]

    # Compute differences along width (horizontal)
    diff_w = images[:, :, :, 1:] - images[:, :, :, :-1]

    # Total variation loss (L2)
    loss = (diff_h**2).mean() + (diff_w**2).mean()

    return loss


def image_regularization_loss(
    transformed_images: torch.Tensor, original_images: torch.Tensor
) -> torch.Tensor:
    """
    Penalize deviation from original images.

    Encourages the LUT to make subtle adjustments rather than
    extreme transformations.

    Args:
        transformed_images: LUT-transformed images, shape (B, C, H, W)
        original_images: Original input images, shape (B, C, H, W)

    Returns:
        Scalar loss (MSE between original and transformed images)
    """
    return torch.nn.functional.mse_loss(transformed_images, original_images)

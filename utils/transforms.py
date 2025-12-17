import torch
import torch.nn.functional as F


def apply_lut(
    image: torch.Tensor,
    lut_tensor: torch.Tensor,
    domain_min: list[float] | None = None,
    domain_max: list[float] | None = None,
) -> torch.Tensor:
    """
    Apply a 3D LUT using PyTorch's grid_sample for trilinear interpolation

    Args:
        image: Input image tensor in one of the following formats:
               - (C, H, W): Single image, channels first
               - (H, W, C): Single image, channels last
               - (B, C, H, W): Batch of images, channels first
               - (B, H, W, C): Batch of images, channels last
        lut_tensor: LUT tensor of shape (size, size, size, 3) or (size, size, size, 1) for grayscale
        domain_min: Minimum domain values for scaling (default [0.0, 0.0, 0.0])
        domain_max: Maximum domain values for scaling (default [1.0, 1.0, 1.0])

    Returns:
        LUT-applied image(s) in the same format as input (always 3-channel output)
    """
    if domain_min is None:
        domain_min = [0.0, 0.0, 0.0]
    if domain_max is None:
        domain_max = [1.0, 1.0, 1.0]

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

    # Check if LUT is grayscale (single-channel)
    is_grayscale_lut = lut_tensor.shape[-1] == 1
    lut_channels = lut_tensor.shape[-1]

    # Apply domain scaling if provided
    assert len(domain_min) == 3, "Domain min must be a 3-element list"
    assert len(domain_max) == 3, "Domain max must be a 3-element list"
    domain_min_t = torch.tensor(domain_min).to(x.device)
    domain_max_t = torch.tensor(domain_max).to(x.device)
    domain_scaled = (x - domain_min_t) / (domain_max_t - domain_min_t)

    # Clamp coordinates for LUT lookup
    clamped_coords = torch.clamp(domain_scaled, 0, 1)

    # Prepare for grid_sample: need (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
    # LUT is indexed spatially as [B][G][R] and stores RGB values at each position
    # permute(3, 0, 1, 2) transforms (B, G, R, C) -> (C, B, G, R) where C is 1 or 3
    lut = lut_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(x.device)  # (1, C, B, G, R)
    # Expand LUT for batch size
    lut = lut.expand(B, -1, -1, -1, -1)  # (B, C, B, G, R)

    # For grid_sample, grid[..., 0] samples W (rightmost) dimension
    # grid[..., 1] samples H dimension, grid[..., 2] samples D (leftmost)
    # LUT dims are (D, H, W) = (B, G, R), so pixel RGB=[r,g,b] maps to grid [r,g,b]
    # which samples [W, H, D] = [R, G, B] correctly!

    # Image coordinates need to be in [-1, 1] range for grid_sample
    # Scale from [0, 1] to [-1, 1]
    coords = clamped_coords * 2.0 - 1.0

    # Reshape coordinates to (B, H*W, 1, 1, 3) for sampling
    coords = coords.view(B, H * W, 1, 1, 3)

    # Sample the LUT with trilinear interpolation
    lut_sampled = F.grid_sample(
        lut, coords, mode="bilinear", padding_mode="border", align_corners=False
    )

    # Reshape LUT output back to (B, H, W, C) where C is 1 or 3
    lut_sampled = lut_sampled.view(B, lut_channels, H, W).permute(0, 2, 3, 1)

    # If grayscale LUT, replicate single channel to 3 channels
    if is_grayscale_lut:
        # Replicate the single channel to RGB (all channels get same value)
        result = lut_sampled.repeat(1, 1, 1, 3)  # (B, H, W, 3)
    else:
        # Result is already in RGB order (LUT stores RGB values)
        result = lut_sampled

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


def identity_lut(resolution: int = 32, grayscale: bool = False) -> torch.Tensor:
    """
    Create identity LUT using meshgrid.
    Creates [B][G][R] spatial indexing for grid_sample compatibility.
    At position [b,g,r], stores RGB output value [r,g,b] to preserve original color.

    Args:
        resolution: Size of the LUT cube (e.g., 16, 32, 64)
        grayscale: If True, creates a single-channel LUT (size, size, size, 1)
                   where the output is the luminance of the input RGB value.
                   If False, creates a 3-channel LUT (size, size, size, 3).

    Returns:
        LUT tensor of shape (size, size, size, 1) if grayscale else (size, size, size, 3)
        with spatial indexing [B][G][R]
    """
    coords = torch.linspace(0, 1, resolution)
    # Create identity LUT: at spatial position [b,g,r], store RGB value [r,g,b]
    # grid_sample expects [B][G][R] where R is last spatial dim
    b, g, r = torch.meshgrid(coords, coords, coords, indexing="ij")

    if grayscale:
        # Compute luminance using Rec. 709 coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
        # Position [b,g,r] should output luminance of [r,g,b]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        identity_lut = luminance.unsqueeze(-1)  # (size, size, size, 1)
    else:
        identity_lut = torch.stack([r, g, b], dim=-1)  # (size, size, size, 3)

    return identity_lut


def _downsample_upsample_3d(
    lut: torch.Tensor, scale_factor: float = 0.5
) -> torch.Tensor:
    original_shape = lut.shape
    num_channels = original_shape[-1]  # Can be 1 or 3
    # Reshape from (D, H, W, C) to (1, C, D, H, W) for interpolation
    lut_reshaped = lut.permute(3, 0, 1, 2).view(
        1, num_channels, original_shape[0], original_shape[1], original_shape[2]
    )
    # Downsample then upsample for smoothing effect
    lut_downsampled = F.interpolate(
        lut_reshaped, scale_factor=scale_factor, mode="trilinear"
    )
    lut_upsampled = F.interpolate(
        lut_downsampled,
        size=(original_shape[0], original_shape[1], original_shape[2]),
        mode="trilinear",
    )
    # Reshape back to original format (D, H, W, C)
    lut_result = (
        lut_upsampled[0]
        .permute(1, 2, 3, 0)
        .view(original_shape[0], original_shape[1], original_shape[2], num_channels)
    )
    return lut_result


def postprocess_lut(lut: torch.Tensor) -> torch.Tensor:
    return _downsample_upsample_3d(lut)

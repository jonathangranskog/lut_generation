import torch
import torch.nn.functional as F

from .constants import REC709_LUMA_R, REC709_LUMA_G, REC709_LUMA_B
from .transforms import _downsample_upsample_3d


def image_smoothness_loss(images: torch.Tensor) -> torch.Tensor:
    B, C, H, W = images.shape
    downsampled_images = F.interpolate(images, scale_factor=0.5, mode="bilinear")
    upsampled_images = F.interpolate(downsampled_images, size=(H, W), mode="bilinear")
    return F.mse_loss(images, upsampled_images)


def image_regularization_loss(
    transformed_images: torch.Tensor, original_images: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(transformed_images, original_images)


def black_level_preservation_loss(
    transformed_images: torch.Tensor,
    original_images: torch.Tensor,
    threshold: float = 1e-2,
) -> torch.Tensor:
    # Compute luminance (approximate perceptual brightness)
    # Using Rec. 709 luma coefficients
    orig_luma = (
        REC709_LUMA_R * original_images[:, 0, :, :]
        + REC709_LUMA_G * original_images[:, 1, :, :]
        + REC709_LUMA_B * original_images[:, 2, :, :]
    )

    trans_luma = (
        REC709_LUMA_R * transformed_images[:, 0, :, :]
        + REC709_LUMA_G * transformed_images[:, 1, :, :]
        + REC709_LUMA_B * transformed_images[:, 2, :, :]
    )

    # Create mask for dark pixels in original image
    dark_mask = (orig_luma < threshold).float()

    # Compute how much dark pixels have been lifted
    # Only penalize when transformed > original (lifting blacks)
    lift = torch.relu(trans_luma - orig_luma)

    # Apply mask and compute loss only on dark regions
    masked_lift = lift * dark_mask

    # Return mean lift in dark regions
    # Add small epsilon to avoid division by zero if no dark pixels
    num_dark_pixels = dark_mask.sum() + 1e-6
    loss = masked_lift.sum() / num_dark_pixels
    return loss


def lut_smoothness_loss(lut: torch.Tensor) -> torch.Tensor:
    smoothed_lut = _downsample_upsample_3d(lut)
    return F.mse_loss(lut, smoothed_lut)

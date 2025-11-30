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


def compute_losses(
    loss_fn,
    transformed_images: torch.Tensor,
    original_images: torch.Tensor,
    lut_tensor: torch.Tensor,
    image_smoothness: float,
    image_regularization: float,
    black_preservation: float,
    lut_smoothness: float,
) -> tuple[torch.Tensor, dict]:
    primary_loss = loss_fn(transformed_images)
    loss = primary_loss
    loss_components = {"primary": primary_loss}

    if image_smoothness > 0:
        img_smooth_loss = image_smoothness_loss(transformed_images)
        loss = loss + image_smoothness * img_smooth_loss
        loss_components["img_smooth"] = img_smooth_loss

    if image_regularization > 0:
        img_reg_loss = image_regularization_loss(transformed_images, original_images)
        loss = loss + image_regularization * img_reg_loss
        loss_components["img_reg"] = img_reg_loss

    if black_preservation > 0:
        black_loss = black_level_preservation_loss(transformed_images, original_images)
        loss = loss + black_preservation * black_loss
        loss_components["black"] = black_loss

    if lut_smoothness > 0:
        lut_smooth_loss = lut_smoothness_loss(lut_tensor)
        loss = loss + lut_smoothness * lut_smooth_loss
        loss_components["lut_smooth"] = lut_smooth_loss

    return loss, loss_components

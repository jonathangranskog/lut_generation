"""Image conversion utilities for tensors and PIL Images."""

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL Image.

    Args:
        tensor: Image tensor of shape (C, H, W) in [0, 1] range

    Returns:
        PIL Image in RGB format
    """
    # Move to CPU and convert to numpy
    img_array = tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    # Create PIL Image
    return Image.fromarray(img_array)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.

    Args:
        image: PIL Image (will be converted to RGB if not already)

    Returns:
        Image tensor of shape (C, H, W) in [0, 1] range
    """
    # Ensure RGB format
    image = image.convert("RGB")
    # Convert to numpy array
    image_array = np.array(image)
    # Convert to tensor and permute to (C, H, W)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    # Normalize to [0, 1] range
    image_tensor = image_tensor.float() / 255.0
    return image_tensor


def save_tensor_as_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a PyTorch tensor as an image file.

    Args:
        tensor: Image tensor of shape (C, H, W) in [0, 1] range
        path: Path where the image will be saved
    """
    img = tensor_to_pil(tensor)
    img.save(path)

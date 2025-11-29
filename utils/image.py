"""Image conversion utilities for tensors and PIL Images."""

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor (C, H, W) in [0, 1] range to PIL Image."""
    img_array = tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to tensor (C, H, W) in [0, 1] range."""
    image = image.convert("RGB")
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = image_tensor.float() / 255.0
    return image_tensor


def save_tensor_as_image(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor (C, H, W) in [0, 1] range as an image file."""
    img = tensor_to_pil(tensor)
    img.save(path)

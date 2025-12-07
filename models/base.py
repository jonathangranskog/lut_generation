"""Base class for LUT optimization loss functions."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LUTLoss(nn.Module, ABC):
    """
    Abstract base class for LUT optimization loss functions.

    All loss functions should inherit from this class and implement the forward method.
    The forward method computes a scalar loss that guides LUT optimization via gradients.

    The loss should be differentiable with respect to the transformed images, allowing
    gradients to flow back through the LUT application.
    """

    @abstractmethod
    def forward(
        self, transformed_images: torch.Tensor, original_images: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute loss for transformed images.

        Args:
            transformed_images: Batch of images after LUT application, shape (B, C, H, W) in [0, 1]
            original_images: Optional batch of original images before LUT application.
                           Some loss functions (e.g., VLM) use this for comparison,
                           others (e.g., CLIP) ignore it. Shape (B, C, H, W) in [0, 1]

        Returns:
            Scalar loss tensor. Lower loss = better alignment with objective.
            Gradients should flow through transformed_images to enable LUT optimization.
        """
        pass

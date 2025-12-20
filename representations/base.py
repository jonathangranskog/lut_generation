"""Abstract base class for optimizable representations."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseRepresentation(nn.Module, ABC):
    """Abstract base class for all optimizable representations.

    All representations must inherit from this class and implement the required
    abstract methods. This ensures a consistent interface for optimization,
    inference, and I/O operations.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def smoothness_loss(self) -> torch.Tensor:
        """Compute the smoothness loss for this representation.

        Returns:
            torch.Tensor: Scalar tensor representing the smoothness loss.
        """
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Apply the representation to a batch of images.

        Args:
            images: Input images of shape (B, H, W, C) in range [0, 1]
            training: Whether in training mode (skip postprocessing if True)

        Returns:
            torch.Tensor: Transformed images of the same shape as input.
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, file_path: str) -> "BaseRepresentation":
        """Load a representation from a file.

        Args:
            file_path: Path to the file to load from.

        Returns:
            BaseRepresentation: Instance of the representation loaded from file.
        """
        pass

    @abstractmethod
    def write(self, file_path: str) -> None:
        """Save the representation to a file.

        Args:
            file_path: Path to save the file to.
        """
        pass

    def postprocess(self) -> None:
        """Apply postprocessing to the representation.

        This is called during writing and during non-training inference.
        By default, this is a no-op. Subclasses can override to implement
        custom postprocessing logic (e.g., clamping, smoothing, etc.).
        """
        pass

    @abstractmethod
    def clamp(self) -> None:
        """Clamp the representation values to valid ranges.

        This is typically called after each optimization step to ensure
        the representation stays within valid bounds.
        """
        pass

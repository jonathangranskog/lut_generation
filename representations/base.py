"""Abstract base class for optimizable representations."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseRepresentation(nn.Module, ABC):
    """Abstract base class for all optimizable representations."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def smoothness_loss(self) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def read(cls, file_path: str) -> "BaseRepresentation":
        pass

    @abstractmethod
    def write(self, file_path: str) -> None:
        pass

    def postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply postprocessing. By default returns tensor unchanged."""
        return tensor

    @abstractmethod
    def clamp(self) -> None:
        pass

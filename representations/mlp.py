"""MLP-based representation for learned color transformation."""

import torch
import torch.nn as nn
from .base import BaseRepresentation


class MLP(BaseRepresentation):
    """
    MLP representation that learns to transform RGB colors.

    Uses a ResNet-style architecture where the network learns a residual
    that is added to the input color: output = input + network(input).

    Args:
        num_layers: Number of hidden layers (default: 2)
        hidden_width: Width of hidden layers (default: 128)
        init_scale: Scale for weight initialization (default: 0.01)
    """

    def __init__(
        self,
        num_layers: int = 2,
        hidden_width: int = 128,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.init_scale = init_scale

        # Build MLP layers
        layers = []

        # Input layer: 3 (RGB) -> hidden_width
        layers.append(nn.Linear(3, hidden_width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())

        # Output layer: hidden_width -> 3 (RGB residual)
        layers.append(nn.Linear(hidden_width, 3))

        self.network = nn.Sequential(*layers)

        # Initialize with low magnitude weights for near-identity start
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with low magnitude for near-identity behavior."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                # Small random initialization
                nn.init.normal_(module.weight, mean=0.0, std=self.init_scale)
                nn.init.zeros_(module.bias)

        # Make the final layer even smaller to start near identity
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=self.init_scale * 0.1)
            nn.init.zeros_(final_layer.bias)

    def smoothness_loss(self) -> torch.Tensor:
        """
        Weight regularization as smoothness loss.

        Returns L2 norm of all weights to encourage smooth transformations.
        """
        l2_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.sum(param**2)
        return l2_reg

    def forward(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Apply learned color transformation to images.

        Args:
            images: Input images, shape (B, C, H, W) or (C, H, W) or (B, H, W, C) or (H, W, C)
            training: Whether in training mode (unused for MLP, kept for interface)

        Returns:
            Transformed images in the same format as input
        """
        is_batched = images.ndim == 4

        # Normalize to (B, H, W, C) format
        if is_batched:
            if images.shape[1] == 3:  # (B, C, H, W)
                x = images.permute(0, 2, 3, 1)
                channels_first = True
            else:  # (B, H, W, C)
                x = images
                channels_first = False
        else:
            if images.shape[0] == 3:  # (C, H, W)
                x = images.permute(1, 2, 0).unsqueeze(0)
                channels_first = True
            else:  # (H, W, C)
                x = images.unsqueeze(0)
                channels_first = False

        B, H, W, C = x.shape

        # Flatten spatial dimensions for MLP: (B, H, W, 3) -> (B*H*W, 3)
        x_flat = x.reshape(-1, 3)

        # Apply network to get residual
        residual = self.network(x_flat)

        # ResNet-style: add residual to input
        output_flat = x_flat + residual

        # Reshape back: (B*H*W, 3) -> (B, H, W, 3)
        result = output_flat.reshape(B, H, W, C)

        # Return in original format
        if not is_batched:
            result = result.squeeze(0)
            if channels_first:
                return result.permute(2, 0, 1)
            else:
                return result
        else:
            if channels_first:
                return result.permute(0, 3, 1, 2)
            else:
                return result

    @classmethod
    def read(cls, file_path: str) -> "MLP":
        """Load MLP from a checkpoint file."""
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        # Create instance with saved hyperparameters
        mlp = cls(
            num_layers=checkpoint["num_layers"],
            hidden_width=checkpoint["hidden_width"],
            init_scale=checkpoint.get("init_scale", 0.01),
        )

        # Load state dict
        mlp.load_state_dict(checkpoint["state_dict"])

        return mlp

    def write(self, file_path: str, title: str = "Generated MLP") -> None:
        """Save MLP to a checkpoint file."""
        # Handle file extension - use .pt for MLP models
        if file_path.endswith(".cube"):
            file_path = file_path.rsplit(".", 1)[0] + ".pt"

        checkpoint = {
            "title": title,
            "num_layers": self.num_layers,
            "hidden_width": self.hidden_width,
            "init_scale": self.init_scale,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, file_path)

    def clamp(self) -> None:
        """No clamping needed for MLP weights."""
        pass

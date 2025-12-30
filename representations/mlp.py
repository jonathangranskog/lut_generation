"""MLP-based representation for learned color transformation."""

import torch
import torch.nn as nn
from .base import BaseRepresentation


class MLP(BaseRepresentation):
    """
    MLP that learns RGB color transformations with ResNet-style residual.
    Output = input + network(input).
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

        # Build network
        layers = []
        layers.append(nn.Linear(3, hidden_width))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_width, 3))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.init_scale)
                nn.init.zeros_(module.bias)

        # Final layer even smaller for near-identity start
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=self.init_scale * 0.1)
            nn.init.zeros_(final_layer.bias)

    def smoothness_loss(self) -> torch.Tensor:
        """L2 weight regularization."""
        l2_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.sum(param**2)
        return l2_reg

    def _apply_mlp(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Apply network + residual addition."""
        residual = self.network(x_flat)
        return x_flat + residual

    def forward(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        is_batched = images.ndim == 4

        # Normalize to (B, H, W, C) format
        if is_batched:
            if images.shape[1] == 3:  # (B, C, H, W)
                x = images.permute(0, 2, 3, 1)
                channels_first = True
            else:
                x = images
                channels_first = False
        else:
            if images.shape[0] == 3:  # (C, H, W)
                x = images.permute(1, 2, 0).unsqueeze(0)
                channels_first = True
            else:
                x = images.unsqueeze(0)
                channels_first = False

        B, H, W, C = x.shape
        x_flat = x.reshape(-1, 3)
        output_flat = self._apply_mlp(x_flat)

        result = output_flat.reshape(B, H, W, C)

        # Return in original format
        if not is_batched:
            result = result.squeeze(0)
            if channels_first:
                return result.permute(2, 0, 1)
            return result
        else:
            if channels_first:
                return result.permute(0, 3, 1, 2)
            return result

    @classmethod
    def read(cls, file_path: str) -> "MLP":
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
        mlp = cls(
            num_layers=checkpoint["num_layers"],
            hidden_width=checkpoint["hidden_width"],
            init_scale=checkpoint.get("init_scale", 0.01),
        )
        mlp.load_state_dict(checkpoint["state_dict"])
        return mlp

    def write(self, file_path: str, title: str = "Generated MLP") -> None:
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
        pass

"""Configuration system for LUT optimization."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Literal

RepresentationType = Literal["lut", "bw_lut"]
ImageTextLossType = Literal["clip", "gemma3_4b", "gemma3_12b", "gemma3_27b", "sds"]


@dataclass
class LossWeights:
    """Loss weight configuration."""

    image_text: float = 1.0
    image_smoothness: float = 1.0
    image_regularization: float = 1.0
    black_preservation: float = 1.0
    repr_smoothness: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "LossWeights":
        """Create LossWeights from a dictionary, using defaults for missing keys."""
        # Get valid field names from the dataclass
        valid_fields = {f.name for f in fields(cls)}
        # Filter to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class Config:
    """Configuration for LUT optimization."""

    representation: RepresentationType = "lut"
    image_text_loss_type: ImageTextLossType = "clip"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    representation_args: Dict[str, Any] = field(default_factory=dict)
    steps: int = 500
    learning_rate: float = 0.005
    batch_size: int = 4

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create a Config from a dictionary."""
        # Handle nested LossWeights
        loss_weights_data = data.get("loss_weights", {})
        loss_weights = LossWeights.from_dict(loss_weights_data)

        # representation_args is just a dict, copy it directly
        repr_args = dict(data.get("representation_args", {}))

        # Get valid field names for Config (excluding nested ones we handle specially)
        config_fields = {f.name for f in fields(cls)}
        simple_fields = config_fields - {"loss_weights", "representation_args"}

        # Build kwargs for simple fields
        kwargs = {k: v for k, v in data.items() if k in simple_fields}

        return cls(
            loss_weights=loss_weights,
            representation_args=repr_args,
            **kwargs,
        )

    @classmethod
    def from_json(cls, file_path: str | Path) -> "Config":
        """Load a Config from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert the Config to a dictionary."""
        result = asdict(self)
        return result

    def to_json(self, file_path: str | Path) -> None:
        """Save the Config to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: str | Path) -> Config:
    """
    Load a configuration from a JSON file.

    Args:
        config_path: Path to the JSON config file

    Returns:
        Config object
    """
    return Config.from_json(config_path)

"""Configuration system for LUT optimization."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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


@dataclass
class RepresentationArgs:
    """Representation-specific arguments."""

    lut_size: int = 16


@dataclass
class Config:
    """Configuration for LUT optimization."""

    representation: RepresentationType = "lut"
    image_text_loss_type: ImageTextLossType = "clip"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    representation_args: RepresentationArgs = field(default_factory=RepresentationArgs)
    steps: int = 500
    learning_rate: float = 0.005
    batch_size: int = 4

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create a Config from a dictionary."""
        loss_weights_data = data.get("loss_weights", {})
        loss_weights = LossWeights(
            image_text=loss_weights_data.get("image_text", 1.0),
            image_smoothness=loss_weights_data.get("image_smoothness", 1.0),
            image_regularization=loss_weights_data.get("image_regularization", 1.0),
            black_preservation=loss_weights_data.get("black_preservation", 1.0),
            repr_smoothness=loss_weights_data.get("repr_smoothness", 1.0),
        )

        repr_args_data = data.get("representation_args", {})
        representation_args = RepresentationArgs(
            lut_size=repr_args_data.get("lut_size", 16),
        )

        return cls(
            representation=data.get("representation", "lut"),
            image_text_loss_type=data.get("image_text_loss_type", "clip"),
            loss_weights=loss_weights,
            representation_args=representation_args,
            steps=data.get("steps", 500),
            learning_rate=data.get("learning_rate", 0.005),
            batch_size=data.get("batch_size", 4),
        )

    @classmethod
    def from_json(cls, file_path: str | Path) -> "Config":
        """Load a Config from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert the Config to a dictionary."""
        return {
            "representation": self.representation,
            "image_text_loss_type": self.image_text_loss_type,
            "loss_weights": {
                "image_text": self.loss_weights.image_text,
                "image_smoothness": self.loss_weights.image_smoothness,
                "image_regularization": self.loss_weights.image_regularization,
                "black_preservation": self.loss_weights.black_preservation,
                "repr_smoothness": self.loss_weights.repr_smoothness,
            },
            "representation_args": {
                "lut_size": self.representation_args.lut_size,
            },
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }

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

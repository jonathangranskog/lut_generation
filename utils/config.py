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
        valid_fields = {f.name for f in fields(cls)}
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
        loss_weights_data = data.get("loss_weights", {})
        loss_weights = LossWeights.from_dict(loss_weights_data)
        repr_args = dict(data.get("representation_args", {}))

        config_fields = {f.name for f in fields(cls)}
        simple_fields = config_fields - {"loss_weights", "representation_args"}
        kwargs = {k: v for k, v in data.items() if k in simple_fields}

        return cls(
            loss_weights=loss_weights,
            representation_args=repr_args,
            **kwargs,
        )

    @classmethod
    def from_json(cls, file_path: str | Path) -> "Config":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, file_path: str | Path) -> None:
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: str | Path) -> Config:
    return Config.from_json(config_path)

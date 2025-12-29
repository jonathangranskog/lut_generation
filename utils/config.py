"""Configuration system for LUT optimization."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, get_args

RepresentationType = Literal["lut", "bw_lut"]
ImageTextLossType = Literal["clip", "gemma3_4b", "gemma3_12b", "gemma3_27b", "sds"]

VALID_REPRESENTATIONS = get_args(RepresentationType)
VALID_IMAGE_TEXT_LOSS_TYPES = get_args(ImageTextLossType)


class ConfigValidationError(ValueError):
    """Raised when config validation fails."""

    pass


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

    def validate(self) -> List[str]:
        errors = []
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (int, float)):
                errors.append(f"loss_weights.{f.name} must be a number, got {type(value).__name__}")
            elif value < 0:
                errors.append(f"loss_weights.{f.name} must be non-negative, got {value}")
        return errors


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

    def validate(self) -> None:
        errors = []

        if self.representation not in VALID_REPRESENTATIONS:
            errors.append(
                f"representation must be one of {VALID_REPRESENTATIONS}, got '{self.representation}'"
            )

        if self.image_text_loss_type not in VALID_IMAGE_TEXT_LOSS_TYPES:
            errors.append(
                f"image_text_loss_type must be one of {VALID_IMAGE_TEXT_LOSS_TYPES}, "
                f"got '{self.image_text_loss_type}'"
            )

        if not isinstance(self.steps, int):
            errors.append(f"steps must be an integer, got {type(self.steps).__name__}")
        elif self.steps <= 0:
            errors.append(f"steps must be positive, got {self.steps}")

        if not isinstance(self.learning_rate, (int, float)):
            errors.append(f"learning_rate must be a number, got {type(self.learning_rate).__name__}")
        elif self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")

        if not isinstance(self.batch_size, int):
            errors.append(f"batch_size must be an integer, got {type(self.batch_size).__name__}")
        elif self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")

        errors.extend(self.loss_weights.validate())

        if not isinstance(self.representation_args, dict):
            errors.append(
                f"representation_args must be a dict, got {type(self.representation_args).__name__}"
            )
        elif "lut_size" in self.representation_args:
            lut_size = self.representation_args["lut_size"]
            if not isinstance(lut_size, int):
                errors.append(
                    f"representation_args.lut_size must be an integer, got {type(lut_size).__name__}"
                )
            elif lut_size <= 0:
                errors.append(f"representation_args.lut_size must be positive, got {lut_size}")

        if errors:
            raise ConfigValidationError("Config validation failed:\n  - " + "\n  - ".join(errors))

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
    def from_json(cls, file_path: str | Path, validate: bool = True) -> "Config":
        with open(file_path, "r") as f:
            data = json.load(f)
        config = cls.from_dict(data)
        if validate:
            config.validate()
        return config

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, file_path: str | Path) -> None:
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: str | Path, validate: bool = True) -> Config:
    return Config.from_json(config_path, validate=validate)

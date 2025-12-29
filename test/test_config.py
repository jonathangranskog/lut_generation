"""Tests for the configuration system."""

import json
from pathlib import Path

import pytest

from utils.config import Config, LossWeights, load_config


class TestLossWeights:
    """Tests for LossWeights dataclass."""

    def test_default_values(self):
        """Test default loss weight values."""
        weights = LossWeights()
        assert weights.image_text == 1.0
        assert weights.image_smoothness == 1.0
        assert weights.image_regularization == 1.0
        assert weights.black_preservation == 1.0
        assert weights.repr_smoothness == 1.0

    def test_custom_values(self):
        """Test custom loss weight values."""
        weights = LossWeights(
            image_text=10.0,
            image_smoothness=0.5,
            image_regularization=2.0,
            black_preservation=0.1,
            repr_smoothness=0.8,
        )
        assert weights.image_text == 10.0
        assert weights.image_smoothness == 0.5
        assert weights.image_regularization == 2.0
        assert weights.black_preservation == 0.1
        assert weights.repr_smoothness == 0.8


class TestRepresentationArgs:
    """Tests for representation_args dict."""

    def test_default_values(self):
        """Test default representation_args is empty dict."""
        config = Config()
        assert config.representation_args == {}

    def test_custom_values(self):
        """Test custom representation argument values."""
        config = Config(representation_args={"lut_size": 32})
        assert config.representation_args.get("lut_size") == 32

    def test_arbitrary_params(self):
        """Test that representation_args can hold arbitrary parameters."""
        config = Config(representation_args={"lut_size": 16, "custom_param": "value", "nested": {"a": 1}})
        assert config.representation_args.get("lut_size") == 16
        assert config.representation_args.get("custom_param") == "value"
        assert config.representation_args.get("nested") == {"a": 1}


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = Config()
        assert config.representation == "lut"
        assert config.image_text_loss_type == "clip"
        assert config.steps == 500
        assert config.learning_rate == 0.005
        assert config.batch_size == 4
        assert config.loss_weights.image_text == 1.0
        assert config.representation_args == {}

    def test_custom_values(self):
        """Test custom config values."""
        config = Config(
            representation="bw_lut",
            image_text_loss_type="sds",
            loss_weights=LossWeights(image_text=10.0),
            representation_args={"lut_size": 32},
            steps=1000,
            learning_rate=0.01,
            batch_size=1,
        )
        assert config.representation == "bw_lut"
        assert config.image_text_loss_type == "sds"
        assert config.steps == 1000
        assert config.learning_rate == 0.01
        assert config.batch_size == 1
        assert config.loss_weights.image_text == 10.0
        assert config.representation_args.get("lut_size") == 32

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "representation": "bw_lut",
            "image_text_loss_type": "gemma3_12b",
            "loss_weights": {
                "image_text": 5.0,
                "image_smoothness": 0.5,
            },
            "representation_args": {
                "lut_size": 32,
            },
            "steps": 250,
            "learning_rate": 0.002,
            "batch_size": 2,
        }
        config = Config.from_dict(data)

        assert config.representation == "bw_lut"
        assert config.image_text_loss_type == "gemma3_12b"
        assert config.steps == 250
        assert config.learning_rate == 0.002
        assert config.batch_size == 2
        assert config.loss_weights.image_text == 5.0
        assert config.loss_weights.image_smoothness == 0.5
        # Defaults should be preserved for unspecified values
        assert config.loss_weights.image_regularization == 1.0
        assert config.representation_args.get("lut_size") == 32

    def test_from_dict_with_defaults(self):
        """Test creating config from empty dictionary uses defaults."""
        config = Config.from_dict({})
        assert config.representation == "lut"
        assert config.image_text_loss_type == "clip"
        assert config.steps == 500
        assert config.learning_rate == 0.005
        assert config.batch_size == 4

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(
            representation="bw_lut",
            image_text_loss_type="sds",
            loss_weights=LossWeights(image_text=10.0, image_smoothness=0.5),
            representation_args={"lut_size": 32},
            steps=1000,
            learning_rate=0.01,
            batch_size=1,
        )
        data = config.to_dict()

        assert data["representation"] == "bw_lut"
        assert data["image_text_loss_type"] == "sds"
        assert data["steps"] == 1000
        assert data["learning_rate"] == 0.01
        assert data["batch_size"] == 1
        assert data["loss_weights"]["image_text"] == 10.0
        assert data["loss_weights"]["image_smoothness"] == 0.5
        assert data["representation_args"]["lut_size"] == 32

    def test_roundtrip_dict(self):
        """Test config survives dict roundtrip."""
        original = Config(
            representation="bw_lut",
            image_text_loss_type="gemma3_4b",
            loss_weights=LossWeights(
                image_text=5.0,
                image_smoothness=0.3,
                image_regularization=0.5,
                black_preservation=0.2,
                repr_smoothness=0.1,
            ),
            representation_args={"lut_size": 8},
            steps=750,
            learning_rate=0.003,
            batch_size=2,
        )

        data = original.to_dict()
        restored = Config.from_dict(data)

        assert restored.representation == original.representation
        assert restored.image_text_loss_type == original.image_text_loss_type
        assert restored.steps == original.steps
        assert restored.learning_rate == original.learning_rate
        assert restored.batch_size == original.batch_size
        assert restored.loss_weights.image_text == original.loss_weights.image_text
        assert restored.loss_weights.image_smoothness == original.loss_weights.image_smoothness
        assert (
            restored.representation_args.get("lut_size")
            == original.representation_args.get("lut_size")
        )


class TestConfigIO:
    """Tests for Config file I/O."""

    def test_to_json_and_from_json(self, temp_dir):
        """Test saving and loading config from JSON file."""
        config_path = temp_dir / "test_config.json"

        original = Config(
            representation="bw_lut",
            image_text_loss_type="sds",
            loss_weights=LossWeights(image_text=10.0),
            representation_args={"lut_size": 32},
            steps=1000,
            learning_rate=0.01,
            batch_size=1,
        )

        # Save to file
        original.to_json(config_path)

        # Verify file exists and is valid JSON
        assert config_path.exists()
        with open(config_path) as f:
            data = json.load(f)
        assert "representation" in data

        # Load from file
        loaded = Config.from_json(config_path)

        assert loaded.representation == original.representation
        assert loaded.image_text_loss_type == original.image_text_loss_type
        assert loaded.steps == original.steps
        assert loaded.learning_rate == original.learning_rate
        assert loaded.batch_size == original.batch_size
        assert loaded.loss_weights.image_text == original.loss_weights.image_text
        assert loaded.representation_args.get("lut_size") == original.representation_args.get("lut_size")

    def test_load_config_function(self, temp_dir):
        """Test the load_config convenience function."""
        config_path = temp_dir / "test_config.json"

        # Create a config file manually
        data = {
            "representation": "lut",
            "image_text_loss_type": "clip",
            "steps": 300,
            "learning_rate": 0.008,
            "batch_size": 8,
        }
        with open(config_path, "w") as f:
            json.dump(data, f)

        config = load_config(config_path)

        assert config.representation == "lut"
        assert config.image_text_loss_type == "clip"
        assert config.steps == 300
        assert config.learning_rate == 0.008
        assert config.batch_size == 8


class TestDefaultConfigs:
    """Tests for the default config files."""

    @pytest.fixture
    def configs_dir(self):
        """Get the configs directory path."""
        return Path(__file__).parent.parent / "configs"

    def test_color_clip_config(self, configs_dir):
        """Test color_clip.json config."""
        config = load_config(configs_dir / "color_clip.json")
        assert config.representation == "lut"
        assert config.image_text_loss_type == "clip"
        assert config.batch_size == 4

    def test_bw_clip_config(self, configs_dir):
        """Test bw_clip.json config."""
        config = load_config(configs_dir / "bw_clip.json")
        assert config.representation == "bw_lut"
        assert config.image_text_loss_type == "clip"
        assert config.batch_size == 4

    def test_color_sds_config(self, configs_dir):
        """Test color_sds.json config."""
        config = load_config(configs_dir / "color_sds.json")
        assert config.representation == "lut"
        assert config.image_text_loss_type == "sds"
        assert config.batch_size == 1
        assert config.loss_weights.image_text == 10.0

    def test_color_gemma3_12b_config(self, configs_dir):
        """Test color_gemma3_12b.json config."""
        config = load_config(configs_dir / "color_gemma3_12b.json")
        assert config.representation == "lut"
        assert config.image_text_loss_type == "gemma3_12b"
        assert config.batch_size == 1

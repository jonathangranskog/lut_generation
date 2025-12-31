"""Integration tests for complete LUT optimization and inference workflows."""

import json
import subprocess

import pytest
import torch

from utils.io import read_cube_file
from utils.transforms import identity_lut


@pytest.fixture
def test_config_color_clip(temp_dir):
    """Create a test config for color CLIP with minimal steps."""
    config_path = temp_dir / "test_color_clip.json"
    config = {
        "representation": "lut",
        "image_text_loss_type": "clip",
        "loss_weights": {
            "image_text": 1.0,
            "image_smoothness": 1.0,
            "image_regularization": 1.0,
            "black_preservation": 1.0,
            "repr_smoothness": 1.0,
        },
        "representation_args": {"size": 16},
        "steps": 5,
        "learning_rate": 0.005,
        "batch_size": 4,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def test_config_bw_clip(temp_dir):
    """Create a test config for B&W CLIP with minimal steps."""
    config_path = temp_dir / "test_bw_clip.json"
    config = {
        "representation": "bw_lut",
        "image_text_loss_type": "clip",
        "loss_weights": {
            "image_text": 1.0,
            "image_smoothness": 1.0,
            "image_regularization": 1.0,
            "black_preservation": 1.0,
            "repr_smoothness": 1.0,
        },
        "representation_args": {"size": 16},
        "steps": 5,
        "learning_rate": 0.005,
        "batch_size": 4,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def test_config_mlp_clip(temp_dir):
    """Create a test config for MLP CLIP with minimal steps."""
    config_path = temp_dir / "test_mlp_clip.json"
    config = {
        "representation": "mlp",
        "image_text_loss_type": "clip",
        "loss_weights": {
            "image_text": 1.0,
            "image_smoothness": 1.0,
            "image_regularization": 1.0,
            "black_preservation": 1.0,
            "repr_smoothness": 0.0001,
        },
        "representation_args": {
            "num_layers": 2,
            "hidden_width": 64,
            "init_scale": 0.01,
        },
        "steps": 5,
        "learning_rate": 0.001,
        "batch_size": 4,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


@pytest.mark.integration
class TestOptimizeWorkflow:
    """Integration tests for the optimize command."""

    @pytest.mark.slow
    def test_optimize_clip_basic(
        self, sample_image_folder, temp_dir, test_config_color_clip
    ):
        """Test basic CLIP optimization workflow with minimal steps."""
        output_lut = temp_dir / "test_clip.cube"

        # Run optimize command with config
        cmd = [
            "python",
            "main.py",
            "optimize",
            "--prompt",
            "warm sunset",
            "--image-folder",
            str(sample_image_folder),
            "--config",
            str(test_config_color_clip),
            "--log-interval",
            "0",  # Disable logging
            "--output-path",
            str(output_lut),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check output LUT was created
        assert output_lut.exists(), "Output LUT file not created"

        # Verify LUT can be loaded
        lut_tensor, domain_min, domain_max = read_cube_file(str(output_lut))
        assert lut_tensor.shape == (16, 16, 16, 3)
        assert lut_tensor.min() >= 0.0
        assert lut_tensor.max() <= 1.0

    @pytest.mark.slow
    def test_optimize_mlp_basic(
        self, sample_image_folder, temp_dir, test_config_mlp_clip
    ):
        """Test basic MLP optimization workflow with minimal steps."""
        output_mlp = temp_dir / "test_mlp.pt"

        cmd = [
            "python",
            "main.py",
            "optimize",
            "--prompt",
            "warm sunset",
            "--image-folder",
            str(sample_image_folder),
            "--config",
            str(test_config_mlp_clip),
            "--log-interval",
            "0",
            "--output-path",
            str(output_mlp),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_mlp.exists(), "Output MLP file not created"

        # Verify MLP can be loaded and applied
        from representations import MLP

        mlp = MLP.read(str(output_mlp))
        assert mlp.num_layers == 2
        assert mlp.hidden_width == 64

        # Test that forward pass works
        test_input = torch.rand(1, 3, 32, 32)
        output = mlp(test_input)
        assert output.shape == test_input.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_optimize_validates_empty_folder(self, temp_dir, test_config_color_clip):
        """Test that optimize rejects empty image folders."""
        empty_folder = temp_dir / "empty"
        empty_folder.mkdir()

        cmd = [
            "python",
            "main.py",
            "optimize",
            "--prompt",
            "test",
            "--image-folder",
            str(empty_folder),
            "--config",
            str(test_config_color_clip),
            "--output-path",
            str(temp_dir / "test.cube"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with error about no images
        assert result.returncode != 0
        assert "No valid images found" in result.stderr

    def test_optimize_with_custom_log_dir(
        self, sample_image_folder, temp_dir, test_config_color_clip
    ):
        """Test optimize with custom log directory."""
        output_lut = temp_dir / "test.cube"
        log_dir = temp_dir / "custom_logs"

        # Create a config with 5 steps and log interval of 5
        config_path = temp_dir / "test_log_config.json"
        config = {
            "representation": "lut",
            "image_text_loss_type": "clip",
            "steps": 5,
            "learning_rate": 0.005,
            "batch_size": 4,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        cmd = [
            "python",
            "main.py",
            "optimize",
            "--prompt",
            "test",
            "--image-folder",
            str(sample_image_folder),
            "--config",
            str(config_path),
            "--log-interval",
            "5",
            "--log-dir",
            str(log_dir),
            "--output-path",
            str(output_lut),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert log_dir.exists(), "Custom log directory not created"
        # Check that logs were saved
        log_files = list(log_dir.glob("*.cube")) + list(log_dir.glob("*.png"))
        assert len(log_files) > 0, "No log files created"

    def test_optimize_grayscale(
        self, sample_image_folder, temp_dir, test_config_bw_clip
    ):
        """Test grayscale LUT optimization using bw_lut config."""
        output_lut = temp_dir / "grayscale.cube"

        cmd = [
            "python",
            "main.py",
            "optimize",
            "--prompt",
            "black and white",
            "--image-folder",
            str(sample_image_folder),
            "--config",
            str(test_config_bw_clip),
            "--log-interval",
            "0",
            "--output-path",
            str(output_lut),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_lut.exists()

        # Load and verify it's a valid grayscale LUT
        lut_tensor, _, _ = read_cube_file(str(output_lut))
        # After loading, all channels should be identical (grayscale)
        torch.testing.assert_close(lut_tensor[..., 0], lut_tensor[..., 1])
        torch.testing.assert_close(lut_tensor[..., 1], lut_tensor[..., 2])


@pytest.mark.integration
class TestInferWorkflow:
    """Integration tests for the infer command."""

    def test_infer_basic(self, sample_image_folder, temp_dir):
        """Test basic inference with identity LUT."""
        # Create an identity LUT
        lut_file = temp_dir / "identity.cube"
        from utils.io import write_cube_file

        identity = identity_lut(resolution=16)
        write_cube_file(str(lut_file), identity, title="Test Identity LUT")

        # Get a test image
        test_image = next(sample_image_folder.glob("*.jpg"))
        output_image = temp_dir / "output.png"

        cmd = [
            "python",
            "main.py",
            "infer",
            str(lut_file),
            str(test_image),
            "--output-path",
            str(output_image),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"Infer failed: {result.stderr}"
        assert output_image.exists(), "Output image not created"

    def test_infer_missing_lut_file(self, sample_image_folder, temp_dir):
        """Test infer with missing LUT file."""
        test_image = next(sample_image_folder.glob("*.jpg"))

        cmd = [
            "python",
            "main.py",
            "infer",
            "nonexistent.cube",
            str(test_image),
            "--output-path",
            str(temp_dir / "output.png"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with file not found error
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_infer_missing_image_file(self, temp_dir):
        """Test infer with missing image file."""
        # Create a dummy LUT
        lut_file = temp_dir / "test.cube"
        from utils.io import write_cube_file

        write_cube_file(str(lut_file), identity_lut(resolution=16))

        cmd = [
            "python",
            "main.py",
            "infer",
            str(lut_file),
            "nonexistent.jpg",
            "--output-path",
            str(temp_dir / "output.png"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with file not found error
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end pipeline tests combining optimize and infer."""

    @pytest.mark.slow
    def test_optimize_then_infer(
        self, sample_image_folder, temp_dir, test_config_color_clip
    ):
        """Test complete pipeline: optimize a LUT, then use it for inference."""
        lut_file = temp_dir / "pipeline.cube"
        output_image = temp_dir / "result.png"
        test_image = next(sample_image_folder.glob("*.jpg"))

        # Step 1: Optimize a LUT using config
        optimize_cmd = [
            "python",
            "main.py",
            "optimize",
            "--prompt",
            "warm golden hour",
            "--image-folder",
            str(sample_image_folder),
            "--config",
            str(test_config_color_clip),
            "--log-interval",
            "0",
            "--output-path",
            str(lut_file),
        ]

        result = subprocess.run(optimize_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Optimize failed: {result.stderr}"
        assert lut_file.exists()

        # Step 2: Apply the LUT to an image
        infer_cmd = [
            "python",
            "main.py",
            "infer",
            str(lut_file),
            str(test_image),
            "--output-path",
            str(output_image),
        ]

        result = subprocess.run(infer_cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Infer failed: {result.stderr}"
        assert output_image.exists()

        # Verify the output image is valid
        from PIL import Image

        img = Image.open(output_image)
        assert img.size[0] > 0 and img.size[1] > 0

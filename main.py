"""
Main LUT optimization and inference script.

This script allows you to optimize a LUT given a small dataset of images and a prompt.

It can use either CLIP, Score Distillation Sampling or VLMs to optimize the LUT.

Using the `infer` command, this script will apply a LUT to an image.
"""

import logging
import os
import random
import re
from pathlib import Path

import torch
import typer
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from models.clip import CLIPLoss
from models.sds import SDSLoss
from models.vlm import VLMLoss
from representations import LUT, BWLUT
from utils import (
    CLIP_IMAGE_SIZE,
    DEEPFLOYD_IMAGE_SIZE,
    VLM_IMAGE_SIZE,
    Config,
    ImageDataset,
    compute_losses,
    get_device,
    load_config,
    load_image_as_tensor,
    save_tensor_as_image,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for CLI output
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def sanitize_prompt_for_filename(prompt: str) -> str:
    """
    Convert a prompt to a safe folder name.

    Replaces spaces with underscores and removes problematic characters.

    Args:
        prompt: The text prompt

    Returns:
        Sanitized string safe for use as a folder name
    """
    # Replace spaces with underscores
    sanitized = prompt.replace(" ", "_")

    # Remove or replace problematic characters
    # Keep alphanumeric, underscores, hyphens, and periods
    sanitized = re.sub(r"[^\w\-.]", "", sanitized)

    # Trim to reasonable length (max 100 chars)
    sanitized = sanitized[:100]

    # Remove leading/trailing underscores or periods
    sanitized = sanitized.strip("_.")

    return sanitized or "untitled"


def format_loss_log(
    step: int,
    total_loss: torch.Tensor,
    loss_components: dict,
    image_text_weight: float,
    image_smoothness: float,
    image_regularization: float,
    black_preservation: float,
    repr_smoothness: float,
) -> str:
    """Format a detailed loss log message."""
    log_msg = f"Step {step}: Loss = {total_loss.item():.4f} (Primary: {(image_text_weight * loss_components['primary']).item():.4f}"

    if image_smoothness > 0 and "img_smooth" in loss_components:
        log_msg += (
            f", Smooth: {(image_smoothness * loss_components['img_smooth']).item():.4f}"
        )

    if image_regularization > 0 and "img_reg" in loss_components:
        log_msg += (
            f", Reg: {(image_regularization * loss_components['img_reg']).item():.4f}"
        )

    if black_preservation > 0 and "black" in loss_components:
        log_msg += (
            f", Black: {(black_preservation * loss_components['black']).item():.4f}"
        )

    if repr_smoothness > 0 and "repr_smooth" in loss_components:
        log_msg += f", Repr Smooth: {(repr_smoothness * loss_components['repr_smooth']).item():.4f}"

    log_msg += ")"
    return log_msg


def save_training_checkpoint(
    step: int,
    representation,
    sample_images: list[torch.Tensor],
    output_dir: Path,
) -> None:
    """
    Save representation and sample transformed images for training monitoring.

    Args:
        step: Current training step
        representation: Representation instance (LUT or BWLUT)
        sample_images: List of sample image tensors, each (C, H, W) in [0, 1] range
        output_dir: Directory to save checkpoints
    """
    # Create checkpoint directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save representation
    lut_path = output_dir / f"lut_step_{step:05d}.cube"
    representation.write(str(lut_path), title=f"Training Step {step}")

    # Apply representation to each sample image and save
    with torch.no_grad():
        for idx, sample_image in enumerate(sample_images):
            # Apply representation (non-training mode, with postprocessing)
            transformed = representation(sample_image, training=False)

            # Concatenate original and transformed images side-by-side
            concatenated = torch.cat([sample_image, transformed], dim=-1)

            # Save using helper function
            img_path = output_dir / f"image_{idx}_step_{step:05d}.png"
            save_tensor_as_image(concatenated, str(img_path))


@app.command()
def optimize(
    prompt: Annotated[str, typer.Option(help="The prompt to optimize the LUT for.")],
    image_folder: Annotated[str, typer.Option(help="Dataset folder of images")],
    config: Annotated[
        str,
        typer.Option(help="Path to JSON config file"),
    ] = "configs/color_clip.json",
    log_interval: int = 50,
    verbose: bool = False,
    output_path: str = "lut.cube",
    test_image: list[str] | None = None,
    log_dir: str | None = None,
) -> None:
    """
    Optimize a LUT given a small dataset of images and a prompt.

    Uses a JSON config file to specify representation type, loss weights, training parameters, etc.
    Default configs are provided in the configs/ directory:
    - configs/color_clip.json: Color LUT with CLIP
    - configs/bw_clip.json: Black & white LUT with CLIP
    - configs/color_sds.json: Color LUT with SDS
    - configs/color_gemma3_12b.json: Color LUT with Gemma 3 12B

    Every log_interval steps, saves LUT and sample image to training logs directory.
    Log directory can be customized via --log-dir (default: tmp/training_logs/<sanitized_prompt>/).

    Note: Gemma 3 models use comparison mode by default, evaluating transformations by comparing
    original and transformed images for more context-aware color grading.
    """
    # Load config
    cfg = load_config(config)
    logger.info(f"Loaded config from {config}")

    # Extract config values
    model_type = cfg.image_text_loss_type
    steps = cfg.steps
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    image_text_weight = cfg.loss_weights.image_text
    image_smoothness = cfg.loss_weights.image_smoothness
    image_regularization = cfg.loss_weights.image_regularization
    black_preservation = cfg.loss_weights.black_preservation
    repr_smoothness = cfg.loss_weights.repr_smoothness

    # Select device (MPS doesn't support grid_sampler_3d_backward)
    device = get_device(allow_mps=False)
    logger.info(f"Using device: {device}")

    # Select image size based on model type
    if model_type == "clip":
        image_size = CLIP_IMAGE_SIZE
    elif model_type == "sds":
        image_size = DEEPFLOYD_IMAGE_SIZE
    else:
        # Gemma 3 models use VLM image size
        image_size = VLM_IMAGE_SIZE

    # Create dataset
    dataset = ImageDataset(image_folder, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(
        f"Loaded {len(dataset)} images from {image_folder} (crop size: {image_size})"
    )

    if test_image is None or len(test_image) == 0:
        # Pick a random sample image for logging (keep on CPU initially)
        sample_idx = random.randint(0, len(dataset) - 1)
        sample_images_cpu = [dataset[sample_idx]]  # List of (C, H, W) tensors
        logger.info(f"Selected sample image index {sample_idx} for logging")
    else:
        sample_images_cpu = [load_image_as_tensor(img_path) for img_path in test_image]
        logger.info(f"Loaded {len(test_image)} test images:")
        for img_path in test_image:
            logger.info(f"  - {img_path}")

    # Create loss function
    if model_type == "clip":
        loss_fn = CLIPLoss(prompt, device=device)
    elif model_type in ["gemma3_4b", "gemma3_12b", "gemma3_27b"]:
        # VLM models use comparison mode to evaluate transformations
        loss_fn = VLMLoss(prompt, model_name=model_type, device=device)
    elif model_type == "sds":
        loss_fn = SDSLoss(prompt, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create representation (trainable!)
    if cfg.representation == "bw_lut":
        representation = BWLUT(**cfg.representation_args).to(device)
    else:
        representation = LUT(**cfg.representation_args).to(device)

    # Create optimizer for representation parameters
    optimizer = Adam(representation.parameters(), lr=learning_rate)

    # Training loop
    step = 0
    stop = False
    pbar = tqdm(total=steps, desc="Optimizing LUT") if not verbose else None

    # Create log directory
    if log_dir is None:
        # Default: tmp/training_logs/<sanitized_prompt>/
        prompt_folder = sanitize_prompt_for_filename(prompt)
        log_dir_path = Path("tmp/training_logs") / prompt_folder
    else:
        log_dir_path = Path(log_dir)

    if log_interval > 0:
        logger.info(f"Training logs will be saved to: {log_dir_path}/\n")

        # Save the original (untransformed) sample images
        log_dir_path.mkdir(parents=True, exist_ok=True)
        for idx, sample_image_cpu in enumerate(sample_images_cpu):
            img_path = log_dir_path / f"image_{idx}_original.png"
            save_tensor_as_image(sample_image_cpu, str(img_path))
            logger.info(f"Saved original sample image {idx} to {img_path}")
        logger.info("\n")

    while not stop:
        for images in dataloader:
            images = images.to(device)

            # Apply representation to images (training mode, no postprocessing)
            transformed_images = representation(images, training=True)

            # Compute all losses
            loss, loss_components = compute_losses(
                loss_fn,
                transformed_images,
                images,
                representation,
                image_text_weight,
                image_smoothness,
                image_regularization,
                black_preservation,
                repr_smoothness,
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent extreme updates (reduces banding)
            torch.nn.utils.clip_grad_norm_(representation.parameters(), max_norm=1.0)

            optimizer.step()

            # Clamp representation to valid range
            with torch.no_grad():
                representation.clamp()

            step += 1

            # Save checkpoint every log_interval steps
            if log_interval > 0 and step % log_interval == 0:
                sample_images_device = [img.to(device) for img in sample_images_cpu]
                save_training_checkpoint(
                    step,
                    representation,
                    sample_images_device,
                    log_dir_path,
                )
                if verbose:
                    logger.info(f"  → Saved checkpoint to {log_dir_path}/")

            # Logging
            if verbose and step % 10 == 0:
                log_msg = format_loss_log(
                    step,
                    loss,
                    loss_components,
                    image_text_weight,
                    image_smoothness,
                    image_regularization,
                    black_preservation,
                    repr_smoothness,
                )
                logger.info(log_msg)
            elif pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if step >= steps:
                stop = True
                break

    if pbar is not None:
        pbar.close()

    # Save final checkpoint
    if log_interval > 0:
        sample_images_device = [img.to(device) for img in sample_images_cpu]
        save_training_checkpoint(
            step,
            representation,
            sample_images_device,
            log_dir_path,
        )
        logger.info(f"Saved final checkpoint to {log_dir_path}/")

    logger.info(f"\nOptimization complete! Final loss: {loss.item():.4f}")

    # Save representation
    representation.write(output_path, title=f"{model_type.upper()}: {prompt}")
    logger.info(f"LUT saved to {output_path}")


@app.command()
def infer(
    ckpt_path: str,
    image: str,
    output_path: str = "output.png",
) -> None:
    """
    Apply a LUT to an image.
    """

    # some error checking
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"LUT file not found: {ckpt_path}")
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image file not found: {image}")

    # Select device (MPS is fine for inference)
    device = get_device(allow_mps=True)

    # Load representation from file
    representation = LUT.read(ckpt_path)
    representation = representation.to(device)

    # Load and prepare image
    image_tensor = load_image_as_tensor(image)
    image_tensor = image_tensor.to(device)

    # Apply representation (non-training mode, with postprocessing)
    with torch.no_grad():
        image_tensor = representation(image_tensor, training=False)

    # Save the transformed image
    save_tensor_as_image(image_tensor, output_path)


if __name__ == "__main__":
    app()

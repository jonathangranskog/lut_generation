"""
Main LUT optimization and inference script.

This script allows you to optimize a LUT given a small dataset of images and a prompt.

It can use either CLIP, Score Distillation Sampling or VLMs to optimize the LUT.

Using the `infer` command, this script will apply a LUT to an image.
"""

import os
import random
import re
from pathlib import Path
from typing import Literal

import torch
import typer
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from models.clip import CLIPLoss
from models.sds import SDSLoss
from models.vlm import VLMLoss
from utils import (
    CLIP_IMAGE_SIZE,
    DEEPFLOYD_IMAGE_SIZE,
    VLM_IMAGE_SIZE,
    ImageDataset,
    apply_lut,
    compute_losses,
    get_device,
    identity_lut,
    load_image_as_tensor,
    postprocess_lut,
    read_cube_file,
    save_tensor_as_image,
    tensor_to_pil,
    write_cube_file,
)

ModelType = Literal["clip", "gemma3_4b", "gemma3_12b", "gemma3_27b", "sds"]

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
    lut_smoothness: float,
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

    if lut_smoothness > 0 and "lut_smooth" in loss_components:
        log_msg += f", LUT Smooth: {(lut_smoothness * loss_components['lut_smooth']).item():.4f}"

    log_msg += ")"
    return log_msg


def save_training_checkpoint(
    step: int,
    lut_tensor: torch.Tensor,
    sample_images: list[torch.Tensor],
    output_dir: Path,
    grayscale: bool = False,
) -> None:
    """
    Save LUT and sample transformed images for training monitoring.

    Args:
        step: Current training step
        lut_tensor: Current LUT state
        sample_images: List of sample image tensors, each (C, H, W) in [0, 1] range
        output_dir: Directory to save checkpoints
        grayscale: Whether to save LUT with replicated grayscale channel
    """
    # Create checkpoint directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LUT
    lut_path = output_dir / f"lut_step_{step:05d}.cube"
    write_cube_file(
        str(lut_path),
        lut_tensor.detach().cpu(),
        domain_min=[0.0, 0.0, 0.0],
        domain_max=[1.0, 1.0, 1.0],
        title=f"Training Step {step}",
        grayscale=grayscale,
    )

    # Apply LUT to each sample image and save
    with torch.no_grad():
        for idx, sample_image in enumerate(sample_images):
            # Apply LUT
            transformed = apply_lut(sample_image, lut_tensor)

            # Concatenate original and transformed images side-by-side
            concatenated = torch.cat([sample_image, transformed], dim=-1)

            # Save using helper function
            img_path = output_dir / f"image_{idx}_step_{step:05d}.png"
            save_tensor_as_image(concatenated, str(img_path))


@app.command()
def optimize(
    prompt: Annotated[str, typer.Option(help="The prompt to optimize the LUT for.")],
    image_folder: Annotated[str, typer.Option(help="Dataset folder of images")],
    model_type: Annotated[
        ModelType,
        typer.Option(help="Model type: clip, gemma3_4b, gemma3_12b, gemma3_27b, or sds"),
    ] = "clip",
    lut_size: int = 16,
    steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 5e-3,
    image_text_weight: float = 1.0,
    image_smoothness: float = 1.0,
    image_regularization: float = 1.0,
    black_preservation: float = 1.0,
    lut_smoothness: float = 1.0,
    log_interval: int = 50,
    verbose: bool = False,
    output_path: str = "lut.cube",
    test_image: list[str] | None = None,
    grayscale: bool = False,
    # SDS-specific options
    sds_guidance_scale: float = 20.0,
    sds_min_timestep: int = 20,
    sds_max_timestep: int = 980,
) -> None:
    """
    Optimize a LUT given a small dataset of images and a prompt.

    Image text weight controls the contribution of the main image-text alignment loss (CLIP or Gemma 3).
    Image smoothness penalizes banding and discontinuities in output images.
    Image regularization keeps output images close to input images (subtle changes).
    Black preservation prevents faded/lifted blacks (maintains deep shadows).
    Every log_interval steps, saves LUT and sample image to tmp/training_logs/.
    Grayscale optimizes a black-and-white LUT (single channel) that outputs same intensity for RGB.

    Note: Gemma 3 models use comparison mode by default, evaluating transformations by comparing
    original and transformed images for more context-aware color grading.

    SDS-specific options (only used when model_type='sds'):
    - sds_guidance_scale: Classifier-free guidance scale (default 20.0)
    - sds_min_timestep: Minimum diffusion timestep (default 20)
    - sds_max_timestep: Maximum diffusion timestep (default 980)
    """
    # Select device (MPS doesn't support grid_sampler_3d_backward)
    device = get_device(allow_mps=False)
    print(f"Using device: {device}")

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
    print(f"Loaded {len(dataset)} images from {image_folder} (crop size: {image_size})")

    if test_image is None or len(test_image) == 0:
        # Pick a random sample image for logging (keep on CPU initially)
        sample_idx = random.randint(0, len(dataset) - 1)
        sample_images_cpu = [dataset[sample_idx]]  # List of (C, H, W) tensors
        print(f"Selected sample image index {sample_idx} for logging")
    else:
        sample_images_cpu = [load_image_as_tensor(img_path) for img_path in test_image]
        print(f"Loaded {len(test_image)} test images:")
        for img_path in test_image:
            print(f"  - {img_path}")

    # Create loss function
    if model_type == "clip":
        loss_fn = CLIPLoss(prompt, device=device)
    elif model_type in ["gemma3_4b", "gemma3_12b", "gemma3_27b"]:
        # VLM models use comparison mode to evaluate transformations
        loss_fn = VLMLoss(prompt, model_name=model_type, device=device)
    elif model_type == "sds":
        loss_fn = SDSLoss(
            prompt,
            device=device,
            guidance_scale=sds_guidance_scale,
            min_timestep=sds_min_timestep,
            max_timestep=sds_max_timestep,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create LUT (trainable!)
    lut_tensor = identity_lut(lut_size, grayscale=grayscale).to(device)
    lut_tensor.requires_grad = True

    # Create optimizer for LUT parameters
    optimizer = Adam([lut_tensor], lr=learning_rate)

    # Training loop
    step = 0
    stop = False
    pbar = tqdm(total=step, desc="Optimizing LUT") if not verbose else None

    # Create log directory based on prompt
    prompt_folder = sanitize_prompt_for_filename(prompt)
    log_dir = Path("tmp/training_logs") / prompt_folder
    if log_interval > 0:
        print(f"Training logs will be saved to: {log_dir}/\n")

        # Save the original (untransformed) sample images
        log_dir.mkdir(parents=True, exist_ok=True)
        for idx, sample_image_cpu in enumerate(sample_images_cpu):
            img_path = log_dir / f"image_{idx}_original.png"
            save_tensor_as_image(sample_image_cpu, str(img_path))
            print(f"Saved original sample image {idx} to {img_path}")
        print()

    while not stop:
        for images in dataloader:
            images = images.to(device)

            # Apply LUT to images
            transformed_images = apply_lut(images, lut_tensor)

            # Compute all losses
            loss, loss_components = compute_losses(
                loss_fn,
                transformed_images,
                images,
                lut_tensor,
                image_text_weight,
                image_smoothness,
                image_regularization,
                black_preservation,
                lut_smoothness,
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent extreme updates (reduces banding)
            torch.nn.utils.clip_grad_norm_([lut_tensor], max_norm=1.0)

            optimizer.step()

            # Clamp LUT to valid range [0, 1]
            with torch.no_grad():
                lut_tensor.clamp_(0, 1)

            step += 1

            # Save checkpoint every log_interval steps
            if log_interval > 0 and step % log_interval == 0:
                sample_images_device = [img.to(device) for img in sample_images_cpu]
                save_training_checkpoint(
                    step,
                    postprocess_lut(lut_tensor),
                    sample_images_device,
                    log_dir,
                    grayscale,
                )
                if verbose:
                    print(f"  → Saved checkpoint to {log_dir}/")

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
                    lut_smoothness,
                )
                print(log_msg)
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
            step, postprocess_lut(lut_tensor), sample_images_device, log_dir, grayscale
        )
        print(f"Saved final checkpoint to {log_dir}/")

    print(f"\nOptimization complete! Final loss: {loss.item():.4f}")

    # Save LUT
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    write_cube_file(
        output_path,
        lut_tensor.detach().cpu(),
        domain_min,
        domain_max,
        title=f"CLIP: {prompt}",
        grayscale=grayscale,
    )
    print(f"LUT saved to {output_path}")


@app.command()
def infer(
    lut: str,
    image: str,
    output_path: str = "output.png",
) -> None:
    """
    Apply a LUT to an image.
    """

    # some error checking
    if not os.path.exists(lut):
        raise FileNotFoundError(f"LUT file not found: {lut}")
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image file not found: {image}")

    # Select device (MPS is fine for inference)
    device = get_device(allow_mps=True)

    # read lut file
    lut_tensor, domain_min, domain_max = read_cube_file(lut)

    # Load and prepare image
    image_tensor = load_image_as_tensor(image)
    image_tensor = image_tensor.to(device).unsqueeze(0)
    image_tensor = apply_lut(image_tensor, lut_tensor, domain_min, domain_max)
    image_tensor = image_tensor.squeeze(0)

    # Save the transformed image
    save_tensor_as_image(image_tensor, output_path)


if __name__ == "__main__":
    app()

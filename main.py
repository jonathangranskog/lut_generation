"""
Main LUT optimization and inference script.

This script allows you to optimize a LUT given a small dataset of images and a prompt.

It can use either CLIP, Score Distillation Sampling or VLMs to optimize the LUT.

Using the `infer` command, this script will apply a LUT to an image.
"""

import torch
import typer
import os
import numpy as np
import random
import re

from PIL import Image
from typing import Literal
from typing_extensions import Annotated
from pathlib import Path
from utils import (
    read_cube_file,
    write_cube_file,
    apply_lut,
    identity_lut,
    postprocess_lut,
    image_smoothness_loss,
    image_regularization_loss,
    black_level_preservation_loss,
    lut_smoothness_loss,
    get_device,
    ImageDataset,
)
from models.clip import CLIPLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

ModelType = Literal["clip"]

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


def save_training_checkpoint(
    step: int,
    lut_tensor: torch.Tensor,
    sample_image: torch.Tensor,
    output_dir: Path,
) -> None:
    """
    Save LUT and a sample transformed image for training monitoring.

    Args:
        step: Current training step
        lut_tensor: Current LUT state
        sample_image: Sample image tensor (C, H, W) in [0, 1] range
        output_dir: Directory to save checkpoints
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
    )

    # Apply LUT to sample image and save
    with torch.no_grad():
        # Apply LUT
        transformed = apply_lut(sample_image, lut_tensor)

        # Convert to numpy and save
        img_array = transformed.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        img_path = output_dir / f"image_step_{step:05d}.png"
        img.save(img_path)


@app.command()
def optimize(
    prompt: Annotated[str, typer.Option(help="The prompt to optimize the LUT for.")],
    image_folder: Annotated[str, typer.Option(help="Dataset folder of images")],
    model_type: Annotated[ModelType, typer.Option(help="Model (clip)")] = "clip",
    lut_size: int = 32,
    steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 5e-3,
    image_smoothness: float = 1.0,
    image_regularization: float = 1.0,
    black_preservation: float = 1.0,
    lut_smoothness: float = 1.0,
    log_interval: int = 50,
    verbose: bool = False,
    output_path: str = "lut.cube",
    test_image: str | None = None,
) -> None:
    """
    Optimize a LUT given a small dataset of images and a prompt.

    Image smoothness penalizes banding and discontinuities in output images.
    Image regularization keeps output images close to input images (subtle changes).
    Black preservation prevents faded/lifted blacks (maintains deep shadows).
    Every log_interval steps, saves LUT and sample image to tmp/training_logs/.
    """
    # Select device (MPS doesn't support grid_sampler_3d_backward)
    device = get_device(allow_mps=False)
    print(f"Using device: {device}")

    # Create dataset
    dataset = ImageDataset(image_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} images from {image_folder}")

    if test_image is None:
        # Pick a random sample image for logging (keep on CPU initially)
        sample_idx = random.randint(0, len(dataset) - 1)
        sample_image_cpu = dataset[sample_idx]  # (C, H, W)
        print(f"Selected sample image index {sample_idx} for logging")
    else:
        # just open the test image
        sample_image_cpu = Image.open(test_image)
        sample_image_cpu = sample_image_cpu.convert("RGB")
        sample_image_cpu = np.array(sample_image_cpu)
        sample_image_cpu = torch.from_numpy(sample_image_cpu).permute(2, 0, 1)
        sample_image_cpu = sample_image_cpu.float() / 255.0
        print(f"Loaded test image from {test_image}")

    # Create loss function
    if model_type == "clip":
        loss_fn = CLIPLoss(prompt, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create LUT (trainable!)
    lut_tensor = identity_lut(lut_size).to(device)
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

        # Save the original (untransformed) sample image
        log_dir.mkdir(parents=True, exist_ok=True)
        original_img = sample_image_cpu.permute(1, 2, 0).clamp(0, 1).numpy()
        original_img = (original_img * 255).astype(np.uint8)
        Image.fromarray(original_img).save(log_dir / "image_original.png")
        print(f"Saved original sample image to {log_dir}/image_original.png\n")

    while not stop:
        for images in dataloader:
            images = images.to(device)

            # Apply LUT to images
            transformed_images = apply_lut(images, lut_tensor)

            # Compute CLIP loss
            clip_loss = loss_fn(transformed_images)
            loss = clip_loss

            # Image-space losses (directly penalize artifacts in output images)
            img_smooth_loss = None
            if image_smoothness > 0:
                img_smooth_loss = image_smoothness_loss(transformed_images)
                loss = loss + image_smoothness * img_smooth_loss

            img_reg_loss = None
            if image_regularization > 0:
                img_reg_loss = image_regularization_loss(transformed_images, images)
                loss = loss + image_regularization * img_reg_loss

            black_loss = None
            if black_preservation > 0:
                black_loss = black_level_preservation_loss(transformed_images, images)
                loss = loss + black_preservation * black_loss

            if lut_smoothness > 0:
                lut_smooth_loss = lut_smoothness_loss(lut_tensor)
                loss = loss + lut_smoothness * lut_smooth_loss

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
                sample_image_device = sample_image_cpu.to(device)
                save_training_checkpoint(
                    step, postprocess_lut(lut_tensor), sample_image_device, log_dir
                )
                if verbose:
                    print(f"  → Saved checkpoint to {log_dir}/")

            # Logging
            if verbose and step % 10 == 0:
                log_msg = f"Step {step}: Loss = {loss.item():.4f} (CLIP: {clip_loss.item():.4f}"
                if image_smoothness > 0 and img_smooth_loss is not None:
                    log_msg += (
                        f", Smooth: {(image_smoothness * img_smooth_loss).item():.4f}"
                    )
                if image_regularization > 0 and img_reg_loss is not None:
                    log_msg += (
                        f", Reg: {(image_regularization * img_reg_loss).item():.4f}"
                    )
                if black_preservation > 0 and black_loss is not None:
                    log_msg += (
                        f", Black: {(black_preservation * black_loss).item():.4f}"
                    )
                if lut_smoothness > 0 and lut_smooth_loss is not None:
                    log_msg += (
                        f", LUT Smooth: {(lut_smoothness * lut_smooth_loss).item():.4f}"
                    )
                log_msg += ")"
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
        sample_image_device = sample_image_cpu.to(device)
        save_training_checkpoint(
            step, postprocess_lut(lut_tensor), sample_image_device, log_dir
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

    # do the rest
    image = Image.open(image)
    image = image.convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    image_tensor = image_tensor.float() / 255.0
    image_tensor = image_tensor.to(device).unsqueeze(0)
    image_tensor = apply_lut(image_tensor, lut_tensor, domain_min, domain_max)
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    image_tensor = (image_tensor * 255.0).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image_tensor)
    image.save(output_path)


if __name__ == "__main__":
    app()

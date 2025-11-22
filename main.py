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
from utils.lut import (
    read_cube_file,
    apply_lut,
    identity_lut,
    write_cube_file,
    lut_smoothness_loss,
)
from models.clip import CLIPLoss
from utils.dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import mse_loss
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
    lut_size: int = 16,
    num_steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 1e-2,
    regularization: float = 0.0,
    smoothness: float = 1e-4,
    log_interval: int = 50,
    verbose: bool = False,
    output_path: str = "lut.cube",
) -> None:
    """
    Optimize a LUT given a small dataset of images and a prompt.

    Regularization keeps LUT close to identity.
    Smoothness penalizes abrupt changes between adjacent LUT cells.
    Every log_interval steps, saves LUT and sample image to tmp/training_logs/.
    """
    # Select device
    # Note: MPS doesn't support grid_sampler_3d_backward, so use CUDA or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create dataset
    dataset = ImageDataset(image_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} images from {image_folder}")

    # Pick a random sample image for logging (keep on CPU initially)
    sample_idx = random.randint(0, len(dataset) - 1)
    sample_image_cpu = dataset[sample_idx]  # (C, H, W)
    print(f"Selected sample image index {sample_idx} for logging")

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
    steps = 0
    stop = False
    pbar = tqdm(total=num_steps, desc="Optimizing LUT") if not verbose else None

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

            # Add regularization to keep LUT close to identity
            reg_loss = None
            if regularization > 0:
                identity = identity_lut(lut_size).to(device)
                reg_loss = mse_loss(lut_tensor, identity)
                loss = loss + regularization * reg_loss

            # Add smoothness regularization to encourage smooth transitions
            smooth_loss = None
            if smoothness > 0:
                smooth_loss = lut_smoothness_loss(lut_tensor)
                loss = loss + smoothness * smooth_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp LUT to valid range [0, 1]
            with torch.no_grad():
                lut_tensor.clamp_(0, 1)

            steps += 1

            # Save checkpoint every log_interval steps
            if log_interval > 0 and steps % log_interval == 0:
                sample_image_device = sample_image_cpu.to(device)
                save_training_checkpoint(
                    steps, lut_tensor, sample_image_device, log_dir
                )
                if verbose:
                    print(f"  → Saved checkpoint to {log_dir}/")

            # Logging
            if verbose and steps % 10 == 0:
                log_msg = f"Step {steps}: Loss = {loss.item():.4f} (CLIP: {clip_loss.item():.4f}"
                if regularization > 0 and reg_loss is not None:
                    log_msg += f", Reg: {(regularization * reg_loss).item():.4f}"
                if smoothness > 0 and smooth_loss is not None:
                    log_msg += f", Smooth: {(smoothness * smooth_loss).item():.4f}"
                log_msg += ")"
                print(log_msg)
            elif pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if steps >= num_steps:
                stop = True
                break

    if pbar is not None:
        pbar.close()

    # Save final checkpoint
    if log_interval > 0:
        sample_image_device = sample_image_cpu.to(device)
        save_training_checkpoint(steps, lut_tensor, sample_image_device, log_dir)
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

    # select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

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

from PIL import Image
from typing import Literal
from typing_extensions import Annotated
from lut import read_cube_file, apply_lut, identity_lut

ModelType = Literal["clip", "sds", "vlm"]

app = typer.Typer()


@app.command()
def optimize(
    prompt: Annotated[str, typer.Option(help="The prompt to optimize the LUT for.")],
    image_folder: Annotated[str, typer.Option(help="Dataset folder of images")],
    model: Annotated[ModelType, typer.Option(help="Model (clip, sds, vlm)")] = "clip",
    lut_size: int = 16,
    num_steps: int = 500,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    verbose: bool = False,
    output_path: str = "lut.cube",
) -> None:
    """
    Optimize a LUT given a small dataset of images and a prompt.
    """
    pass


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

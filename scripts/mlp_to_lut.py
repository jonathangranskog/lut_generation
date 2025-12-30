#!/usr/bin/env python3
"""
Convert an MLP checkpoint to a LUT file.

This script loads a trained MLP representation and converts it to a standard
.cube LUT file by evaluating the MLP at each voxel position of an identity LUT.
"""

import sys
from pathlib import Path

import torch
import typer
from typing_extensions import Annotated

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from representations import MLP, LUT

app = typer.Typer()


@app.command()
def convert(
    mlp_path: Annotated[str, typer.Argument(help="Path to MLP checkpoint (.pt file)")],
    output_path: Annotated[str, typer.Argument(help="Output path for LUT (.cube file)")],
    lut_size: Annotated[int, typer.Option(help="LUT resolution")] = 32,
) -> None:
    """
    Convert an MLP checkpoint to a LUT file.

    Examples:
        # Convert MLP to 32x32x32 LUT (default)
        python scripts/mlp_to_lut.py model.pt output.cube

        # Convert MLP to 64x64x64 LUT for higher precision
        python scripts/mlp_to_lut.py model.pt output.cube --lut-size 64
    """
    # Validate inputs
    if not Path(mlp_path).exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {mlp_path}")

    if not output_path.endswith(".cube"):
        output_path = output_path + ".cube"

    print(f"Loading MLP from: {mlp_path}")
    mlp = MLP.read(mlp_path)
    mlp.eval()

    print(f"Creating identity LUT with size {lut_size}x{lut_size}x{lut_size}")
    lut = LUT(size=lut_size, initialize_identity=True)

    # Get the identity LUT tensor (size, size, size, 3)
    identity_tensor = lut.lut_tensor.data.clone()

    # Flatten to (size^3, 3) for MLP processing
    flat_rgb = identity_tensor.reshape(-1, 3)

    print(f"Evaluating MLP on {flat_rgb.shape[0]} color samples...")
    with torch.no_grad():
        # Process through MLP
        transformed_rgb = mlp.network(flat_rgb) + flat_rgb  # ResNet-style

    # Reshape back to (size, size, size, 3)
    transformed_lut = transformed_rgb.reshape(lut_size, lut_size, lut_size, 3)

    # Clamp to valid range
    transformed_lut = transformed_lut.clamp(0, 1)

    # Update LUT tensor and save
    lut.lut_tensor.data = transformed_lut
    lut.write(output_path, title=f"MLP converted to LUT (size={lut_size})")

    print(f"Saved LUT to: {output_path}")


if __name__ == "__main__":
    app()

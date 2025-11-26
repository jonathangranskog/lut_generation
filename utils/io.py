import re
from typing import Tuple

import torch


def read_cube_file(lut_path: str) -> Tuple[torch.Tensor, list, list]:
    """
    Load a .cube LUT file and return as a PyTorch tensor

    Args:
        lut_path: Path to the .cube file

    Returns:
        Tuple of (lut_tensor, domain_min, domain_max)
        lut_tensor: 3D tensor of shape (size, size, size, 3) for RGB lookup
        domain_min: List of 3 floats for minimum domain values
        domain_max: List of 3 floats for maximum domain values
    """
    with open(lut_path, "r") as f:
        lines = f.readlines()

    # Parse header information
    lut_size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]

    # Find where the actual LUT data starts
    data_start_idx = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip comments and empty lines
        if line.startswith("#") or not line:
            continue

        # Parse LUT_3D_SIZE
        if line.startswith("LUT_3D_SIZE"):
            lut_size = int(line.split()[1])

        # Parse DOMAIN_MIN (optional)
        elif line.startswith("DOMAIN_MIN"):
            domain_min = [float(x) for x in line.split()[1:4]]

        # Parse DOMAIN_MAX (optional)
        elif line.startswith("DOMAIN_MAX"):
            domain_max = [float(x) for x in line.split()[1:4]]

        # Check if this line looks like RGB data (3 floats)
        elif re.match(r"^[\d\.\-\s]+$", line) and len(line.split()) == 3:
            data_start_idx = i
            break

    if lut_size is None:
        raise ValueError("LUT_3D_SIZE not found in cube file")

    # Parse the RGB data
    lut_data = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith("#"):
            try:
                r, g, b = map(float, line.split())
                lut_data.append([r, g, b])
            except ValueError:
                continue  # Skip invalid lines

    # Verify we have the right amount of data
    expected_entries = lut_size**3
    if len(lut_data) != expected_entries:
        raise ValueError(f"Expected {expected_entries} entries, got {len(lut_data)}")

    # Convert to tensor and reshape
    lut_tensor = torch.tensor(lut_data, dtype=torch.float32)

    # Reshape to 3D cube indexed as [B][G][R] (blue varies slowest, red varies fastest)
    # Output values remain in RGB order as stored in the file
    lut_cube = lut_tensor.reshape(lut_size, lut_size, lut_size, 3)

    return lut_cube, domain_min, domain_max


def write_cube_file(
    lut_path: str,
    lut_tensor: torch.Tensor,
    domain_min: list = [0.0, 0.0, 0.0],
    domain_max: list = [1.0, 1.0, 1.0],
    title: str = "Generated LUT",
) -> None:
    """
    Save a PyTorch LUT tensor to a .cube file

    Args:
        lut_path: Path where the .cube file will be saved
        lut_tensor: LUT tensor of shape (size, size, size, 3)
        domain_min: Minimum domain values (default [0.0, 0.0, 0.0])
        domain_max: Maximum domain values (default [1.0, 1.0, 1.0])
        title: Optional title for the LUT file
    """
    assert lut_tensor.ndim == 4, "LUT tensor must be 4D (size, size, size, 3)"
    assert lut_tensor.shape[-1] == 3, "LUT tensor must have 3 channels (RGB)"
    assert len(domain_min) == 3, "Domain min must be a 3-element list"
    assert len(domain_max) == 3, "Domain max must be a 3-element list"

    lut_size = lut_tensor.shape[0]

    # Verify cube shape
    assert lut_tensor.shape[0] == lut_tensor.shape[1] == lut_tensor.shape[2], (
        "LUT must be cubic (same size in all dimensions)"
    )

    with open(lut_path, "w") as f:
        # Write header
        f.write(f"# {title}\n")
        f.write("# Generated with PyTorch LUT tools\n")
        f.write(f"LUT_3D_SIZE {lut_size}\n")

        # Write domain info if not default
        if domain_min != [0.0, 0.0, 0.0]:
            f.write(
                f"DOMAIN_MIN {domain_min[0]:.6f} {domain_min[1]:.6f} {domain_min[2]:.6f}\n"
            )
        if domain_max != [1.0, 1.0, 1.0]:
            f.write(
                f"DOMAIN_MAX {domain_max[0]:.6f} {domain_max[1]:.6f} {domain_max[2]:.6f}\n"
            )

        f.write("\n")

        # Flatten LUT data in BGR indexing order (blue varies slowest, red fastest)
        # The tensor is already in [B][G][R] order from our format
        lut_data = lut_tensor.reshape(-1, 3).cpu()

        # Write RGB values, one per line
        for rgb in lut_data:
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

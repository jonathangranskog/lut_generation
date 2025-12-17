import re

import numpy as np
import torch
from PIL import Image


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = image_tensor.float() / 255.0
    return image_tensor


def read_cube_file(lut_path: str) -> tuple[torch.Tensor, list[float], list[float]]:
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

    # Parse the RGB data (cube format stores values as RGB on each line)
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

    # Convert to numpy array first for Fortran-style reshape
    lut_array = np.array(lut_data, dtype=np.float32)

    # .cube format uses column-major (Fortran-style) ordering where R varies fastest
    # and stores RGB values on each line
    # Reshape with Fortran order to get [R][G][B] spatial indexing with RGB values
    lut_cube_np = lut_array.reshape((lut_size, lut_size, lut_size, 3), order="F")

    # Convert to torch tensor - keep [R][G][B] indexing
    lut_cube = torch.from_numpy(lut_cube_np)

    return lut_cube, domain_min, domain_max


def write_cube_file(
    lut_path: str,
    lut_tensor: torch.Tensor,
    domain_min: list[float] | None = None,
    domain_max: list[float] | None = None,
    title: str = "Generated LUT",
    grayscale: bool = False,
) -> None:
    """
    Save a PyTorch LUT tensor to a .cube file

    Args:
        lut_path: Path where the .cube file will be saved
        lut_tensor: LUT tensor of shape (size, size, size, 3) or (size, size, size, 1) for grayscale
        domain_min: Minimum domain values (default [0.0, 0.0, 0.0])
        domain_max: Maximum domain values (default [1.0, 1.0, 1.0])
        title: Optional title for the LUT file
        grayscale: If True, replicates single-channel LUT to 3 channels (R=G=B) when saving
    """
    if domain_min is None:
        domain_min = [0.0, 0.0, 0.0]
    if domain_max is None:
        domain_max = [1.0, 1.0, 1.0]

    assert lut_tensor.ndim == 4, "LUT tensor must be 4D (size, size, size, C)"
    assert lut_tensor.shape[-1] in [1, 3], "LUT tensor must have 1 or 3 channels"
    assert len(domain_min) == 3, "Domain min must be a 3-element list"
    assert len(domain_max) == 3, "Domain max must be a 3-element list"

    # If grayscale and single-channel, replicate to 3 channels
    if grayscale and lut_tensor.shape[-1] == 1:
        lut_tensor = lut_tensor.repeat(1, 1, 1, 3)  # (size, size, size, 3)

    # Ensure we have 3 channels after potential replication
    assert lut_tensor.shape[-1] == 3, (
        "LUT tensor must have 3 channels after grayscale replication"
    )

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

        # Flatten LUT data using Fortran-style (column-major) ordering
        # The tensor has [R][G][B] spatial indexing with RGB values
        # Flatten with 'F' order so R varies fastest when writing
        lut_array = lut_tensor.cpu().numpy()
        lut_data = lut_array.reshape(-1, 3, order="F")

        # Write RGB values (cube format stores RGB on each line)
        for rgb in lut_data:
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

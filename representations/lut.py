"""LUT-based representations for color transformation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseRepresentation


class LUT(BaseRepresentation):
    """3D LUT representation with shape (size, size, size, 3)."""

    def __init__(self, size: int = 32, initialize_identity: bool = True):
        super().__init__()
        self.size = size
        self.domain_min = [0.0, 0.0, 0.0]
        self.domain_max = [1.0, 1.0, 1.0]

        if initialize_identity:
            lut_data = self._create_identity_lut()
        else:
            lut_data = torch.zeros(size, size, size, 3)

        self.lut_tensor = nn.Parameter(lut_data, requires_grad=True)

    def _create_identity_lut(self) -> torch.Tensor:
        coords = torch.linspace(0, 1, self.size)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
        return torch.stack([r, g, b], dim=-1)

    def smoothness_loss(self) -> torch.Tensor:
        smoothed_lut = self._downsample_upsample_3d(self.lut_tensor)
        return F.mse_loss(self.lut_tensor, smoothed_lut)

    def _downsample_upsample_3d(
        self, lut: torch.Tensor, scale_factor: float = 0.5
    ) -> torch.Tensor:
        original_shape = lut.shape
        num_channels = original_shape[-1]

        lut_reshaped = lut.permute(3, 0, 1, 2).view(
            1, num_channels, original_shape[0], original_shape[1], original_shape[2]
        )

        lut_downsampled = F.interpolate(
            lut_reshaped, scale_factor=scale_factor, mode="trilinear"
        )
        lut_upsampled = F.interpolate(
            lut_downsampled,
            size=(original_shape[0], original_shape[1], original_shape[2]),
            mode="trilinear",
        )

        lut_result = (
            lut_upsampled[0]
            .permute(1, 2, 3, 0)
            .view(original_shape[0], original_shape[1], original_shape[2], num_channels)
        )
        return lut_result

    def forward(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        if not training:
            lut_to_use = self.postprocess(self.lut_tensor)
        else:
            lut_to_use = self.lut_tensor

        return self._apply_lut(images, lut_to_use, self.domain_min, self.domain_max)

    def _apply_lut(
        self,
        image: torch.Tensor,
        lut_tensor: torch.Tensor,
        domain_min: list[float],
        domain_max: list[float],
    ) -> torch.Tensor:
        is_batched = image.ndim == 4

        # Normalize to (B, H, W, C) format
        if is_batched:
            if image.shape[1] == 3:  # (B, C, H, W)
                x = image.permute(0, 2, 3, 1)
                channels_first = True
            else:  # (B, H, W, C)
                x = image
                channels_first = False
        else:
            # Add batch dimension for single image
            if image.shape[0] == 3:  # (C, H, W)
                x = image.permute(1, 2, 0).unsqueeze(0)
                channels_first = True
            else:  # (H, W, C)
                x = image.unsqueeze(0)
                channels_first = False

        B, H, W, C = x.shape

        # Check if LUT is grayscale (single-channel)
        is_grayscale_lut = lut_tensor.shape[-1] == 1
        lut_channels = lut_tensor.shape[-1]

        # Apply domain scaling
        domain_min_t = torch.tensor(domain_min, device=x.device)
        domain_max_t = torch.tensor(domain_max, device=x.device)
        domain_scaled = (x - domain_min_t) / (domain_max_t - domain_min_t)

        # Clamp coordinates for LUT lookup
        clamped_coords = torch.clamp(domain_scaled, 0, 1)

        # Prepare for grid_sample: need (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
        lut = lut_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, R, G, B)
        lut = lut.expand(B, -1, -1, -1, -1)  # (B, C, R, G, B)

        # Flip RGB to BGR for correct sampling
        clamped_coords_bgr = clamped_coords.flip(-1)

        # Scale from [0, 1] to [-1, 1] for grid_sample
        coords = clamped_coords_bgr * 2.0 - 1.0

        # Reshape coordinates to (B, H*W, 1, 1, 3) for sampling
        coords = coords.view(B, H * W, 1, 1, 3)

        # Sample the LUT with trilinear interpolation
        lut_sampled = F.grid_sample(
            lut, coords, mode="bilinear", padding_mode="border", align_corners=False
        )

        # Reshape LUT output back to (B, H, W, C)
        lut_sampled = lut_sampled.view(B, lut_channels, H, W).permute(0, 2, 3, 1)

        # If grayscale LUT, replicate single channel to 3 channels
        if is_grayscale_lut:
            result = lut_sampled.repeat(1, 1, 1, 3)  # (B, H, W, 3)
        else:
            result = lut_sampled

        # Return in original format
        if not is_batched:
            result = result.squeeze(0)  # Remove batch dimension
            if channels_first:
                return result.permute(2, 0, 1)
            else:
                return result
        else:
            if channels_first:
                return result.permute(0, 3, 1, 2)
            else:
                return result

    @classmethod
    def read(cls, file_path: str) -> "LUT":
        import re
        import numpy as np

        with open(file_path, "r") as f:
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

        # Convert to numpy array and reshape with Fortran order
        lut_array = np.array(lut_data, dtype=np.float32)
        lut_cube_np = lut_array.reshape((lut_size, lut_size, lut_size, 3), order="F")

        # Create LUT instance
        lut = cls(size=lut_size, initialize_identity=False)
        lut.lut_tensor.data = torch.from_numpy(lut_cube_np)
        lut.domain_min = domain_min
        lut.domain_max = domain_max

        return lut

    @torch.no_grad()
    def write(self, file_path: str, title: str = "Generated LUT") -> None:
        # Apply postprocessing and get a smoothed copy
        lut_tensor = self.postprocess(self.lut_tensor).detach()

        assert lut_tensor.ndim == 4, "LUT tensor must be 4D (size, size, size, C)"
        assert lut_tensor.shape[-1] in [1, 3], "LUT tensor must have 1 or 3 channels"

        # Handle grayscale LUTs
        if lut_tensor.shape[-1] == 1:
            lut_tensor = lut_tensor.repeat(1, 1, 1, 3)  # Replicate to 3 channels

        lut_size = lut_tensor.shape[0]

        # Verify cube shape
        assert lut_tensor.shape[0] == lut_tensor.shape[1] == lut_tensor.shape[2], (
            "LUT must be cubic (same size in all dimensions)"
        )

        with open(file_path, "w") as f:
            # Write header
            f.write(f"# {title}\n")
            f.write("# Generated with PyTorch LUT representations\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n")

            # Write domain info if not default
            if self.domain_min != [0.0, 0.0, 0.0]:
                f.write(
                    f"DOMAIN_MIN {self.domain_min[0]:.6f} {self.domain_min[1]:.6f} {self.domain_min[2]:.6f}\n"
                )
            if self.domain_max != [1.0, 1.0, 1.0]:
                f.write(
                    f"DOMAIN_MAX {self.domain_max[0]:.6f} {self.domain_max[1]:.6f} {self.domain_max[2]:.6f}\n"
                )

            f.write("\n")

            # Flatten LUT data using Fortran-style (column-major) ordering
            lut_array = lut_tensor.cpu().numpy()
            lut_data = lut_array.reshape(-1, 3, order="F")

            # Write RGB values
            for rgb in lut_data:
                f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

    def postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._downsample_upsample_3d(tensor)

    def clamp(self) -> None:
        self.lut_tensor.clamp_(0, 1)


class BWLUT(LUT):
    """Black-and-white (grayscale) LUT representation with shape (size, size, size, 1)."""

    def __init__(self, size: int = 32, initialize_identity: bool = True):
        nn.Module.__init__(self)
        self.size = size
        self.domain_min = [0.0, 0.0, 0.0]
        self.domain_max = [1.0, 1.0, 1.0]

        # Create grayscale LUT tensor
        if initialize_identity:
            lut_data = self._create_identity_grayscale_lut()
        else:
            lut_data = torch.zeros(size, size, size, 1)

        self.lut_tensor = nn.Parameter(lut_data, requires_grad=True)

    def _create_identity_grayscale_lut(self) -> torch.Tensor:
        # Rec. 709 luma coefficients
        REC709_LUMA_R = 0.2126
        REC709_LUMA_G = 0.7152
        REC709_LUMA_B = 0.0722

        coords = torch.linspace(0, 1, self.size)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")

        # Compute luminance: Y = 0.2126*R + 0.7152*G + 0.0722*B
        luminance = REC709_LUMA_R * r + REC709_LUMA_G * g + REC709_LUMA_B * b
        identity_lut = luminance.unsqueeze(-1)  # (size, size, size, 1)

        return identity_lut

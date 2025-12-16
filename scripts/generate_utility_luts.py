#!/usr/bin/env python3
"""
Generate utility LUTs using fixed mathematical transformations.
These are technical adjustment LUTs for common color operations.
"""

import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.transforms.functional as TF
import typer
from PIL import Image
from typing_extensions import Annotated

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.transforms import identity_lut, apply_lut
from utils.io import write_cube_file, load_image_as_tensor

app = typer.Typer()


def apply_saturation(lut: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    """
    Adjust saturation using torchvision.

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        saturation_factor: Saturation multiplier (0.0 = grayscale, 1.0 = original, >1.0 = oversaturated)
    """
    # Reshape from (S, S, S, 3) to (3, S*S*S) for torchvision
    shape = lut.shape
    lut_flat = lut.reshape(-1, 3).permute(1, 0)  # (3, S*S*S)

    # Apply saturation adjustment
    lut_adjusted = TF.adjust_saturation(lut_flat.unsqueeze(1), saturation_factor)  # (3, 1, S*S*S)

    # Reshape back to (S, S, S, 3)
    lut_result = lut_adjusted.squeeze(1).permute(1, 0).reshape(shape)

    return lut_result.clamp(0, 1)


def apply_contrast(lut: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """
    Adjust contrast using torchvision.

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        contrast_factor: Contrast multiplier (0.0 = gray, 1.0 = original, >1.0 = more contrast)
    """
    shape = lut.shape
    lut_flat = lut.reshape(-1, 3).permute(1, 0)  # (3, S*S*S)

    lut_adjusted = TF.adjust_contrast(lut_flat.unsqueeze(1), contrast_factor)

    lut_result = lut_adjusted.squeeze(1).permute(1, 0).reshape(shape)

    return lut_result.clamp(0, 1)


def apply_brightness(lut: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """
    Adjust brightness using torchvision.

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        brightness_factor: Brightness multiplier (0.0 = black, 1.0 = original, >1.0 = brighter)
    """
    shape = lut.shape
    lut_flat = lut.reshape(-1, 3).permute(1, 0)  # (3, S*S*S)

    lut_adjusted = TF.adjust_brightness(lut_flat.unsqueeze(1), brightness_factor)

    lut_result = lut_adjusted.squeeze(1).permute(1, 0).reshape(shape)

    return lut_result.clamp(0, 1)


def apply_gamma(lut: torch.Tensor, gamma: float, gain: float = 1.0) -> torch.Tensor:
    """
    Apply gamma correction using torchvision.

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        gamma: Gamma value (< 1.0 = brighter, 1.0 = original, > 1.0 = darker)
        gain: Multiplier applied before gamma correction
    """
    shape = lut.shape
    lut_flat = lut.reshape(-1, 3).permute(1, 0)  # (3, S*S*S)

    lut_adjusted = TF.adjust_gamma(lut_flat.unsqueeze(1), gamma, gain)

    lut_result = lut_adjusted.squeeze(1).permute(1, 0).reshape(shape)

    return lut_result.clamp(0, 1)


def apply_hue(lut: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """
    Shift hue using torchvision.

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        hue_factor: Hue shift factor in range [-0.5, 0.5]
                    -0.5 = -180 degrees, 0.5 = 180 degrees
    """
    shape = lut.shape
    lut_flat = lut.reshape(-1, 3).permute(1, 0)  # (3, S*S*S)

    lut_adjusted = TF.adjust_hue(lut_flat.unsqueeze(1), hue_factor)

    lut_result = lut_adjusted.squeeze(1).permute(1, 0).reshape(shape)

    return lut_result.clamp(0, 1)


def apply_grayscale(lut: torch.Tensor) -> torch.Tensor:
    """
    Convert to grayscale using torchvision.

    Args:
        lut: Input LUT tensor (size, size, size, 3)
    """
    shape = lut.shape
    lut_flat = lut.reshape(-1, 3).permute(1, 0)  # (3, S*S*S)

    # Convert to grayscale and back to 3 channels
    lut_gray = TF.rgb_to_grayscale(lut_flat.unsqueeze(1), num_output_channels=3)

    lut_result = lut_gray.squeeze(1).permute(1, 0).reshape(shape)

    return lut_result.clamp(0, 1)


def apply_exposure(lut: torch.Tensor, exposure: float) -> torch.Tensor:
    """
    Adjust exposure (custom implementation using brightness).

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        exposure: Exposure adjustment in stops (-2.0 to 2.0)
    """
    # Convert stops to linear multiplier: 2^exposure
    multiplier = 2.0 ** exposure
    result = lut * multiplier

    return result.clamp(0, 1)


def apply_temperature(lut: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Apply color temperature shift (custom implementation).

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        temp: Temperature shift (-1.0 = cool/blue, 0.0 = neutral, 1.0 = warm/orange)
    """
    result = lut.clone()

    if temp > 0:  # Warm
        result[..., 0] = result[..., 0] + temp * 0.2  # Add red
        result[..., 2] = result[..., 2] - temp * 0.1  # Subtract blue
    else:  # Cool
        result[..., 0] = result[..., 0] + temp * 0.1  # Subtract red
        result[..., 2] = result[..., 2] - temp * 0.2  # Add blue

    return result.clamp(0, 1)


def apply_tint(lut: torch.Tensor, tint: float) -> torch.Tensor:
    """
    Apply green/magenta tint (custom implementation).

    Args:
        lut: Input LUT tensor (size, size, size, 3)
        tint: Tint shift (-1.0 = magenta, 0.0 = neutral, 1.0 = green)
    """
    result = lut.clone()

    if tint > 0:  # Green
        result[..., 1] = result[..., 1] + tint * 0.2
    else:  # Magenta (add red and blue, reduce green)
        result[..., 0] = result[..., 0] - tint * 0.1
        result[..., 1] = result[..., 1] + tint * 0.1
        result[..., 2] = result[..., 2] - tint * 0.1

    return result.clamp(0, 1)


def apply_lut_to_test_image(
    lut_tensor: torch.Tensor,
    test_image_path: Path,
    output_path: Path
) -> None:
    """Apply a LUT to a test image and save the result."""
    try:
        # Load test image
        image_tensor = load_image_as_tensor(str(test_image_path))

        # Apply LUT
        result = apply_lut(image_tensor, lut_tensor)

        # Convert back to PIL and save
        result_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        result_img = Image.fromarray(result_np)
        result_img.save(output_path)

        print(f"  Saved test image result: {output_path}")
    except Exception as e:
        print(f"  WARNING: Failed to apply LUT to test image: {e}")


def generate_utility_lut(
    name: str,
    transform_fn: Callable[[torch.Tensor], torch.Tensor],
    output_dir: Path,
    lut_size: int = 32,
    grayscale: bool = False,
    test_image: Optional[Path] = None,
    dry_run: bool = False
) -> None:
    """Generate and save a single utility LUT."""
    # Create identity LUT
    lut = identity_lut(resolution=lut_size, grayscale=grayscale)

    # Apply transformation
    lut_transformed = transform_fn(lut)

    # Save
    output_path = output_dir / f"{name}.cube"

    if dry_run:
        print(f"[DRY RUN] Would save: {output_path}")
    else:
        write_cube_file(
            str(output_path),
            lut_transformed,
            title=name.replace('_', ' ').title(),
            grayscale=grayscale
        )
        print(f"Generated: {output_path}")

        # Apply to test image if provided
        if test_image:
            test_output_path = output_path.with_suffix('.png')
            apply_lut_to_test_image(lut_transformed, test_image, test_output_path)


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option(help="Directory to save generated LUTs")] = Path("luts/utility"),
    lut_size: Annotated[int, typer.Option(help="LUT resolution")] = 32,
    saturation_only: Annotated[bool, typer.Option(help="Generate only saturation LUTs")] = False,
    contrast_only: Annotated[bool, typer.Option(help="Generate only contrast LUTs")] = False,
    brightness_only: Annotated[bool, typer.Option(help="Generate only brightness LUTs")] = False,
    exposure_only: Annotated[bool, typer.Option(help="Generate only exposure LUTs")] = False,
    gamma_only: Annotated[bool, typer.Option(help="Generate only gamma LUTs")] = False,
    hue_only: Annotated[bool, typer.Option(help="Generate only hue LUTs")] = False,
    temperature_only: Annotated[bool, typer.Option(help="Generate only temperature LUTs")] = False,
    tint_only: Annotated[bool, typer.Option(help="Generate only tint LUTs")] = False,
    grayscale_only: Annotated[bool, typer.Option(help="Generate only grayscale LUTs")] = False,
    test_image: Annotated[Optional[Path], typer.Option(help="Test image to apply each LUT to")] = None,
    dry_run: Annotated[bool, typer.Option(help="Preview without generating")] = False,
):
    """
    Generate utility LUTs with fixed mathematical transformations.

    Examples:
      # Generate all utility LUTs
      python scripts/generate_utility_luts.py --output-dir luts/utility/

      # Generate only saturation LUTs
      python scripts/generate_utility_luts.py --saturation-only --output-dir luts/
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which categories to generate
    generate_all = not any([
        saturation_only, contrast_only, brightness_only,
        exposure_only, gamma_only, hue_only,
        temperature_only, tint_only, grayscale_only
    ])

    luts_to_generate = []

    # Saturation LUTs
    if generate_all or saturation_only:
        luts_to_generate.extend([
            ("desaturate_25", lambda lut: apply_saturation(lut, 0.75)),
            ("desaturate_50", lambda lut: apply_saturation(lut, 0.5)),
            ("desaturate_75", lambda lut: apply_saturation(lut, 0.25)),
            ("oversaturate_125", lambda lut: apply_saturation(lut, 1.25)),
            ("oversaturate_150", lambda lut: apply_saturation(lut, 1.5)),
            ("oversaturate_200", lambda lut: apply_saturation(lut, 2.0)),
        ])

    # Contrast LUTs
    if generate_all or contrast_only:
        luts_to_generate.extend([
            ("low_contrast_50", lambda lut: apply_contrast(lut, 0.5)),
            ("low_contrast_75", lambda lut: apply_contrast(lut, 0.75)),
            ("high_contrast_125", lambda lut: apply_contrast(lut, 1.25)),
            ("high_contrast_150", lambda lut: apply_contrast(lut, 1.5)),
            ("high_contrast_200", lambda lut: apply_contrast(lut, 2.0)),
        ])

    # Brightness LUTs
    if generate_all or brightness_only:
        luts_to_generate.extend([
            ("brightness_50", lambda lut: apply_brightness(lut, 0.5)),
            ("brightness_75", lambda lut: apply_brightness(lut, 0.75)),
            ("brightness_125", lambda lut: apply_brightness(lut, 1.25)),
            ("brightness_150", lambda lut: apply_brightness(lut, 1.5)),
            ("brightness_200", lambda lut: apply_brightness(lut, 2.0)),
        ])

    # Exposure LUTs
    if generate_all or exposure_only:
        luts_to_generate.extend([
            ("exposure_minus_2", lambda lut: apply_exposure(lut, -2.0)),
            ("exposure_minus_1", lambda lut: apply_exposure(lut, -1.0)),
            ("exposure_minus_half", lambda lut: apply_exposure(lut, -0.5)),
            ("exposure_plus_half", lambda lut: apply_exposure(lut, 0.5)),
            ("exposure_plus_1", lambda lut: apply_exposure(lut, 1.0)),
            ("exposure_plus_2", lambda lut: apply_exposure(lut, 2.0)),
        ])

    # Gamma LUTs
    if generate_all or gamma_only:
        luts_to_generate.extend([
            ("gamma_0_5", lambda lut: apply_gamma(lut, 0.5)),
            ("gamma_0_75", lambda lut: apply_gamma(lut, 0.75)),
            ("gamma_1_25", lambda lut: apply_gamma(lut, 1.25)),
            ("gamma_1_5", lambda lut: apply_gamma(lut, 1.5)),
            ("gamma_2_0", lambda lut: apply_gamma(lut, 2.0)),
            ("gamma_2_2", lambda lut: apply_gamma(lut, 2.2)),  # sRGB standard
        ])

    # Hue shift LUTs (torchvision uses -0.5 to 0.5 range)
    if generate_all or hue_only:
        luts_to_generate.extend([
            ("hue_shift_30", lambda lut: apply_hue(lut, 30/360)),      # ~0.083
            ("hue_shift_60", lambda lut: apply_hue(lut, 60/360)),      # ~0.167
            ("hue_shift_90", lambda lut: apply_hue(lut, 90/360)),      # 0.25
            ("hue_shift_120", lambda lut: apply_hue(lut, 120/360)),    # ~0.333
            ("hue_shift_180", lambda lut: apply_hue(lut, 0.5)),        # 0.5 (max)
            ("hue_shift_240", lambda lut: apply_hue(lut, -120/360)),   # ~-0.333
            ("hue_shift_300", lambda lut: apply_hue(lut, -60/360)),    # ~-0.167
        ])

    # Temperature LUTs
    if generate_all or temperature_only:
        luts_to_generate.extend([
            ("cool_slight", lambda lut: apply_temperature(lut, -0.3)),
            ("cool_moderate", lambda lut: apply_temperature(lut, -0.6)),
            ("cool_strong", lambda lut: apply_temperature(lut, -1.0)),
            ("warm_slight", lambda lut: apply_temperature(lut, 0.3)),
            ("warm_moderate", lambda lut: apply_temperature(lut, 0.6)),
            ("warm_strong", lambda lut: apply_temperature(lut, 1.0)),
        ])

    # Tint LUTs
    if generate_all or tint_only:
        luts_to_generate.extend([
            ("tint_magenta_slight", lambda lut: apply_tint(lut, -0.3)),
            ("tint_magenta_moderate", lambda lut: apply_tint(lut, -0.6)),
            ("tint_magenta_strong", lambda lut: apply_tint(lut, -1.0)),
            ("tint_green_slight", lambda lut: apply_tint(lut, 0.3)),
            ("tint_green_moderate", lambda lut: apply_tint(lut, 0.6)),
            ("tint_green_strong", lambda lut: apply_tint(lut, 1.0)),
        ])

    # Grayscale LUTs
    if generate_all or grayscale_only:
        # Grayscale using torchvision
        luts_to_generate.append(
            ("grayscale_rec709", lambda lut: apply_grayscale(lut))
        )

        # Also generate grayscale identity LUT
        print("\nGenerating grayscale identity LUT...")
        generate_utility_lut(
            "grayscale_identity",
            lambda lut: lut,  # No transformation, just identity
            output_dir,
            lut_size=lut_size,
            grayscale=True,
            test_image=test_image,
            dry_run=dry_run
        )

    # Generate all LUTs
    print(f"\nGenerating {len(luts_to_generate)} utility LUTs...")
    print(f"Output directory: {output_dir}")
    print(f"LUT size: {lut_size}\n")

    for name, transform_fn in luts_to_generate:
        generate_utility_lut(
            name,
            transform_fn,
            output_dir,
            lut_size=lut_size,
            test_image=test_image,
            dry_run=dry_run
        )

    print(f"\n{'='*80}")
    print(f"UTILITY LUT GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {len(luts_to_generate) + (1 if (generate_all or grayscale_only) else 0)} LUTs")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    app()

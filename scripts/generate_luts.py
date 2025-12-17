#!/usr/bin/env python3
"""
Batch LUT generation script using reference files.
Creates sensible combinations of prompts and generates LUTs automatically.
"""

import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import typer
from PIL import Image
from typing_extensions import Annotated

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import read_cube_file, load_image_as_tensor
from utils.transforms import apply_lut

app = typer.Typer()


def load_references(file_path: Path) -> List[str]:
    """Load non-comment, non-empty lines from a reference file."""
    with open(file_path, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return lines


def generate_color_prompts(
    colors: List[str],
    emotions: List[str],
    film_formats: List[str],
    sample_size: int = None,
) -> List[Tuple[str, bool]]:
    """
    Generate color LUT prompts with sensible combinations.
    Returns list of (prompt, is_grayscale) tuples.
    """
    prompts = []

    # 1. Film stock + emotion
    for film, emotion in itertools.product(film_formats, emotions):
        prompts.append((f"{film} {emotion}", False))

    # 2. Film stock + color
    for film, color in itertools.product(film_formats, colors):
        prompts.append((f"{film} {color}", False))

    # 3. Emotion + color
    for emotion, color in itertools.product(emotions, colors):
        prompts.append((f"{emotion} {color}", False))

    # 4. Colors standalone
    for color in colors:
        prompts.append((color, False))

    # 5. Emotions standalone
    for emotion in emotions:
        prompts.append((emotion, False))

    # 6. Film stocks standalone
    for film in film_formats:
        prompts.append((film, False))

    if sample_size and sample_size < len(prompts):
        prompts = random.sample(prompts, sample_size)

    return prompts


def generate_bw_prompts(
    emotions: List[str], film_formats_bw: List[str], sample_size: int = None
) -> List[Tuple[str, bool]]:
    """
    Generate black & white LUT prompts.
    Returns list of (prompt, is_grayscale) tuples.
    """
    prompts = []

    # 1. B&W film stock + emotion
    for film, emotion in itertools.product(film_formats_bw, emotions):
        prompts.append((f"{film} {emotion}", True))

    # 2. B&W film stocks standalone
    for film in film_formats_bw:
        prompts.append((film, True))

    # 3. Emotions with "black and white" prefix
    for emotion in emotions:
        prompts.append((f"black and white {emotion}", True))

    if sample_size and sample_size < len(prompts):
        prompts = random.sample(prompts, sample_size)

    return prompts


def generate_standalone_prompts(
    movies: List[str],
    directors: List[str],
    movies_bw: List[str],
    directors_bw: List[str],
    sample_size: int = None,
) -> List[Tuple[str, bool]]:
    """
    Generate prompts from movies and directors (standalone only).
    Returns list of (prompt, is_grayscale) tuples.
    """
    prompts = []

    # Movies (color)
    for movie in movies:
        prompts.append((movie, False))

    # Directors (color)
    for director in directors:
        prompts.append((director, False))

    # Movies (B&W)
    for movie in movies_bw:
        prompts.append((movie, True))

    # Directors (B&W)
    for director in directors_bw:
        prompts.append((director, True))

    if sample_size and sample_size < len(prompts):
        prompts = random.sample(prompts, sample_size)

    return prompts


def parse_steps_range(steps_str: str) -> tuple[int, int]:
    """
    Parse steps argument which can be either a single value or a range.

    Args:
        steps_str: Either "500" or "200-600"

    Returns:
        Tuple of (min_steps, max_steps)
    """
    if "-" in steps_str:
        parts = steps_str.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid steps range format: {steps_str}. Expected 'min-max' or single integer"
            )
        min_steps, max_steps = int(parts[0]), int(parts[1])
        if min_steps > max_steps:
            raise ValueError(f"Invalid range: min ({min_steps}) > max ({max_steps})")
        return min_steps, max_steps
    else:
        # Single value - use same for min and max (no randomization)
        steps = int(steps_str)
        return steps, steps


def sanitize_filename(prompt: str) -> str:
    """Convert prompt to a safe filename."""
    # Replace spaces with underscores, remove special chars
    safe = prompt.lower()
    safe = safe.replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    # Limit length
    return safe[:100]


def save_lut_metadata(
    output_path: Path,
    prompt: str,
    is_grayscale: bool,
    model_type: str,
    steps: int,
    lut_size: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Save metadata JSON file alongside the LUT."""
    metadata: Dict[str, Any] = {
        "utility": False,
        "prompt": prompt,
        "black_and_white": is_grayscale,
        "model": model_type,
        "settings": {
            "steps": steps,
            "lut_size": lut_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_path}")


def apply_lut_to_test_image(
    lut_path: Path, test_image_path: Path, output_path: Path
) -> None:
    """Apply a LUT to a test image and save the result."""
    try:
        # Load LUT
        lut_tensor, domain_min, domain_max = read_cube_file(str(lut_path))

        # Load test image
        image_tensor = load_image_as_tensor(str(test_image_path))

        # Apply LUT
        result = apply_lut(image_tensor, lut_tensor, domain_min, domain_max)

        # Convert back to PIL and save
        result_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        result_img = Image.fromarray(result_np)
        result_img.save(output_path)

        print(f"  Saved test image result: {output_path}")
    except Exception as e:
        print(f"  WARNING: Failed to apply LUT to test image: {e}")


def generate_lut(
    prompt: str,
    is_grayscale: bool,
    image_folder: Path,
    output_dir: Path,
    model_type: str = "clip",
    steps: int = 500,
    lut_size: int = 16,
    batch_size: int = 4,
    learning_rate: float = 0.005,
    test_image: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """
    Generate a single LUT using main.py optimize command.
    Returns True if successful.
    """
    output_path = output_dir / f"{sanitize_filename(prompt)}.cube"

    cmd = [
        "python",
        "main.py",
        "optimize",
        "--prompt",
        prompt,
        "--image-folder",
        str(image_folder),
        "--output-path",
        str(output_path),
        "--model-type",
        model_type,
        "--steps",
        str(steps),
        "--lut-size",
        str(lut_size),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
    ]

    if is_grayscale:
        cmd.append("--grayscale")

    # Pass test image to optimization for training visualization
    if test_image:
        cmd.extend(["--test-image", str(test_image)])

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return True

    print(f"\n{'=' * 80}")
    print(f"Generating LUT: {prompt}")
    print(f"Grayscale: {is_grayscale}")
    print(f"Steps: {steps}")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(cmd, check=True)

        # Save metadata
        if not dry_run:
            save_lut_metadata(
                output_path=output_path,
                prompt=prompt,
                is_grayscale=is_grayscale,
                model_type=model_type,
                steps=steps,
                lut_size=lut_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        # Apply LUT to test image if provided
        if test_image and not dry_run:
            test_output_path = output_path.with_suffix(".png")
            apply_lut_to_test_image(output_path, test_image, test_output_path)

        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate LUT for '{prompt}': {e}")
        return False


# This is the typer-based main() to replace argparse version in generate_luts.py


@app.command()
def main(
    image_folder: Annotated[
        Path, typer.Option(help="Folder containing training images")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save generated LUTs")
    ] = Path("luts"),
    sample: Annotated[
        Optional[int],
        typer.Option(help="Random sample size (generates this many LUTs total)"),
    ] = None,
    color_only: Annotated[bool, typer.Option(help="Generate only color LUTs")] = False,
    bw_only: Annotated[
        bool, typer.Option(help="Generate only black & white LUTs")
    ] = False,
    standalone_only: Annotated[
        bool, typer.Option(help="Generate only standalone LUTs (movies and directors)")
    ] = False,
    model_type: Annotated[str, typer.Option(help="Model to use")] = "clip",
    steps: Annotated[str, typer.Option(help="Training iterations per LUT")] = "200-600",
    lut_size: Annotated[int, typer.Option(help="LUT resolution")] = 16,
    batch_size: Annotated[Optional[int], typer.Option(help="Batch size")] = None,
    learning_rate: Annotated[
        float, typer.Option(help="Learning rate for optimization")
    ] = 0.005,
    test_image: Annotated[
        Optional[Path], typer.Option(help="Test image to apply each LUT to")
    ] = None,
    seed: Annotated[
        Optional[int], typer.Option(help="Random seed for reproducible sampling")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Print commands without executing")
    ] = False,
):
    """
    Batch generate LUTs from reference files.

    Examples:
      # Generate 100 random color LUTs
      python scripts/generate_luts.py --image-folder images/ --sample 100 --output-dir luts/

      # Generate with randomized steps between 300-800
      python scripts/generate_luts.py --image-folder images/ --sample 50 --steps 300-800 --output-dir luts/
    """
    if seed:
        random.seed(seed)

    # Parse steps range
    min_steps, max_steps = parse_steps_range(steps)
    if min_steps == max_steps:
        print(f"Using fixed steps: {min_steps}")
    else:
        print(f"Using randomized steps: {min_steps}-{max_steps}")

    # Auto-adjust batch size based on model type if not specified
    if batch_size is None:
        if model_type == "clip":
            batch_size = 4
        else:  # sds, gemma3_4b, gemma3_12b, gemma3_27b
            batch_size = 1
        print(f"Auto-set batch size to {batch_size} for {model_type}")
    else:
        print(f"Using user-specified batch size: {batch_size}")

    # Load reference files
    prompts_dir = Path(__file__).parent / "prompts"

    colors = load_references(prompts_dir / "colors.txt")
    emotions = load_references(prompts_dir / "emotions.txt")
    film_formats = load_references(prompts_dir / "film_formats.txt")
    film_formats_bw = load_references(prompts_dir / "film_formats_bw.txt")
    movies = load_references(prompts_dir / "movies.txt")
    movies_bw = load_references(prompts_dir / "movies_bw.txt")
    directors = load_references(prompts_dir / "directors.txt")
    directors_bw = load_references(prompts_dir / "directors_bw.txt")

    print("Loaded references:")
    print(f"  Colors: {len(colors)}")
    print(f"  Emotions: {len(emotions)}")
    print(f"  Film formats (color): {len(film_formats)}")
    print(f"  Film formats (B&W): {len(film_formats_bw)}")
    print(f"  Movies (color): {len(movies)}")
    print(f"  Movies (B&W): {len(movies_bw)}")
    print(f"  Directors (color): {len(directors)}")
    print(f"  Directors (B&W): {len(directors_bw)}\n")

    # Generate prompts based on mode
    all_prompts = []

    if not bw_only and not standalone_only:
        # Generate color combination prompts
        color_prompts = generate_color_prompts(colors, emotions, film_formats)
        print(f"Generated {len(color_prompts)} color combination prompts")
        all_prompts.extend(color_prompts)

    if not color_only and not standalone_only:
        # Generate B&W prompts
        bw_prompts = generate_bw_prompts(emotions, film_formats_bw)
        print(f"Generated {len(bw_prompts)} B&W prompts")
        all_prompts.extend(bw_prompts)

    if not color_only and not bw_only:
        # Generate standalone prompts (movies + directors)
        standalone_prompts = generate_standalone_prompts(
            movies, directors, movies_bw, directors_bw
        )
        print(f"Generated {len(standalone_prompts)} standalone prompts")
        all_prompts.extend(standalone_prompts)

    # Sample if requested
    if sample and sample < len(all_prompts):
        all_prompts = random.sample(all_prompts, sample)
        print(f"\nSampled {sample} prompts from total")

    print(f"\nTotal prompts to generate: {len(all_prompts)}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LUTs
    successful = 0
    failed = 0

    for i, (prompt, is_grayscale) in enumerate(all_prompts, 1):
        print(f"\n[{i}/{len(all_prompts)}]")

        # Randomize steps for each LUT within the specified range
        steps_value = random.randint(min_steps, max_steps)

        success = generate_lut(
            prompt=prompt,
            is_grayscale=is_grayscale,
            image_folder=image_folder,
            output_dir=output_dir,
            model_type=model_type,
            steps=steps_value,
            lut_size=lut_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            test_image=test_image,
            dry_run=dry_run,
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 80}")
    print("BATCH GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    app()

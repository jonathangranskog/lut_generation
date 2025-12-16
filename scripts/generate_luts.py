#!/usr/bin/env python3
"""
Batch LUT generation script using reference files.
Creates sensible combinations of prompts and generates LUTs automatically.
"""

import argparse
import itertools
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import read_cube_file, load_image_as_tensor
from utils.transforms import apply_lut
from PIL import Image
import torch


def load_references(file_path: Path) -> List[str]:
    """Load non-comment, non-empty lines from a reference file."""
    with open(file_path, 'r') as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]
    return lines


def generate_color_prompts(
    colors: List[str],
    emotions: List[str],
    film_formats: List[str],
    sample_size: int = None
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
    emotions: List[str],
    film_formats_bw: List[str],
    sample_size: int = None
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
    sample_size: int = None
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
    if '-' in steps_str:
        parts = steps_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid steps range format: {steps_str}. Expected 'min-max' or single integer")
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
    safe = safe.replace(' ', '_')
    safe = ''.join(c for c in safe if c.isalnum() or c == '_')
    # Limit length
    return safe[:100]


def apply_lut_to_test_image(
    lut_path: Path,
    test_image_path: Path,
    output_path: Path
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
        result_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
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
    test_image: Optional[Path] = None,
    dry_run: bool = False
) -> bool:
    """
    Generate a single LUT using main.py optimize command.
    Returns True if successful.
    """
    output_path = output_dir / f"{sanitize_filename(prompt)}.cube"

    cmd = [
        "python", "main.py", "optimize",
        "--prompt", prompt,
        "--image-folder", str(image_folder),
        "--output-path", str(output_path),
        "--model-type", model_type,
        "--steps", str(steps),
        "--lut-size", str(lut_size),
        "--batch-size", str(batch_size),
    ]

    if is_grayscale:
        cmd.append("--grayscale")

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return True

    print(f"\n{'='*80}")
    print(f"Generating LUT: {prompt}")
    print(f"Grayscale: {is_grayscale}")
    print(f"Steps: {steps}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    try:
        subprocess.run(cmd, check=True)

        # Apply LUT to test image if provided
        if test_image and not dry_run:
            test_output_path = output_path.with_suffix('.png')
            apply_lut_to_test_image(output_path, test_image, test_output_path)

        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate LUT for '{prompt}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate LUTs from reference files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 random color LUTs
  python scripts/generate_luts.py --image-folder images/ --sample 100 --output-dir luts/

  # Generate all movie-based LUTs (standalone only)
  python scripts/generate_luts.py --image-folder images/ --standalone-only --output-dir luts/

  # Generate only B&W LUTs
  python scripts/generate_luts.py --image-folder images/ --bw-only --sample 50 --output-dir luts/

  # Generate with randomized steps between 300-800
  python scripts/generate_luts.py --image-folder images/ --sample 50 --steps 300-800 --output-dir luts/

  # Generate with fixed 1000 steps
  python scripts/generate_luts.py --image-folder images/ --sample 10 --steps 1000 --output-dir luts/

  # Dry run to see what would be generated
  python scripts/generate_luts.py --image-folder images/ --sample 10 --dry-run
        """
    )

    # Required arguments
    parser.add_argument(
        "--image-folder",
        type=Path,
        required=True,
        help="Folder containing training images"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("luts"),
        help="Directory to save generated LUTs (default: luts/)"
    )

    # Generation options
    parser.add_argument(
        "--sample",
        type=int,
        help="Random sample size (generates this many LUTs total)"
    )

    parser.add_argument(
        "--color-only",
        action="store_true",
        help="Generate only color LUTs (film+emotion, film+color, emotion+color combos)"
    )

    parser.add_argument(
        "--bw-only",
        action="store_true",
        help="Generate only black & white LUTs"
    )

    parser.add_argument(
        "--standalone-only",
        action="store_true",
        help="Generate only standalone LUTs (movies and directors)"
    )

    # LUT parameters
    parser.add_argument(
        "--model-type",
        default="clip",
        choices=["clip", "gemma3_4b", "gemma3_12b", "gemma3_27b", "sds"],
        help="Model to use for optimization (default: clip)"
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="200-600",
        help="Training iterations per LUT. Can be a single value (e.g., '500') or a range (e.g., '200-600' for randomization). Default: '200-600'"
    )

    parser.add_argument(
        "--lut-size",
        type=int,
        default=16,
        help="LUT resolution (default: 16)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )

    # Other options
    parser.add_argument(
        "--test-image",
        type=Path,
        help="Test image to apply each generated LUT to. Result saved as .png next to the LUT file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling"
    )

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    # Parse steps range
    min_steps, max_steps = parse_steps_range(args.steps)
    if min_steps == max_steps:
        print(f"Using fixed steps: {min_steps}")
    else:
        print(f"Using randomized steps: {min_steps}-{max_steps}")

    # Load reference files
    scripts_dir = Path(__file__).parent

    colors = load_references(scripts_dir / "colors.txt")
    emotions = load_references(scripts_dir / "emotions.txt")
    film_formats = load_references(scripts_dir / "film_formats.txt")
    film_formats_bw = load_references(scripts_dir / "film_formats_bw.txt")
    movies = load_references(scripts_dir / "movies.txt")
    movies_bw = load_references(scripts_dir / "movies_bw.txt")
    directors = load_references(scripts_dir / "directors.txt")
    directors_bw = load_references(scripts_dir / "directors_bw.txt")

    print(f"Loaded references:")
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

    if not args.bw_only and not args.standalone_only:
        # Generate color combination prompts
        color_prompts = generate_color_prompts(colors, emotions, film_formats)
        print(f"Generated {len(color_prompts)} color combination prompts")
        all_prompts.extend(color_prompts)

    if not args.color_only and not args.standalone_only:
        # Generate B&W prompts
        bw_prompts = generate_bw_prompts(emotions, film_formats_bw)
        print(f"Generated {len(bw_prompts)} B&W prompts")
        all_prompts.extend(bw_prompts)

    if not args.color_only and not args.bw_only:
        # Generate standalone prompts (movies + directors)
        standalone_prompts = generate_standalone_prompts(
            movies, directors, movies_bw, directors_bw
        )
        print(f"Generated {len(standalone_prompts)} standalone prompts")
        all_prompts.extend(standalone_prompts)

    # Sample if requested
    if args.sample and args.sample < len(all_prompts):
        all_prompts = random.sample(all_prompts, args.sample)
        print(f"\nSampled {args.sample} prompts from total")

    print(f"\nTotal prompts to generate: {len(all_prompts)}\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LUTs
    successful = 0
    failed = 0

    for i, (prompt, is_grayscale) in enumerate(all_prompts, 1):
        print(f"\n[{i}/{len(all_prompts)}]")

        # Randomize steps for each LUT within the specified range
        steps = random.randint(min_steps, max_steps)

        success = generate_lut(
            prompt=prompt,
            is_grayscale=is_grayscale,
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            model_type=args.model_type,
            steps=steps,
            lut_size=args.lut_size,
            batch_size=args.batch_size,
            test_image=args.test_image,
            dry_run=args.dry_run
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

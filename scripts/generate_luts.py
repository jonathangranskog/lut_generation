#!/usr/bin/env python3
"""
Batch LUT generation script using reference files.
Creates sensible combinations of prompts and generates LUTs automatically.
"""

import atexit
import itertools
import json
import random
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import typer
from PIL import Image
from typing_extensions import Annotated

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from representations import MLP
from utils.config import Config, load_config
from utils.io import load_image_as_tensor, read_cube_file
from utils.transforms import apply_lut

# Track temp files for cleanup on interrupt
_temp_files: Set[Path] = set()


def _cleanup_temp_files() -> None:
    for path in _temp_files:
        path.unlink(missing_ok=True)


def _signal_handler(signum: int, frame: Any) -> None:
    _cleanup_temp_files()
    sys.exit(1)


atexit.register(_cleanup_temp_files)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

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
    cities: List[str],
    weather: List[str],
) -> List[Tuple[str, bool]]:
    """
    Generate color LUT prompts with sensible combinations.
    Returns list of (prompt, is_grayscale) tuples.
    """
    prompts = []

    # Film stock + emotion
    for film, emotion in itertools.product(film_formats, emotions):
        prompts.append((f"{film} {emotion}", False))

    # Film stock + color
    for film, color in itertools.product(film_formats, colors):
        prompts.append((f"{film} {color}", False))

    # Emotion + color
    for emotion, color in itertools.product(emotions, colors):
        prompts.append((f"{emotion} {color}", False))

    # City + weather
    for city, weather_term in itertools.product(cities, weather):
        prompts.append((f"{city} {weather_term}", False))

    # City + emotion
    for city, emotion in itertools.product(cities, emotions):
        prompts.append((f"{city} {emotion}", False))

    # Weather + emotion
    for weather_term, emotion in itertools.product(weather, emotions):
        prompts.append((f"{weather_term} {emotion}", False))

    # City + color
    for city, color in itertools.product(cities, colors):
        prompts.append((f"{city} {color}", False))

    # Weather + color
    for weather_term, color in itertools.product(weather, colors):
        prompts.append((f"{weather_term} {color}", False))

    # Standalone terms
    for color in colors:
        prompts.append((color, False))
    for emotion in emotions:
        prompts.append((emotion, False))
    for film in film_formats:
        prompts.append((film, False))
    for city in cities:
        prompts.append((city, False))
    for weather_term in weather:
        prompts.append((weather_term, False))

    return prompts


def generate_bw_prompts(
    emotions: List[str],
    film_formats_bw: List[str],
    cities: List[str],
    weather: List[str],
) -> List[Tuple[str, bool]]:
    """
    Generate black & white LUT prompts.
    Returns list of (prompt, is_grayscale) tuples.
    """
    prompts = []

    # B&W film stock + emotion (film stocks are already B&W, no prefix needed)
    for film, emotion in itertools.product(film_formats_bw, emotions):
        prompts.append((f"{film} {emotion}", True))

    # B&W film stocks standalone
    for film in film_formats_bw:
        prompts.append((film, True))

    # Generic terms need "black and white" prefix
    for emotion in emotions:
        prompts.append((f"black and white {emotion}", True))

    for city in cities:
        prompts.append((f"black and white {city}", True))

    for weather_term in weather:
        prompts.append((f"black and white {weather_term}", True))

    # Combinations with "black and white" prefix
    for city, weather_term in itertools.product(cities, weather):
        prompts.append((f"black and white {city} {weather_term}", True))

    for city, emotion in itertools.product(cities, emotions):
        prompts.append((f"black and white {city} {emotion}", True))

    for weather_term, emotion in itertools.product(weather, emotions):
        prompts.append((f"black and white {weather_term} {emotion}", True))

    return prompts


def generate_standalone_prompts(
    movies: List[str],
    directors: List[str],
    movies_bw: List[str],
    directors_bw: List[str],
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

    # Movies (B&W) - these are already B&W movies, no prefix needed
    for movie in movies_bw:
        prompts.append((movie, True))

    # Directors (B&W) - these are already B&W directors, no prefix needed
    for director in directors_bw:
        prompts.append((director, True))

    return prompts


def parse_range(value_str: str, value_type: type, param_name: str) -> tuple:
    """Parse a parameter that can be either a single value or a range."""
    if "-" in value_str:
        parts = value_str.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid {param_name} range format: {value_str}. Expected 'min-max' or single {value_type.__name__}"
            )
        min_val, max_val = value_type(parts[0]), value_type(parts[1])
        if min_val > max_val:
            raise ValueError(
                f"Invalid {param_name} range: min ({min_val}) > max ({max_val})"
            )
        return min_val, max_val
    else:
        val = value_type(value_str)
        return val, val


def sanitize_filename(prompt: str) -> str:
    """Convert prompt to a safe filename."""
    safe = prompt.lower()
    safe = safe.replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    return safe[:100]


def save_lut_metadata(
    output_path: Path,
    prompt: str,
    config: Config,
) -> None:
    """Save metadata JSON file alongside the LUT."""
    metadata: Dict[str, Any] = {
        "prompt": prompt,
        "utility": False,
        "config": config.to_dict(),
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
        lut_tensor, domain_min, domain_max = read_cube_file(str(lut_path))
        image_tensor = load_image_as_tensor(str(test_image_path))
        result = apply_lut(image_tensor, lut_tensor, domain_min, domain_max)
        result_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        result_img = Image.fromarray(result_np)
        result_img.save(output_path)
        print(f"  Saved test image result: {output_path}")
    except Exception as e:
        print(f"  WARNING: Failed to apply LUT to test image: {e}")


def apply_mlp_to_test_image(
    mlp_path: Path, test_image_path: Path, output_path: Path
) -> None:
    """Apply an MLP to a test image and save the result."""
    import torch

    try:
        mlp = MLP.read(str(mlp_path))
        mlp.eval()
        image_tensor = load_image_as_tensor(str(test_image_path))
        with torch.no_grad():
            result = mlp(image_tensor, training=False)
        result_np = (result.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        result_img = Image.fromarray(result_np)
        result_img.save(output_path)
        print(f"  Saved test image result: {output_path}")
    except Exception as e:
        print(f"  WARNING: Failed to apply MLP to test image: {e}")


def generate_lut(
    prompt: str,
    is_grayscale: bool,
    image_folder: Path,
    output_dir: Path,
    base_config: Config,
    steps: int,
    learning_rate: float,
    test_image: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """
    Generate a single LUT or MLP using main.py optimize command.
    Creates a temporary config file with the appropriate representation type.
    Returns True if successful.
    """
    # Check if using MLP representation
    is_mlp = base_config.representation == "mlp"

    if is_mlp:
        # MLP representation - use .pt extension
        output_path = output_dir / f"{sanitize_filename(prompt)}.pt"
        representation = "mlp"
    else:
        # LUT representation - use .cube extension
        output_path = output_dir / f"{sanitize_filename(prompt)}.cube"
        representation = "bw_lut" if is_grayscale else "lut"

    # Create a modified config for this run
    config = Config(
        representation=representation,
        image_text_loss_type=base_config.image_text_loss_type,
        loss_weights=base_config.loss_weights,
        representation_args=base_config.representation_args,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=base_config.batch_size,
    )

    # Create a temporary config file and track it for cleanup
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        config.to_json(tmp_file.name)
        tmp_config_path = Path(tmp_file.name)
        _temp_files.add(tmp_config_path)

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
        "--config",
        str(tmp_config_path),
    ]

    if test_image:
        cmd.extend(["--test-image", str(test_image)])

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        tmp_config_path.unlink(missing_ok=True)
        _temp_files.discard(tmp_config_path)
        return True

    print(f"\n{'=' * 80}")
    print(f"Generating: {prompt}")
    print(f"Representation: {representation}")
    print(f"Steps: {steps}")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(cmd, check=True)

        save_lut_metadata(
            output_path=output_path,
            prompt=prompt,
            config=config,
        )

        # Handle test image based on representation type
        if test_image:
            test_output_path = output_path.with_suffix(".png")
            if is_mlp:
                apply_mlp_to_test_image(output_path, test_image, test_output_path)
            else:
                apply_lut_to_test_image(output_path, test_image, test_output_path)

        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate for '{prompt}': {e}")
        return False
    finally:
        tmp_config_path.unlink(missing_ok=True)
        _temp_files.discard(tmp_config_path)


@app.command()
def main(
    image_folder: Annotated[
        Path, typer.Option(help="Folder containing training images")
    ],
    config: Annotated[
        str,
        typer.Option(help="Path to JSON config file"),
    ] = "configs/color_clip.json",
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save generated LUTs")
    ] = Path("luts"),
    sample: Annotated[
        Optional[int],
        typer.Option(help="Random sample size (generates this many LUTs total)"),
    ] = None,
    bw_percentage: Annotated[
        float,
        typer.Option(help="Percentage of LUTs to generate as black & white (0.0-1.0)"),
    ] = 0.05,
    color_only: Annotated[
        bool,
        typer.Option(help="Only generate color LUTs"),
    ] = False,
    bw_only: Annotated[
        bool,
        typer.Option(help="Only generate black & white LUTs"),
    ] = False,
    standalone_only: Annotated[
        bool,
        typer.Option(help="Only generate standalone LUTs (movies and directors)"),
    ] = False,
    steps: Annotated[
        Optional[str],
        typer.Option(
            help="Training steps (overrides config, supports ranges like 200-600)"
        ),
    ] = None,
    learning_rate: Annotated[
        Optional[str],
        typer.Option(
            help="Learning rate (overrides config, supports ranges like 0.001-0.01)"
        ),
    ] = None,
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
    Batch generate LUTs from reference files using a config file.

    Examples:
      # Generate 100 random LUTs with 5% B&W (default)
      python scripts/generate_luts.py --image-folder images/ --sample 100 --output-dir luts/

      # Generate 100 LUTs with 10% B&W
      python scripts/generate_luts.py --image-folder images/ --sample 100 --bw-percentage 0.1

      # Generate with randomized steps (overrides config)
      python scripts/generate_luts.py --image-folder images/ --sample 50 --steps 300-800

      # Use a different config
      python scripts/generate_luts.py --image-folder images/ --sample 10 --config configs/color_sds.json

      # Generate using MLP config
      python scripts/generate_luts.py --image-folder images/ --sample 10 --config configs/mlp_clip.json
    """
    if seed:
        random.seed(seed)

    # Load base config
    base_config = load_config(config)
    print(f"Loaded config from: {config}")
    print(f"  Representation: {base_config.representation}")
    print(f"  Model type: {base_config.image_text_loss_type}")
    print(f"  Batch size: {base_config.batch_size}")
    if base_config.representation == "mlp":
        print(f"  MLP args: {base_config.representation_args}")

    # Parse steps range (use config default if not specified)
    if steps is None:
        min_steps = max_steps = base_config.steps
        print(f"Using config steps: {min_steps}")
    else:
        min_steps, max_steps = parse_range(steps, int, "steps")
        if min_steps == max_steps:
            print(f"Using fixed steps: {min_steps}")
        else:
            print(f"Using randomized steps: {min_steps}-{max_steps}")

    # Parse learning rate range (use config default if not specified)
    if learning_rate is None:
        min_lr = max_lr = base_config.learning_rate
        print(f"Using config learning rate: {min_lr}")
    else:
        min_lr, max_lr = parse_range(learning_rate, float, "learning rate")
        if min_lr == max_lr:
            print(f"Using fixed learning rate: {min_lr}")
        else:
            print(f"Using randomized learning rate: {min_lr}-{max_lr}")

    # Load reference files
    prompts_dir = Path(__file__).parent / "prompts"

    colors = load_references(prompts_dir / "colors.txt")
    emotions = load_references(prompts_dir / "emotions.txt")
    film_formats = load_references(prompts_dir / "film_formats.txt")
    film_formats_bw = load_references(prompts_dir / "film_formats_bw.txt")
    cities = load_references(prompts_dir / "cities.txt")
    weather = load_references(prompts_dir / "weather.txt")
    movies = load_references(prompts_dir / "movies.txt")
    movies_bw = load_references(prompts_dir / "movies_bw.txt")
    directors = load_references(prompts_dir / "directors.txt")
    directors_bw = load_references(prompts_dir / "directors_bw.txt")

    print("\nLoaded references:")
    print(f"  Colors: {len(colors)}")
    print(f"  Emotions: {len(emotions)}")
    print(f"  Film formats (color): {len(film_formats)}")
    print(f"  Film formats (B&W): {len(film_formats_bw)}")
    print(f"  Cities: {len(cities)}")
    print(f"  Weather: {len(weather)}")
    print(f"  Movies (color): {len(movies)}")
    print(f"  Movies (B&W): {len(movies_bw)}")
    print(f"  Directors (color): {len(directors)}")
    print(f"  Directors (B&W): {len(directors_bw)}")

    # Generate prompts based on mode
    all_prompts = []

    if not bw_only and not standalone_only:
        color_prompts = generate_color_prompts(
            colors, emotions, film_formats, cities, weather
        )
        print(f"\nGenerated {len(color_prompts)} color combination prompts")
        all_prompts.extend(color_prompts)

    if not color_only and not standalone_only:
        bw_prompts = generate_bw_prompts(emotions, film_formats_bw, cities, weather)
        print(f"Generated {len(bw_prompts)} B&W prompts")
        all_prompts.extend(bw_prompts)

    if not color_only and not bw_only:
        standalone_prompts = generate_standalone_prompts(
            movies, directors, movies_bw, directors_bw
        )
        print(f"Generated {len(standalone_prompts)} standalone prompts")
        all_prompts.extend(standalone_prompts)

    # Sample if requested, respecting B&W percentage
    if sample and sample < len(all_prompts):
        if not 0.0 <= bw_percentage <= 1.0:
            print(
                f"ERROR: bw_percentage must be between 0.0 and 1.0, got {bw_percentage}"
            )
            raise typer.Exit(1)

        color_prompts = [p for p in all_prompts if not p[1]]
        bw_prompts = [p for p in all_prompts if p[1]]

        num_bw = int(sample * bw_percentage)
        num_color = sample - num_bw

        if num_bw > len(bw_prompts):
            print(
                f"WARNING: Requested {num_bw} B&W LUTs but only {len(bw_prompts)} available"
            )
            num_bw = len(bw_prompts)
            num_color = sample - num_bw

        if num_color > len(color_prompts):
            print(
                f"WARNING: Requested {num_color} color LUTs but only {len(color_prompts)} available"
            )
            num_color = len(color_prompts)
            num_bw = sample - num_color

        sampled_bw = random.sample(bw_prompts, num_bw) if num_bw > 0 else []
        sampled_color = random.sample(color_prompts, num_color) if num_color > 0 else []

        all_prompts = sampled_color + sampled_bw
        random.shuffle(all_prompts)

        print(f"\nSampled {len(all_prompts)} prompts:")
        print(f"  Color: {num_color} ({num_color / len(all_prompts) * 100:.1f}%)")
        print(f"  B&W: {num_bw} ({num_bw / len(all_prompts) * 100:.1f}%)")

    print(f"\nTotal prompts to generate: {len(all_prompts)}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LUTs
    successful = 0
    failed = 0

    for i, (prompt, is_grayscale) in enumerate(all_prompts, 1):
        print(f"\n[{i}/{len(all_prompts)}]")

        steps_value = random.randint(min_steps, max_steps)
        learning_rate_value = random.uniform(min_lr, max_lr)

        success = generate_lut(
            prompt=prompt,
            is_grayscale=is_grayscale,
            image_folder=image_folder,
            output_dir=output_dir,
            base_config=base_config,
            steps=steps_value,
            learning_rate=learning_rate_value,
            test_image=test_image,
            dry_run=dry_run,
        )

        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n{'=' * 80}")
    print("BATCH GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    app()

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
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import typer
from PIL import Image
from typing_extensions import Annotated

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config, load_config
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


def generate_prompts(
    colors: List[str],
    emotions: List[str],
    film_formats: List[str],
    film_formats_bw: List[str],
    cities: List[str],
    weather: List[str],
    movies: List[str],
    movies_bw: List[str],
    directors: List[str],
    directors_bw: List[str],
) -> List[str]:
    """
    Generate all possible LUT prompts with sensible combinations.
    Returns list of prompt strings.
    """
    prompts = []

    # Film stock + emotion
    for film, emotion in itertools.product(film_formats, emotions):
        prompts.append(f"{film} {emotion}")

    # Film stock + color
    for film, color in itertools.product(film_formats, colors):
        prompts.append(f"{film} {color}")

    # Emotion + color
    for emotion, color in itertools.product(emotions, colors):
        prompts.append(f"{emotion} {color}")

    # City + weather
    for city, weather_term in itertools.product(cities, weather):
        prompts.append(f"{city} {weather_term}")

    # City + emotion
    for city, emotion in itertools.product(cities, emotions):
        prompts.append(f"{city} {emotion}")

    # Weather + emotion
    for weather_term, emotion in itertools.product(weather, emotions):
        prompts.append(f"{weather_term} {emotion}")

    # City + color
    for city, color in itertools.product(cities, colors):
        prompts.append(f"{city} {color}")

    # Weather + color
    for weather_term, color in itertools.product(weather, colors):
        prompts.append(f"{weather_term} {color}")

    # Standalone prompts
    prompts.extend(colors)
    prompts.extend(emotions)
    prompts.extend(film_formats)
    prompts.extend(film_formats_bw)
    prompts.extend(cities)
    prompts.extend(weather)
    prompts.extend(movies)
    prompts.extend(movies_bw)
    prompts.extend(directors)
    prompts.extend(directors_bw)

    return prompts


def parse_range(value_str: str, value_type: type, param_name: str) -> tuple:
    """
    Parse a parameter that can be either a single value or a range.

    Args:
        value_str: Either a single value like "500" or a range like "200-600"
        value_type: Type to convert to (int or float)
        param_name: Name of parameter for error messages

    Returns:
        Tuple of (min_value, max_value)
    """
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
        # Single value - use same for min and max (no randomization)
        val = value_type(value_str)
        return val, val


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
    config: Config,
    steps: int,
    learning_rate: float,
) -> None:
    """Save metadata JSON file alongside the LUT."""
    metadata: Dict[str, Any] = {
        "utility": False,
        "prompt": prompt,
        "representation": config.representation,
        "black_and_white": config.representation == "bw_lut",
        "model": config.image_text_loss_type,
        "settings": {
            "steps": steps,
            "lut_size": config.representation_args.get("lut_size", 16),
            "batch_size": config.batch_size,
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
    image_folder: Path,
    output_dir: Path,
    base_config: Config,
    steps: int = 500,
    learning_rate: float = 0.005,
    test_image: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """
    Generate a single LUT using main.py optimize command.
    Uses the representation type from the config.
    Returns True if successful.
    """
    output_path = output_dir / f"{sanitize_filename(prompt)}.cube"

    # Create a modified config for this LUT (with potentially overridden steps/lr)
    config = Config(
        representation=base_config.representation,
        image_text_loss_type=base_config.image_text_loss_type,
        loss_weights=base_config.loss_weights,
        representation_args=base_config.representation_args,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=base_config.batch_size,
    )

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        config.to_json(tmp_file.name)
        tmp_config_path = tmp_file.name

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
        tmp_config_path,
    ]

    # Pass test image to optimization for training visualization
    if test_image:
        cmd.extend(["--test-image", str(test_image)])

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        # Clean up temp file
        Path(tmp_config_path).unlink(missing_ok=True)
        return True

    is_grayscale = base_config.representation == "bw_lut"
    print(f"\n{'=' * 80}")
    print(f"Generating LUT: {prompt}")
    print(f"Representation: {base_config.representation}")
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
                config=config,
                steps=steps,
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
    finally:
        # Clean up temp file
        Path(tmp_config_path).unlink(missing_ok=True)


@app.command()
def main(
    image_folder: Annotated[
        Path, typer.Option(help="Folder containing training images")
    ],
    config: Annotated[
        str,
        typer.Option(help="Path to JSON config file (determines representation type, model, etc.)"),
    ] = "configs/color_clip.json",
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save generated LUTs")
    ] = Path("luts"),
    sample: Annotated[
        Optional[int],
        typer.Option(help="Random sample size (generates this many LUTs total)"),
    ] = None,
    steps: Annotated[
        str, typer.Option(help="Training iterations per LUT (overrides config, supports ranges like 200-600)")
    ] = None,
    learning_rate: Annotated[
        str, typer.Option(help="Learning rate for optimization (overrides config, supports ranges like 0.001-0.01)")
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

    The config file determines the representation type (color LUT or B&W LUT),
    the model to use (CLIP, SDS, Gemma), and other training parameters.

    Examples:
      # Generate 100 random color LUTs using default color_clip config
      python scripts/generate_luts.py --image-folder images/ --sample 100 --output-dir luts/

      # Generate B&W LUTs using bw_clip config
      python scripts/generate_luts.py --image-folder images/ --sample 50 --config configs/bw_clip.json

      # Generate using SDS config
      python scripts/generate_luts.py --image-folder images/ --sample 10 --config configs/color_sds.json

      # Generate with randomized steps between 300-800 (overrides config)
      python scripts/generate_luts.py --image-folder images/ --sample 50 --steps 300-800 --output-dir luts/

      # Generate with randomized learning rate between 0.001-0.01 (overrides config)
      python scripts/generate_luts.py --image-folder images/ --sample 50 --learning-rate 0.001-0.01 --output-dir luts/
    """
    if seed:
        random.seed(seed)

    # Load base config
    base_config = load_config(config)
    print(f"Loaded config from: {config}")
    print(f"  Representation: {base_config.representation}")
    print(f"  Model type: {base_config.image_text_loss_type}")
    print(f"  Batch size: {base_config.batch_size}")
    print(f"  Default steps: {base_config.steps}")
    print(f"  Default learning rate: {base_config.learning_rate}")

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
    print(f"  Film formats: {len(film_formats)}")
    print(f"  Film formats (B&W): {len(film_formats_bw)}")
    print(f"  Cities: {len(cities)}")
    print(f"  Weather: {len(weather)}")
    print(f"  Movies: {len(movies)}")
    print(f"  Movies (B&W): {len(movies_bw)}")
    print(f"  Directors: {len(directors)}")
    print(f"  Directors (B&W): {len(directors_bw)}")

    # Generate all prompts
    all_prompts = generate_prompts(
        colors=colors,
        emotions=emotions,
        film_formats=film_formats,
        film_formats_bw=film_formats_bw,
        cities=cities,
        weather=weather,
        movies=movies,
        movies_bw=movies_bw,
        directors=directors,
        directors_bw=directors_bw,
    )
    print(f"\nGenerated {len(all_prompts)} total prompts")

    # Sample if requested
    if sample and sample < len(all_prompts):
        all_prompts = random.sample(all_prompts, sample)
        print(f"Sampled {len(all_prompts)} prompts")

    print(f"\nTotal prompts to generate: {len(all_prompts)}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LUTs
    successful = 0
    failed = 0

    for i, prompt in enumerate(all_prompts, 1):
        print(f"\n[{i}/{len(all_prompts)}]")

        # Randomize steps for each LUT within the specified range
        steps_value = random.randint(min_steps, max_steps)

        # Randomize learning rate for each LUT within the specified range
        learning_rate_value = random.uniform(min_lr, max_lr)

        success = generate_lut(
            prompt=prompt,
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

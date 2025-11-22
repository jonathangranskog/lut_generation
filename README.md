# LUT Generation with CLIP

Generate custom 3D LUTs (Look-Up Tables) for color grading using CLIP-guided optimization. Transform your images to match any text prompt (e.g., "golden hour", "cinematic teal and orange", "vintage film").

## Features

- 🎨 **Text-to-LUT**: Generate LUTs from natural language prompts using CLIP
- 🖼️ **Image-space optimization**: Direct anti-banding and artifact prevention
- 📊 **Training monitoring**: Automatic logging of intermediate results
- 🔧 **Multiple formats**: Export standard .cube files compatible with most photo/video software
- ⚡ **GPU & CPU support**: Works with CUDA, MPS (Apple Silicon), or CPU

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/jonathangranskog/lut_generation.git
cd lut_generation

# Install dependencies
uv sync
```

## Quick Start

### 1. Optimize a LUT

```bash
uv run python main.py optimize \
  --prompt "golden hour warm sunlight" \
  --image-folder images_resized/ \
  --steps 500
```

This will:
- Load images from `images_resized/`
- Optimize a LUT to make images match the prompt
- Save training progress to `tmp/training_logs/golden_hour_warm_sunlight/`
- Save the final LUT to `lut.cube`

### 2. Apply a LUT to Images

```bash
uv run python main.py infer \
  lut.cube \
  input_image.jpg \
  --output-path result.jpg
```

## Commands

### `optimize` - Generate a LUT

```bash
uv run python main.py optimize [OPTIONS]
```

**Required Options:**
- `--prompt TEXT`: Text description of desired look (e.g., "warm sunset", "cinematic look")
- `--image-folder PATH`: Folder containing training images

**Key Options:**
- `--lut-size INT`: LUT resolution (default: 32). Higher = smoother but slower (16, 32, 64)
- `--steps INT`: Training iterations (default: 500)
- `--learning-rate FLOAT`: Learning rate (default: 0.005)
- `--image-smoothness FLOAT`: Anti-banding strength (default: 1.0, range: 0.1-2.0)
- `--image-regularization FLOAT`: Keep changes subtle (default: 1.0, range: 0.1-2.0)
- `--batch-size INT`: Batch size (default: 4)
- `--log-interval INT`: Save progress every N steps (default: 50, 0 to disable)
- `--output-path PATH`: Output .cube file (default: "lut.cube")
- `--verbose`: Show detailed loss breakdown every 10 steps

**Example:**
```bash
uv run python main.py optimize \
  --prompt "cinematic teal and orange" \
  --image-folder images_resized/ \
  --lut-size 32 \
  --steps 500 \
  --learning-rate 0.005 \
  --image-smoothness 1.0 \
  --image-regularization 1.0 \
  --output-path cinematic.cube \
  --verbose
```

### `infer` - Apply a LUT

```bash
uv run python main.py infer LUT_FILE IMAGE_FILE [OPTIONS]
```

**Arguments:**
- `LUT_FILE`: Path to .cube file
- `IMAGE_FILE`: Path to input image

**Options:**
- `--output-path PATH`: Output image path (default: "output.png")

**Example:**
```bash
uv run python main.py infer \
  golden_hour.cube \
  photo.jpg \
  --output-path photo_graded.jpg
```

## Understanding the Parameters

### Loss Components

During optimization, the total loss consists of:

1. **CLIP Loss** (~0.6-0.9): Semantic similarity to text prompt
2. **Image Smoothness** (0.0-0.01): Prevents banding and posterization
3. **Image Regularization** (0.0-0.1): Keeps output close to input (subtle changes)

Watch the verbose output to see how each contributes:
```
Step 100: Loss = 0.7234 (CLIP: 0.7123, Smooth: 0.0089, Reg: 0.0022)
```

## License

MIT.


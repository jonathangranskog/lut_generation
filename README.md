# LUT Generation (🚧 WIP 🚧)

Generate custom 3D LUTs (Look-Up Tables) for color grading using CLIP-guided optimization. Transform your images to match any text prompt (e.g., "golden hour", "cinematic teal and orange", "vintage film").

Note: a lot of vibe-coding was used to write this code. 

## Features

- 🎨 **Text-to-LUT**: Generate LUTs from natural language prompts using CLIP
- 🔧 **Export format**: Exports standard .cube files compatible with most photo/video software
- ⚡ **GPU & CPU support**: Works with CUDA, MPS (Apple Silicon), or CPU

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- A small dataset of 25-100 images

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
python main.py optimize \
  --prompt "golden hour warm sunlight" \
  --image-folder images/ \
  --steps 500
```

This will:
- Load images from `images/`
- Optimize a LUT to make images match the prompt
- Save training progress to `tmp/training_logs/golden_hour_warm_sunlight/`
- Save the final LUT to `lut.cube`

### 2. Apply a LUT to Images

```bash
python main.py infer \
  lut.cube \
  input_image.jpg \
  --output-path result.jpg
```

## Commands

### `optimize` - Generate a LUT

```bash
python main.py optimize [OPTIONS]
```

**Required Options:**
- `--prompt TEXT`: Text description of desired look (e.g., "warm sunset", "cinematic look")
- `--image-folder PATH`: Folder containing training images

**Key Options:**
- `--lut-size INT`: LUT resolution (default: 16). Higher = more detailed, but more prone to banding artifacts
- `--steps INT`: Training iterations (default: 500)
- `--learning-rate FLOAT`: Learning rate (default: 0.005)
- `--image-smoothness FLOAT`: Image-space anti-banding strength (default: 1.0)
- `--image-regularization FLOAT`: Keep changes subtle (default: 1.0)
- `--black-preservation FLOAT`: Retain black values to reduce fading (default: 1.0)
- `--lut-smoothness FLOAT`: LUT-space anti-banding strength (default: 1.0) 
- `--batch-size INT`: Batch size (default: 4)
- `--log-interval INT`: Save progress every N steps (default: 50, 0 to disable)
- `--output-path PATH`: Output .cube file (default: "lut.cube")
- `--test-image PATH`: Image to apply LUT to during logging (default picks a random training image)
- `--verbose`: Show detailed loss breakdown every 10 steps

**Example:**
```bash
python main.py optimize \
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
python main.py infer LUT_FILE IMAGE_FILE [OPTIONS]
```

**Arguments:**
- `LUT_FILE`: Path to .cube file
- `IMAGE_FILE`: Path to input image

**Options:**
- `--output-path PATH`: Output image path (default: "output.png")

**Example:**
```bash
python main.py infer \
  golden_hour.cube \
  photo.jpg \
  --output-path photo_graded.jpg
```

## Understanding the Parameters

### Loss Components

During optimization, the total loss consists of:

1. **Image-text Loss** (~0.6-0.9): Semantic similarity to text prompt
2. **Image Smoothness** (0.0-0.01): Prevents banding and posterization
3. **Image Regularization** (0.0-0.1): Keeps output close to input (subtle changes)
4. **Black Preservation** (~0.0-0.0001): Retains black levels
5. **LUT Smoothness** (~0.0-0.01): Prevents banding in LUT-space

Watch the verbose output to see how each contributes:
```
Step 100: Loss = 0.7935 (CLIP: 0.7905, Smooth: 0.0004, Reg: 0.0017, Black: 0.0000, LUT Smooth: 0.0008)
```

# Limitations

### Accuracy

Large transformations, like black-and-white filters, are harder to learn. Some prompts might lead to incorrect results based on the model's understanding of language (e.g. "not red" might lead to "red" LUTs).

### Artifacts

Higher resolution or long training of LUTs might lead to banding artifacts. To combat, increase weights of regularization losses, lower learning rate or reduce LUT resolution.

# Future Improvements

* Support VLM-based optimization (similar to our paper [Dual-Process Image Generation](https://dual-process.github.io/)) or SDS optimization (like in [DreamFusion](https://dreamfusion3d.github.io/))
* Better regularization so more complex LUTs can be generated
* Other representations besides LUTs
* Large-scale LUT library generation

## License

MIT.


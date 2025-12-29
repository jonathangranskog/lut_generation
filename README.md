# LUT Generation

Generate custom 3D LUTs (Look-Up Tables) for color grading using AI-guided optimization. Transform your images to match any text prompt (e.g., "golden hour", "cinematic teal and orange", "vintage film").

Note: a lot of vibe-coding was used to write this code.

![LUT Generation Banner](assets/banner.png)

## Features

- 🎨 **Text-to-LUT**: Generate LUTs from natural language prompts using CLIP, Gemma 3, or DeepFloyd IF
- 🤖 **Multiple Models**: Choose from CLIP, Gemma 3 (4B, 12B, 27B), or SDS for different quality/speed tradeoffs
- 🔍 **Context-Aware VLM**: Gemma 3 models evaluate transformations by comparing before/after images
- 🔧 **Export format**: Exports standard .cube files compatible with most photo/video software
- ⚡ **GPU & CPU support**: Works with CUDA, MPS (Apple Silicon), or CPU

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- A small dataset of 25-100 images, alternatively you can download [some images from me](https://drive.google.com/file/d/1bN03SkGs6E_K1C9Dc9r1Hl0BK48LqMQ2/view?usp=drive_link)

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
  --config configs/color_clip.json
```

This will:
- Load images from `images/`
- Use the color CLIP configuration (500 steps, learning rate 0.005, batch size 4)
- Optimize a LUT to make images match the prompt
- Save training progress to `tmp/training_logs/golden_hour_warm_sunlight/`
- Save the final LUT to `lut.cube`

Available default configs:
- `configs/color_clip.json` - Color LUT with CLIP (default)
- `configs/bw_clip.json` - Black & white LUT with CLIP
- `configs/color_sds.json` - Color LUT with SDS (DeepFloyd IF)
- `configs/color_gemma3_12b.json` - Color LUT with Gemma 3 12B

### 2. Apply a LUT to Images

```bash
python main.py infer \
  lut.cube \
  input_image.jpg \
  --output-path result.jpg
```

### 3. Run tests

```bash
pytest -v
```

It will skip the VLM tests if you are not logged in to huggingface or if CUDA is unavailable. 

### 4. LUT library generation

```bash
python scripts/generate_luts.py --sample 100 --test-image images/IMG_0001.png --image-folder images --output-dir tmp/generated_luts/
```

This command generates 100 LUTs automatically using the default color_clip config. You can also specify a different config:

```bash
python scripts/generate_luts.py --sample 100 --config configs/color_sds.json --image-folder images --output-dir tmp/generated_luts/
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
- `--config PATH`: Path to JSON config file (default: "configs/color_clip.json")
- `--log-interval INT`: Save progress every N steps (default: 50, 0 to disable)
- `--output-path PATH`: Output .cube file (default: "lut.cube")
- `--test-image PATH`: Image to apply LUT to during logging (repeat flag for multiple images, default picks a random training image)
- `--verbose`: Show detailed loss breakdown every 10 steps
- `--log-dir PATH`: Custom directory for training logs

**Config File Format:**

Config files are JSON files that specify all training parameters:

```json
{
  "representation": "lut",
  "image_text_loss_type": "clip",
  "loss_weights": {
    "image_text": 1.0,
    "image_smoothness": 1.0,
    "image_regularization": 1.0,
    "black_preservation": 1.0,
    "repr_smoothness": 1.0
  },
  "representation_args": {
    "size": 16
  },
  "steps": 500,
  "learning_rate": 0.005,
  "batch_size": 4
}
```

- `representation`: "lut" for color or "bw_lut" for black & white
- `image_text_loss_type`: "clip", "gemma3_4b", "gemma3_12b", "gemma3_27b", or "sds"
- `loss_weights`: Weight for each loss component
- `representation_args`: Representation-specific arguments (e.g., `{"size": 16}` for LUT representations)
- `steps`: Number of training iterations
- `learning_rate`: Optimizer learning rate
- `batch_size`: Training batch size

**Examples:**

Standard color LUT with CLIP:
```bash
python main.py optimize \
  --prompt "cinematic teal and orange" \
  --image-folder images/ \
  --config configs/color_clip.json \
  --output-path cinematic.cube \
  --test-image photo1.jpg \
  --test-image photo2.jpg \
  --verbose
```

VLM with context-aware transformations (Gemma 3 12B):
```bash
python main.py optimize \
  --prompt "warm golden hour" \
  --image-folder images/ \
  --config configs/color_gemma3_12b.json \
  --output-path golden_hour.cube
```

SDS with DeepFloyd IF:
```bash
python main.py optimize \
  --prompt "kodak portra 400 film" \
  --image-folder images/ \
  --config configs/color_sds.json \
  --output-path portra.cube
```

Black-and-white LUT with CLIP:
```bash
python main.py optimize \
  --prompt "black and white noir film" \
  --image-folder images/ \
  --config configs/bw_clip.json \
  --output-path noir_bw.cube
```

### `infer` - Apply a LUT

```bash
python main.py infer CKPT_PATH IMAGE_FILE [OPTIONS]
```

**Arguments:**
- `CKPT_PATH`: Path to .cube file
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

1. **Image-text Loss** (dependent on model): Semantic similarity to text prompt
2. **Image Smoothness** (0.0-0.01): Prevents banding and posterization
3. **Image Regularization** (0.0-0.1): Keeps output close to input (subtle changes)
4. **Black Preservation** (~0.0-0.0001): Retains black levels
5. **Repr Smoothness** (~0.0-0.01): Prevents banding in representation-space

Watch the verbose output to see how each contributes:
```
Step 100: Loss = 0.7935 (CLIP: 0.7905, Smooth: 0.0004, Reg: 0.0017, Black: 0.0000, Repr Smooth: 0.0008)
```

# Limitations

### Accuracy

Large transformations, like black-and-white filters, are harder to learn. Some prompts might lead to incorrect results based on the model's understanding of language (e.g. "not red" might lead to "red" LUTs).

### Artifacts

Higher resolution or long training of LUTs might lead to banding artifacts. To combat, increase weights of regularization losses, lower learning rate or reduce LUT resolution.

# Future Improvements

* Better regularization so more complex LUTs can be generated
* Other representations besides LUTs

# License

The code is licensed under the MIT License.

Note that [Gemma](https://ai.google.dev/gemma/terms) and [DeepFloydIF](https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license) have their own licenses that you must follow if you use them.
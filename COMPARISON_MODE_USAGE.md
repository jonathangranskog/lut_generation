# Gemma 3 Context-Aware Color Grading

## Overview

All Gemma 3 models (gemma3_4b, gemma3_12b, gemma3_27b) use a **context-aware comparison approach** that evaluates transformations by comparing original and transformed images, rather than just assessing the final result.

The VLM is asked: *"Looking at these two images, has the '{prompt}' color grade been successfully applied to transform the first image into the second?"*

## Why This Approach Works Better

Context-aware evaluation provides **better gradient signals** by:

- **Relative assessment**: Judges the transformation relative to the starting point, not just the final result
- **Prevents over-transformation**: Can identify when a transformation is too aggressive or not appropriate for the input
- **Handles diverse inputs**: Works better when input images vary widely (some already close to target, others far away)
- **Transformation awareness**: Explicitly evaluates the change applied, not just whether the output matches the prompt

## Usage

### Basic Usage

Simply select a Gemma 3 model type to automatically use context-aware evaluation:

```bash
python main.py optimize \
  --prompt "warm golden hour" \
  --image-folder ./images \
  --model-type gemma3_12b
```

### Full Example with All Options

```bash
python main.py optimize \
  --prompt "cinematic teal and orange" \
  --image-folder ./dataset \
  --model-type gemma3_12b \
  --steps 1000 \
  --batch-size 4 \
  --learning-rate 0.005 \
  --image-text-weight 1.0 \
  --image-smoothness 1.0 \
  --image-regularization 0.5 \
  --black-preservation 1.0 \
  --lut-smoothness 1.0 \
  --output-path cinematic_lut.cube
```

### Model Options

- `gemma3_4b`: Gemma 3 4B (fastest, good quality)
- `gemma3_12b`: Gemma 3 12B (balanced - recommended)
- `gemma3_27b`: Gemma 3 27B (most capable, slowest)

All Gemma models use context-aware evaluation by default.

### When to Use Gemma 3 vs CLIP

**Gemma 3 Models (Context-Aware):**
- Diverse input images with different lighting, colors, styles
- Want appropriate transformations relative to input
- Prevent excessive color shifts
- More natural-looking results that preserve input characteristics
- Better handling of edge cases (e.g., images already close to target)

**CLIP (Single Image Assessment):**
- All input images are similar in style
- Want the absolute best match to the prompt
- Faster inference (no need to process two images)
- Simpler transformations

## Technical Details

### How It Works

Gemma 3 models use context-aware evaluation:
- VLM sees: `[original_image, transformed_image]`
- Evaluates: "Was the color grade successfully applied?"
- Optimizes: Make the transformation appropriate for the input

### Memory Considerations

Gemma 3 processes **2 images per example** (original + transformed), which:
- Uses approximately **2x GPU memory**
- Takes slightly longer per iteration

If you encounter memory issues:
1. Reduce `--batch-size` (e.g., from 4 to 2)
2. The VLM model automatically resizes images to 896x896

### Gradient Flow

Full differentiability is maintained through the LUT transformation:
- Gradients flow from VLM → transformed image → LUT parameters
- Original images are constant (no gradients flow through them)
- Only the LUT parameters are optimized

## Expected Results

When using Gemma 3 models, you should observe:

1. **More conservative transformations** - The VLM considers whether the change is appropriate
2. **Better preservation of input characteristics** - Natural lighting and composition maintained
3. **Consistent style application** - Similar transformation strength across diverse inputs
4. **Smoother convergence** - The VLM has more context to provide useful gradient signals

## Troubleshooting

### High Memory Usage
- Reduce `--batch-size` to 2 or 1
- Gemma 3 models require more VRAM due to processing two images
- Ensure you're using GPU (CUDA) if available

### Slow Training
- Use smaller model: `gemma3_4b` instead of `gemma3_12b` or `gemma3_27b`
- Reduce batch size
- Gemma 3 is slower than CLIP but produces better context-aware results

## Implementation Notes

Context-aware evaluation is implemented by:
1. VLMLoss always uses two image tokens in the prompt
2. Processing original and transformed images together as pairs
3. Automatically detecting VLM models in `compute_losses()` and passing both images
4. CLIP models continue to work with single-image assessment

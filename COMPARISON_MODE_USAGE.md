# VLM Comparison Mode Usage Guide

## Overview

The VLM loss now supports two modes:

1. **Assessment Mode** (default): Evaluates if a single image has the desired color grade
   - Question: "Does this image have the color grade or look of '{prompt}'?"

2. **Comparison Mode** (new): Evaluates if the transformation from original to transformed image correctly applies the color grade
   - Question: "Looking at these two images, has the '{prompt}' color grade been successfully applied to transform the first image into the second?"

## Why Use Comparison Mode?

Comparison mode provides **better contextual understanding** by:

- **Relative assessment**: Judges the transformation relative to the starting point, not just the final result
- **Prevents over-transformation**: Can identify when a transformation is too aggressive or not appropriate for the input
- **Handles diverse inputs**: Works better when input images vary widely (some already close to target, others far away)
- **Transformation awareness**: Explicitly evaluates the change applied, not just whether the output matches the prompt

## Usage

### Basic Usage

To enable comparison mode, add the `--vlm-comparison-mode` flag when using VLM:

```bash
python main.py optimize \
  --prompt "warm golden hour" \
  --image-folder ./images \
  --model-type vlm \
  --vlm-comparison-mode
```

### Full Example with All Options

```bash
python main.py optimize \
  --prompt "cinematic teal and orange" \
  --image-folder ./dataset \
  --model-type vlm \
  --vlm-comparison-mode \
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

### Comparison: Assessment vs Comparison Mode

#### Assessment Mode (default)
```bash
# Asks: "Does this image have the color grade or look of 'warm golden hour'?"
python main.py optimize \
  --prompt "warm golden hour" \
  --image-folder ./images \
  --model-type vlm
```

**Best for:**
- When all input images are similar in style
- When you want the absolute best match to the prompt regardless of input
- Simpler cases where transformation magnitude doesn't matter

#### Comparison Mode (new)
```bash
# Asks: "Has the 'warm golden hour' color grade been successfully applied?"
python main.py optimize \
  --prompt "warm golden hour" \
  --image-folder ./images \
  --model-type vlm \
  --vlm-comparison-mode
```

**Best for:**
- Diverse input images (different lighting, colors, styles)
- When you want appropriate transformation relative to input
- Preventing excessive color shifts
- More natural-looking results that preserve input characteristics

## Technical Details

### How It Works

**Assessment Mode:**
- VLM sees: `[transformed_image]`
- Evaluates: "Does this look like {prompt}?"
- Optimizes: Make the output match the prompt

**Comparison Mode:**
- VLM sees: `[original_image, transformed_image]`
- Evaluates: "Was the transformation correct?"
- Optimizes: Make the transformation appropriate for the input

### Memory Considerations

Comparison mode processes **2 images per example** instead of 1, which:
- Uses approximately **2x GPU memory**
- Takes slightly longer per iteration

If you encounter memory issues:
1. Reduce `--batch-size` (e.g., from 4 to 2)
2. The VLM model automatically resizes images to 896x896

### Gradient Flow

Both modes maintain full differentiability through the LUT transformation, allowing:
- Gradients flow from VLM → transformed image → LUT parameters
- In comparison mode, gradients flow through the transformed image only (original is constant)

## Example Results

When using comparison mode, you should observe:

1. **More conservative transformations** - The VLM considers whether the change is appropriate
2. **Better preservation of input characteristics** - Natural lighting and composition maintained
3. **Consistent style application** - Similar transformation strength across diverse inputs
4. **Smoother convergence** - The VLM has more context to provide useful gradient signals

## Troubleshooting

### "original_images must be provided" Error
- This error occurs if comparison_mode is enabled but original images aren't passed
- This is automatically handled in main.py - if you see this, there may be a bug

### High Memory Usage
- Reduce `--batch-size` to 2 or 1
- Ensure you're using GPU (CUDA) if available

### Comparison Mode Ignored Warning
- You'll see this if you use `--vlm-comparison-mode` with `--model-type clip`
- Comparison mode only works with VLM, not CLIP

## Implementation Notes

The comparison mode is implemented by:
1. Modifying `VLMLoss` class to accept a `comparison_mode` parameter
2. Using two image tokens in the prompt when in comparison mode
3. Processing original and transformed images together
4. Automatically detecting comparison mode in `compute_losses()` and passing both images

The implementation maintains backward compatibility - existing code continues to work with default assessment mode.

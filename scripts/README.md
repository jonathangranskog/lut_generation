# LUT Generation Scripts

This folder contains reference files and batch generation tools for creating large libraries of LUTs.

## Reference Files

### Color LUTs
- **colors.txt** - 100+ colors, combinations, and qualities
- **emotions.txt** - 270+ emotional and atmospheric adjectives
- **film_formats.txt** - Color film stocks (Kodak, Fuji, specialty)
- **movies.txt** - Famous films with iconic color grades
- **directors.txt** - Filmmakers with distinctive visual styles

### Black & White LUTs (use with `--grayscale`)
- **film_formats_bw.txt** - B&W film stocks (Tri-X, HP5, Delta, etc.)
- **movies_bw.txt** - Classic and modern B&W cinema
- **directors_bw.txt** - Directors known for monochrome work

## Batch Generation Script

The `generate_luts.py` script intelligently combines references to create sensible prompts.

### Combination Strategy

**Sensible combinations:**
- Film stock + emotion → `"Kodak Portra 400 nostalgic"`
- Film stock + color → `"Fuji Velvia 50 warm saturated"`
- Emotion + color → `"melancholic blue tones"`
- Movies standalone → `"Blade Runner 2049"`
- Directors standalone → `"Wes Anderson"`
- Colors standalone → `"teal and orange"`
- Emotions standalone → `"cinematic moody"`

**Avoided combinations:**
- Movie + director (redundant)
- Movie + film stock (conflicting aesthetics)
- Director + film stock (potentially confusing)

### Usage Examples

#### Generate 100 random LUTs
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --sample 100 \
  --output-dir luts/
```

#### Generate all movie-based LUTs
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --standalone-only \
  --output-dir luts/movies/
```

#### Generate only B&W LUTs
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --bw-only \
  --sample 50 \
  --output-dir luts/bw/
```

#### Generate color combination LUTs only
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --color-only \
  --sample 200 \
  --output-dir luts/color_combos/
```

#### Dry run to preview prompts
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --sample 10 \
  --dry-run
```

#### Use a different model (VLM)
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --model-type gemma3_12b \
  --batch-size 1 \
  --sample 50 \
  --output-dir luts/vlm/
```

#### High-quality, slow generation
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --model-type sds \
  --steps 1000 \
  --batch-size 1 \
  --sample 25 \
  --output-dir luts/high_quality/
```

### Command-Line Options

#### Required
- `--image-folder PATH` - Folder containing training images

#### Output
- `--output-dir PATH` - Where to save LUTs (default: `luts/`)

#### Generation Modes
- `--sample N` - Generate N random LUTs
- `--color-only` - Only color combinations (film+emotion, film+color, etc.)
- `--bw-only` - Only black & white LUTs
- `--standalone-only` - Only movies and directors

#### Model Parameters
- `--model-type {clip,gemma3_4b,gemma3_12b,gemma3_27b,sds}` - Model (default: clip)
- `--steps N` - Training iterations (default: 500)
- `--lut-size N` - LUT resolution (default: 16)
- `--batch-size N` - Batch size (default: 4)

#### Other
- `--dry-run` - Preview without generating
- `--seed N` - Random seed for reproducibility

### Estimated Generation Counts

Without sampling, the script would generate:

**Color combinations:**
- Film + emotion: ~26,000 prompts
- Film + color: ~14,000 prompts
- Emotion + color: ~39,000 prompts
- Standalone: ~500 prompts
- **Total: ~80,000+ color LUTs**

**B&W combinations:**
- Film + emotion: ~16,000 prompts
- Standalone films: ~60 prompts
- B&W emotions: ~270 prompts
- Movies: ~90 prompts
- Directors: ~50 prompts
- **Total: ~16,500+ B&W LUTs**

**Grand total: 96,000+ possible LUTs**

Use `--sample` to generate a manageable subset!

### Practical Recommendations

**Quick diverse library (1-2 hours):**
```bash
python scripts/generate_luts.py \
  --image-folder images/ \
  --sample 100 \
  --output-dir luts/ \
  --seed 42
```

**Curated high-quality collection (overnight):**
```bash
# Movies and directors only
python scripts/generate_luts.py \
  --image-folder images/ \
  --standalone-only \
  --output-dir luts/curated/

# Then add some film stock combos
python scripts/generate_luts.py \
  --image-folder images/ \
  --color-only \
  --sample 200 \
  --output-dir luts/curated/
```

**Production library (days/weeks):**
```bash
# Generate in batches
python scripts/generate_luts.py \
  --image-folder images/ \
  --sample 1000 \
  --output-dir luts/batch1/ \
  --seed 1

python scripts/generate_luts.py \
  --image-folder images/ \
  --sample 1000 \
  --output-dir luts/batch2/ \
  --seed 2
```

## Adding Custom References

To expand the library, simply add new lines to any `.txt` file:

```bash
echo "sunset orange glow" >> scripts/colors.txt
echo "Kodak Vision3 5219" >> scripts/film_formats.txt
echo "euphoric neon dreamscape" >> scripts/emotions.txt
```

Avoid lines starting with `#` (treated as comments).

## File Naming

Generated LUT files are automatically named based on the prompt:
- Prompt: `"Kodak Portra 400 nostalgic"` → `kodak_portra_400_nostalgic.cube`
- Prompt: `"Blade Runner 2049"` → `blade_runner_2049.cube`

---

## Utility LUT Generation Script

The `generate_utility_luts.py` script creates technical adjustment LUTs using fixed mathematical transformations. Unlike AI-guided LUTs, these apply precise, predictable color adjustments.

### Available Transformations

**Saturation** (uses torchvision):
- `desaturate_25`, `desaturate_50`, `desaturate_75`
- `oversaturate_125`, `oversaturate_150`, `oversaturate_200`

**Contrast** (uses torchvision):
- `low_contrast_50`, `low_contrast_75`
- `high_contrast_125`, `high_contrast_150`, `high_contrast_200`

**Brightness** (uses torchvision):
- `brightness_50`, `brightness_75`
- `brightness_125`, `brightness_150`, `brightness_200`

**Exposure** (stops):
- `exposure_minus_2`, `exposure_minus_1`, `exposure_minus_half`
- `exposure_plus_half`, `exposure_plus_1`, `exposure_plus_2`

**Gamma**:
- `gamma_0_5`, `gamma_0_75`, `gamma_1_25`
- `gamma_1_5`, `gamma_2_0`, `gamma_2_2` (sRGB standard)

**Hue Shift** (degrees, uses torchvision):
- `hue_shift_30`, `hue_shift_60`, `hue_shift_90`
- `hue_shift_120`, `hue_shift_180`, `hue_shift_240`, `hue_shift_300`

**Temperature**:
- `cool_slight`, `cool_moderate`, `cool_strong`
- `warm_slight`, `warm_moderate`, `warm_strong`

**Tint**:
- `tint_magenta_slight`, `tint_magenta_moderate`, `tint_magenta_strong`
- `tint_green_slight`, `tint_green_moderate`, `tint_green_strong`

**Grayscale**:
- `grayscale_rec709` (Rec. 709 luminance coefficients)
- `grayscale_identity` (single-channel identity LUT)

### Usage Examples

#### Generate all utility LUTs
```bash
python scripts/generate_utility_luts.py --output-dir luts/utility/
```

#### Generate only saturation adjustments
```bash
python scripts/generate_utility_luts.py --saturation-only --output-dir luts/
```

#### Generate with high resolution
```bash
python scripts/generate_utility_luts.py --lut-size 64 --output-dir luts/utility/
```

#### Preview without generating
```bash
python scripts/generate_utility_luts.py --dry-run
```

### Command-Line Options

```
--output-dir PATH        Output directory (default: luts/utility/)
--lut-size N            LUT resolution (default: 32)

# Category filters (generate only specific types)
--saturation-only
--contrast-only
--brightness-only
--exposure-only
--gamma-only
--hue-only
--temperature-only
--tint-only
--grayscale-only

--dry-run               Preview without generating
```

### Use Cases

**Quick technical adjustments:**
- Apply `oversaturate_150.cube` for vibrant product photos
- Use `gamma_2_2.cube` for sRGB gamma correction
- Apply `cool_moderate.cube` for a subtle blue tone

**Stacking LUTs:**
Many video editing tools allow LUT stacking. Combine utility LUTs with AI-generated ones:
1. Apply `exposure_plus_half.cube` to brighten
2. Then apply `blade_runner_2049.cube` for the look
3. Finally `desaturate_25.cube` to taste

**Baseline corrections:**
Use utility LUTs to prepare footage before applying creative LUTs:
- `gamma_2_2.cube` for gamma standardization
- `exposure_minus_1.cube` to bring down hot highlights
- `warm_slight.cube` to counteract cool camera sensors

### Total LUTs Generated

Running with `--output-dir luts/utility/` generates **~43 utility LUTs**:
- 6 saturation
- 5 contrast
- 5 brightness
- 6 exposure
- 6 gamma
- 7 hue
- 6 temperature
- 6 tint
- 2 grayscale

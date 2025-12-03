"""Constants used throughout the LUT generation codebase."""

# CLIP model constants
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# VLM model constants (Gemma 3)
VLM_IMAGE_SIZE = 448  # Balance between quality and memory/speed (native is 896)

# Rec. 709 luma coefficients for RGB to luminance conversion
# Y = 0.2126*R + 0.7152*G + 0.0722*B
REC709_LUMA_R = 0.2126
REC709_LUMA_G = 0.7152
REC709_LUMA_B = 0.0722

# DeepFloyd IF constants for Score Distillation Sampling (SDS)
# Stage I UNet operates at 64x64 pixel resolution, but we sample higher-res crops
# from the dataset to preserve scene context, then downsample before the UNet
DEEPFLOYD_UNET_SIZE = 64  # Native UNet resolution
DEEPFLOYD_IMAGE_SIZE = 256  # Dataset crop size (higher res for context)
DEEPFLOYD_STAGE1_MODEL = "DeepFloyd/IF-I-XL-v1.0"  # 4.3B params
DEEPFLOYD_STAGE1_MODEL_MEDIUM = "DeepFloyd/IF-I-M-v1.0"  # Smaller variant

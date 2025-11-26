"""Constants used throughout the LUT generation codebase."""

# CLIP model constants
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Rec. 709 luma coefficients for RGB to luminance conversion
# Y = 0.2126*R + 0.7152*G + 0.0722*B
REC709_LUMA_R = 0.2126
REC709_LUMA_G = 0.7152
REC709_LUMA_B = 0.0722

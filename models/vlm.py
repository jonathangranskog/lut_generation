"""
VLM-based loss for LUT optimization.

Uses a Vision-Language Model (Gemma 3 4B) to compute a loss between
transformed images and a text prompt via question-answering.

Adapted from Dual-Process Image Generation:
https://github.com/g-luo/dual_process
"""

import logging

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from models.base import LUTLoss

logger = logging.getLogger(__name__)

# Gemma 3 prompt template
GEMMA_IMAGE_TOKEN = "<start_of_image>"
GEMMA_TEMPLATE = (
    "<bos><start_of_turn>user\n"
    "You are a helpful assistant.\n\n"
    "{image_token}{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Model type to HuggingFace model name mapping
MODEL_NAME_MAP = {
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
}


class VLMLoss(LUTLoss):
    """
    VLM-based loss for image-text alignment via contrastive Yes/No scoring.

    Uses comparison mode: evaluates if the color grade has been successfully applied
    to transform the original image into the transformed image. This provides better
    context-aware gradients than assessing the final image alone.

    Computes a contrastive loss based on log P(Yes) - log P(No). This normalizes out
    model biases and provides a cleaner gradient signal than just optimizing for P(Yes).
    """

    def __init__(
        self,
        prompt: str,
        model_name: str = "gemma3_12b",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        question_template: str | None = None,
    ):
        """
        Initialize VLM loss.

        Args:
            prompt: Text prompt describing desired image style (e.g., "warm golden hour")
            model_name: Model identifier - either a model type key (gemma3_4b, gemma3_12b, gemma3_27b)
                       or a full HuggingFace model identifier (e.g., "google/gemma-3-12b-it")
            device: Device to run the model on
            dtype: Model dtype (None = auto-select: bfloat16 for CUDA, float32 for CPU)
            question_template: Custom question template (None = use default comparison question)
        """
        super().__init__()

        self.device = device
        # Auto-select dtype: bfloat16 for CUDA (faster, less memory), float32 for CPU
        if dtype is None:
            self.dtype = torch.bfloat16 if device != "cpu" else torch.float32
        else:
            self.dtype = dtype
        self.prompt = prompt

        # Set default question template (comparison mode)
        if question_template is None:
            question_template = f"Looking at these two images, has the '{prompt}' color grade been successfully applied to transform the first image into the second? Answer Yes or No."
            self.question = question_template
        else:
            # Custom template may have {prompt} placeholder
            self.question = question_template.replace("{prompt}", prompt)

        # Map model type to HuggingFace identifier if needed
        if model_name in MODEL_NAME_MAP:
            hf_model_name = MODEL_NAME_MAP[model_name]
            logger.info(f"Loading VLM model: {model_name} ({hf_model_name})")
        else:
            hf_model_name = model_name
            logger.info(f"Loading VLM model: {hf_model_name}")

        if device == "cpu":
            logger.info("  Warning: Running VLM on CPU will be very slow (~minutes per step)")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            hf_model_name,
            device_map=device,
            dtype=self.dtype,
        )
        self.processor = AutoProcessor.from_pretrained(hf_model_name)

        # Disable image splitting for simpler processing
        if hasattr(self.processor.image_processor, "do_image_splitting"):
            self.processor.image_processor.do_image_splitting = False

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Get image dimensions from processor
        processor_size = getattr(
            self.processor.image_processor, "size", {"height": 896, "width": 896}
        )
        self.image_size = (processor_size["height"], processor_size["width"])

        # Pre-compute prompt tokens and Yes/No token IDs
        self._prepare_prompt()

        logger.info(f"VLM initialized for prompt: '{prompt}'")
        logger.info(f"  Question: {self.question}")
        logger.info(f"  Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}")

    def _prepare_prompt(self):
        """Pre-compute input_ids and Yes/No token IDs for comparison mode (2 images)."""
        # Format the prompt with question using two image tokens
        image_token = GEMMA_IMAGE_TOKEN + GEMMA_IMAGE_TOKEN
        num_images = 2

        text = (
            f"<bos><start_of_turn>user\n"
            f"You are a helpful assistant.\n\n"
            f"{image_token}{self.question}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        # Create blank images for tokenization (will be replaced during forward)
        blank_images = [
            Image.new("RGB", self.image_size, color="white") for _ in range(num_images)
        ]

        # Tokenize the prompt
        inputs = self.processor(
            text=text,
            images=blank_images,
            return_tensors="pt",
        )

        # Store for forward pass
        self.input_ids = inputs["input_ids"].to(self.device)

        # Get token IDs for "Yes" and "No"
        # Tokenize with space prefix to get the standalone token
        yes_tokens = self.processor.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.processor.tokenizer.encode("No", add_special_tokens=False)

        # Use the first token (main token for Yes/No)
        self.yes_token_id = yes_tokens[0]
        self.no_token_id = no_tokens[0]

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for VLM using differentiable operations.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Preprocessed pixel values ready for VLM
        """
        # Resize to VLM expected size
        if images.shape[-2:] != self.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )

        # Normalize using processor's image_mean and image_std
        mean = torch.tensor(self.processor.image_processor.image_mean).to(images.device)
        std = torch.tensor(self.processor.image_processor.image_std).to(images.device)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)

        normalized = (images - mean) / std

        return normalized

    def forward(
        self, transformed_images: torch.Tensor, original_images: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute contrastive VLM loss between original and transformed images.

        Uses log P(No) - log P(Yes) as the loss, which:
        - Normalizes out model biases (if model always favors Yes, it cancels out)
        - Provides cleaner gradients that depend on relative probabilities
        - Lower loss = model more confident transformation matches prompt

        Args:
            transformed_images: Batch of transformed images in [0, 1] range, shape (B, C, H, W)
            original_images: Batch of original images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Scalar loss value (log P(No) - log P(Yes), lower is better)
        """
        if original_images is None:
            raise ValueError("VLMLoss requires original_images for comparison mode")

        batch_size = transformed_images.shape[0]

        # Preprocess both image sets (differentiable)
        original_pixel_values = self.preprocess_images(original_images).to(self.dtype)
        transformed_pixel_values = self.preprocess_images(transformed_images).to(self.dtype)

        # Interleave original and transformed images for the batch
        # Shape: (B*2, C, H, W) where pairs are [orig_0, trans_0, orig_1, trans_1, ...]
        pixel_values = torch.stack(
            [original_pixel_values, transformed_pixel_values], dim=1
        )
        pixel_values = pixel_values.view(batch_size * 2, *pixel_values.shape[2:])

        # Expand input_ids for batch (each pair shares the same prompt)
        input_ids = self.input_ids.expand(batch_size, -1)

        # Run VLM forward pass
        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

        # Get logits for the last position (where Yes/No would be predicted)
        # Shape: (batch_size, vocab_size)
        last_logits = outputs.logits[:, -1, :]

        # Get log probabilities for Yes and No tokens
        log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
        log_p_yes = log_probs[:, self.yes_token_id]
        log_p_no = log_probs[:, self.no_token_id]

        # Contrastive loss: we want P(Yes) > P(No)
        # Loss = log P(No) - log P(Yes) = log(P(No)/P(Yes))
        # When P(Yes) > P(No), loss is negative (good)
        # When P(No) > P(Yes), loss is positive (bad)
        loss = (log_p_no - log_p_yes).mean()

        return loss

    def compute_probability(
        self, images: torch.Tensor, original_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute probability of Yes answer (higher is better).

        Args:
            images: Batch of transformed images in [0, 1] range, shape (B, C, H, W)
            original_images: Batch of original images in [0, 1] range, shape (B, C, H, W)

        Returns:
            P(Yes) for each image in batch
        """
        batch_size = images.shape[0]

        with torch.no_grad():
            original_pixel_values = self.preprocess_images(original_images).to(
                self.dtype
            )
            transformed_pixel_values = self.preprocess_images(images).to(self.dtype)
            pixel_values = torch.stack(
                [original_pixel_values, transformed_pixel_values], dim=1
            )
            pixel_values = pixel_values.view(batch_size * 2, *pixel_values.shape[2:])
            input_ids = self.input_ids.expand(batch_size, -1)

            outputs = self.model.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )

            last_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            p_yes = probs[:, self.yes_token_id]

        return p_yes

    def get_yes_no_probs(
        self, images: torch.Tensor, original_images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get both Yes and No probabilities for debugging.

        Args:
            images: Batch of transformed images in [0, 1] range, shape (B, C, H, W)
            original_images: Batch of original images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Tuple of (P(Yes), P(No)) tensors
        """
        batch_size = images.shape[0]

        with torch.no_grad():
            original_pixel_values = self.preprocess_images(original_images).to(
                self.dtype
            )
            transformed_pixel_values = self.preprocess_images(images).to(self.dtype)
            pixel_values = torch.stack(
                [original_pixel_values, transformed_pixel_values], dim=1
            )
            pixel_values = pixel_values.view(batch_size * 2, *pixel_values.shape[2:])
            input_ids = self.input_ids.expand(batch_size, -1)

            outputs = self.model.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )

            last_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            p_yes = probs[:, self.yes_token_id]
            p_no = probs[:, self.no_token_id]

        return p_yes, p_no

    def get_prediction(
        self, images: torch.Tensor, original_images: torch.Tensor
    ) -> list[str]:
        """
        Get the model's actual predicted answer for debugging.

        Args:
            images: Batch of transformed images in [0, 1] range, shape (B, C, H, W)
            original_images: Batch of original images in [0, 1] range, shape (B, C, H, W)

        Returns:
            List of "Yes" or "No" predictions
        """
        p_yes, p_no = self.get_yes_no_probs(images, original_images)
        predictions = ["Yes" if y > n else "No" for y, n in zip(p_yes, p_no)]
        return predictions


if __name__ == "__main__":
    # Test the VLM loss with gradient flow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create loss function
    vlm_loss = VLMLoss(prompt="kodak aerochrome infrared film", device=device)
    logger.info(f"Using dtype: {vlm_loss.dtype}")

    # Create dummy batch of images with gradient tracking
    batch_size = 1
    dummy_images = torch.rand(
        batch_size, 3, 512, 512, device=device, requires_grad=True
    )

    # Compute loss
    loss = vlm_loss(dummy_images)
    logger.info(f"\nContrastive Loss: {loss.item():.4f}")
    logger.info("  (negative = Yes more likely, positive = No more likely)")

    # Get probabilities
    p_yes, p_no = vlm_loss.get_yes_no_probs(dummy_images)
    logger.info(f"P(Yes): {p_yes.item():.4f}, P(No): {p_no.item():.4f}")

    pred = vlm_loss.get_prediction(dummy_images)
    logger.info(f"Prediction: {pred}")

    # Test gradient flow
    loss.backward()
    logger.info(f"\nGradients flowing: {dummy_images.grad is not None}")
    if dummy_images.grad is not None:
        logger.info(f"Gradient magnitude: {dummy_images.grad.abs().mean().item():.6f}")

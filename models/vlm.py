"""
VLM-based loss for LUT optimization.

Uses a Vision-Language Model (Gemma 3 4B) to compute a loss between
transformed images and a text prompt via question-answering.

Adapted from Dual-Process Image Generation:
https://github.com/g-luo/dual_process
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

IGNORE_INDEX = -100

# Gemma 3 prompt template
GEMMA_IMAGE_TOKEN = "<start_of_image>"
GEMMA_TEMPLATE = (
    "<bos><start_of_turn>user\n"
    "You are a helpful assistant.\n\n"
    "{image_token}{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


class VLMLoss(nn.Module):
    """
    VLM-based loss for image-text alignment via question-answering.

    Asks the VLM "Does this image have {prompt}?" and optimizes for "Yes" answer.
    Uses cross-entropy loss on the answer tokens.
    """

    def __init__(
        self,
        prompt: str,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        question_template: str = "Does the color grade of this image match the following prompt '{prompt}'?",
        answer: str = "Yes",
    ):
        """
        Initialize VLM loss.

        Args:
            prompt: Text prompt describing desired image style (e.g., "warm golden hour")
            model_name: HuggingFace model identifier for Gemma 3
            device: Device to run the model on
            dtype: Model dtype (None = auto-select: bfloat16 for CUDA, float32 for CPU)
            question_template: Template for the question, must contain {prompt}
            answer: Expected answer to optimize towards (default: "Yes")
        """
        super().__init__()

        self.device = device
        # Auto-select dtype: bfloat16 for CUDA (faster, less memory), float32 for CPU
        if dtype is None:
            self.dtype = torch.bfloat16 if device != "cpu" else torch.float32
        else:
            self.dtype = dtype
        self.prompt = prompt
        self.answer = answer
        self.question = question_template.format(prompt=prompt)

        # Load VLM model and processor
        print(f"Loading VLM model: {model_name}")
        if device == "cpu":
            print("  Warning: Running VLM on CPU will be very slow (~minutes per step)")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            dtype=self.dtype,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

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

        # Pre-compute prompt tokens and answer mask
        self._prepare_prompt()

        print(f"VLM initialized for prompt: '{prompt}'")
        print(f"  Question: {self.question}")
        print(f"  Answer: {self.answer}")

    def _prepare_prompt(self):
        """Pre-compute input_ids and answer token mask."""
        # Format the full prompt with question
        prefix = GEMMA_TEMPLATE.format(
            image_token=GEMMA_IMAGE_TOKEN,
            question=self.question,
        )

        # Create a blank image for tokenization (will be replaced during forward)
        blank_image = Image.new("RGB", self.image_size, color="white")

        # Tokenize prefix only (without answer)
        prefix_inputs = self.processor(
            text=prefix,
            images=[blank_image],
            return_tensors="pt",
        )
        prefix_ids = prefix_inputs["input_ids"]

        # Tokenize full prompt (prefix + answer)
        full_text = prefix + self.answer
        full_inputs = self.processor(
            text=full_text,
            images=[blank_image],
            return_tensors="pt",
        )
        full_ids = full_inputs["input_ids"]

        # Find answer token indices (where prefix and full differ, plus any new tokens)
        # This handles cases where the answer token might modify the prefix tokenization
        min_len = min(prefix_ids.shape[1], full_ids.shape[1])
        search_idxs = (
            (prefix_ids[:, :min_len] != full_ids[:, :min_len])
            .nonzero(as_tuple=True)[1]
            .tolist()
        )
        search_idxs += list(range(prefix_ids.shape[1], full_ids.shape[1]))

        # Store for forward pass
        self.input_ids = full_ids.to(self.device)
        self.answer_token_idxs = search_idxs

        # Create answer mask
        self.answer_mask = torch.zeros(full_ids.shape[1], dtype=torch.bool)
        self.answer_mask[search_idxs] = True

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute VLM loss between images and text prompt.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Scalar loss value (cross-entropy on answer tokens)
        """
        batch_size = images.shape[0]

        # Preprocess images (differentiable)
        pixel_values = self.preprocess_images(images)
        pixel_values = pixel_values.to(self.dtype)

        # Expand input_ids for batch
        input_ids = self.input_ids.expand(batch_size, -1)

        # Run VLM forward pass
        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

        # Compute cross-entropy loss on answer tokens only
        logits = outputs.logits

        # Create labels: IGNORE_INDEX everywhere except answer tokens
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, self.answer_mask] = input_ids[:, self.answer_mask].detach().clone()

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        return loss

    def compute_probability(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute probability of the expected answer (higher is better).

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Probability score derived from loss
        """
        with torch.no_grad():
            loss = self.forward(images)
            # Convert cross-entropy to probability
            prob = (-loss).exp()
        return prob

    def get_prediction(self, images: torch.Tensor) -> list[str]:
        """
        Get the model's actual predicted answer for debugging.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            List of predicted answer strings
        """
        batch_size = images.shape[0]

        with torch.no_grad():
            pixel_values = self.preprocess_images(images)
            pixel_values = pixel_values.to(self.dtype)
            input_ids = self.input_ids.expand(batch_size, -1)

            outputs = self.model.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )

            # Get predicted tokens at answer positions
            logits = outputs.logits
            # Shift to align with next-token prediction
            answer_mask_shifted = self.answer_mask[1:]
            preds = logits[:, :-1][:, answer_mask_shifted].argmax(dim=-1)
            predictions = self.processor.tokenizer.batch_decode(
                preds, skip_special_tokens=True
            )

        return predictions


if __name__ == "__main__":
    # Test the VLM loss with gradient flow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create loss function
    vlm_loss = VLMLoss(prompt="warm golden hour", device=device)
    print(f"Using dtype: {vlm_loss.dtype}")

    # Create dummy batch of images with gradient tracking
    batch_size = 1
    dummy_images = torch.rand(batch_size, 3, 512, 512, requires_grad=True).to(device)

    # Compute loss
    loss = vlm_loss(dummy_images)
    print(f"\nInitial Loss: {loss.item():.4f}")

    # Get probability and prediction
    prob = vlm_loss.compute_probability(dummy_images)
    print(f"Probability: {prob.item():.4f}")

    preds = vlm_loss.get_prediction(dummy_images)
    print(f"Predicted answer: {preds}")

    # Test gradient flow
    loss.backward()
    print(f"Gradients flowing: {dummy_images.grad is not None}")
    if dummy_images.grad is not None:
        print(f"Gradient magnitude: {dummy_images.grad.abs().mean().item():.6f}")

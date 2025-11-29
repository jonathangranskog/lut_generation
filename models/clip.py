"""
CLIP-based loss for LUT optimization.

Uses CLIP to compute a loss between transformed images and a text prompt.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

from utils.constants import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD


class CLIPLoss(nn.Module):
    """
    CLIP-based cosine similarity loss for image-text alignment.

    Computes 1 - cosine_similarity between CLIP image embeddings and text embeddings.
    Loss range: [0, 2], typically [0, 1]. Lower loss = higher similarity.
    """

    def __init__(
        self,
        prompt: str,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        """
        Initialize CLIP loss.

        Args:
            prompt: Text prompt to optimize towards
            model_name: HuggingFace model identifier for CLIP
            device: Device to run the model on
        """
        super().__init__()

        self.device = device
        self.prompt = prompt

        # Load CLIP model and tokenizer
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # Freeze CLIP parameters - we're not training CLIP
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Pre-compute and cache text embeddings
        with torch.no_grad():
            text_inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            text_features = self.model.get_text_features(**text_inputs)
            # Normalize text embedding for cosine similarity
            self.text_embedding = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        print(f"Text embedding computed for prompt: '{prompt}'")

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP using differentiable operations.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Preprocessed images ready for CLIP, with gradients preserved
        """
        # CLIP's normalization values (from OpenAI's CLIP preprocessing)
        # https://github.com/openai/CLIP/blob/main/clip/clip.py
        mean = torch.tensor(CLIP_MEAN).to(images.device)
        std = torch.tensor(CLIP_STD).to(images.device)

        # Reshape for broadcasting: (1, 3, 1, 1)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)

        # Resize to CLIP's expected input size if needed
        if images.shape[-2:] != (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE):
            images = torch.nn.functional.interpolate(
                images,
                size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        # Normalize: (x - mean) / std
        normalized = (images - mean) / std

        return normalized

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP cosine similarity loss between images and text prompt.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Scalar loss value (1 - mean cosine similarity across batch)
        """
        # Preprocess images for CLIP (differentiable)
        processed_images = self.preprocess_images(images)

        # Get image embeddings (gradients flow through images, not CLIP params)
        image_features = self.model.get_image_features(pixel_values=processed_images)

        # Normalize embeddings (required for cosine similarity)
        image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity (dot product of normalized vectors)
        # Shape: (B, D) * (1, D) -> (B,)
        cosine_sim = (image_embeddings * self.text_embedding).sum(dim=-1)

        # Convert to loss: 1 - similarity (range: [0, 2], typically [0, 1])
        # Lower is better (high similarity = low loss)
        loss = 1.0 - cosine_sim

        # Return mean loss across batch
        return loss.mean()

    def compute_similarity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between images and text (higher is better).

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Cosine similarity scores, shape (B,)
        """
        processed_images = self.preprocess_images(images)

        with torch.no_grad():
            image_features = self.model.get_image_features(
                pixel_values=processed_images
            )
            # Normalize image embeddings for cosine similarity
            image_embeddings = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        # Compute cosine similarity
        similarity = (image_embeddings * self.text_embedding).sum(dim=-1)

        return similarity


if __name__ == "__main__":
    # Test the CLIP loss with gradient flow
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create loss function
    clip_loss = CLIPLoss(prompt="a warm sunset photograph", device=device)

    # Create dummy batch of images with gradient tracking
    batch_size = 4
    dummy_images = torch.rand(batch_size, 3, 224, 224, requires_grad=True).to(device)

    # Compute loss
    loss = clip_loss(dummy_images)
    print(f"\nInitial Loss: {loss.item():.4f}")

    # Test gradient flow
    loss.backward()
    print(f"✓ Gradients flowing: {dummy_images.grad is not None}")
    print(f"  Gradient magnitude: {dummy_images.grad.abs().mean().item():.6f}")

    # Simulate a few optimization steps
    print("\nSimulating LUT optimization:")
    dummy_images = torch.rand(batch_size, 3, 224, 224, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([dummy_images], lr=0.1)

    for step in range(5):
        optimizer.zero_grad()
        loss = clip_loss(dummy_images)
        loss.backward()
        optimizer.step()

        # Clamp to valid range
        with torch.no_grad():
            dummy_images.clamp_(0, 1)

        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")

    # Compute final similarity
    with torch.no_grad():
        similarity = clip_loss.compute_similarity(dummy_images)
        print(f"\nFinal similarities: {similarity.cpu().numpy()}")
        print(f"Mean similarity: {similarity.mean().item():.4f}")

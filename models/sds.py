"""
Score Distillation Sampling (SDS) loss for LUT optimization using DeepFloyd IF.

Uses DeepFloyd IF Stage I (pixel-space diffusion model) to compute gradients
that guide the LUT optimization toward a text prompt.

DeepFloyd IF advantages for LUT optimization:
- Operates in pixel space (not latent space) - ideal for pixel-based LUTs
- T5-XXL text encoder provides superior language understanding
- Stage I operates at 64x64 which is computationally efficient

References:
- DreamFusion: https://dreamfusion3d.github.io/
- DeepFloyd IF: https://github.com/deep-floyd/IF
"""

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, IFPipeline

from utils.constants import DEEPFLOYD_STAGE1_MODEL, DEEPFLOYD_UNET_SIZE


class SDSLoss(nn.Module):
    """
    Score Distillation Sampling loss using DeepFloyd IF.

    SDS gradient: ∇L = E[w(t) · (ε_θ(x_t, c, t) - ε) · ∂x/∂θ]

    The key insight is that we don't backprop through the diffusion model.
    Instead, we compute the noise prediction difference and use it as a
    pseudo-gradient to guide optimization.
    """

    def __init__(
        self,
        prompt: str,
        model_name: str = DEEPFLOYD_STAGE1_MODEL,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        guidance_scale: float = 50.0,
        min_timestep: int = 20,
        max_timestep: int = 800,
        use_medium_model: bool = False,
    ):
        """
        Initialize SDS loss with DeepFloyd IF.

        Args:
            prompt: Text prompt describing desired image style
            model_name: HuggingFace model identifier for DeepFloyd IF Stage I
            device: Device to run the model on
            dtype: Model dtype (None = auto-select: float16 for CUDA, float32 for CPU)
            guidance_scale: Classifier-free guidance scale (default 50.0)
            min_timestep: Minimum timestep for noise sampling (avoid very clean images)
            max_timestep: Maximum timestep for noise sampling (avoid pure noise)
            use_medium_model: Use smaller IF-I-M-v1.0 model (less VRAM, faster)
        """
        super().__init__()

        self.device = device
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep

        # Auto-select dtype
        if dtype is None:
            self.dtype = torch.float16 if device != "cpu" else torch.float32
        else:
            self.dtype = dtype

        # Use medium model if requested
        if use_medium_model:
            model_name = "DeepFloyd/IF-I-M-v1.0"

        print(f"Loading DeepFloyd IF model: {model_name}")
        print(f"  This requires accepting the license at: https://huggingface.co/{model_name}")
        print(f"  Device: {device}, dtype: {self.dtype}")

        # Load the IF pipeline
        # Note: This requires HF authentication and license acceptance
        self.pipe = IFPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None,  # Disable for optimization use
            watermarker=None,
        )

        # Move components to device
        self.pipe.to(device)

        # Get references to UNet and scheduler
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler

        # Freeze UNet parameters
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        # Use DDPM scheduler for adding noise (better for SDS than default)
        self.noise_scheduler = DDPMScheduler.from_config(self.scheduler.config)

        # Pre-compute and cache text embeddings
        print(f"Computing text embeddings for: '{prompt}'")
        with torch.no_grad():
            # Get both conditional and unconditional embeddings for CFG
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=True,
                num_images_per_prompt=1,
                device=device,
            )
            # Store embeddings
            self.prompt_embeds = prompt_embeds  # (1, seq_len, hidden_dim)
            self.negative_prompt_embeds = negative_prompt_embeds

        # Get alphas for noise scheduling
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)

        print(f"SDS initialized for prompt: '{prompt}'")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Timestep range: [{min_timestep}, {max_timestep}]")

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for DeepFloyd IF UNet.

        Images are typically sampled at higher resolution (e.g., 256x256) from
        the dataset to preserve scene context, then downsampled here to 64x64
        for the UNet.

        Note: We keep images in float32 for gradient computation. Only the UNet
        forward pass uses float16.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)

        Returns:
            Preprocessed images in [-1, 1] range at 64x64 resolution (float32)
        """
        # Resize to IF Stage I UNet resolution (64x64)
        # Input images may be higher resolution (e.g., 256x256) for context
        if images.shape[-2:] != (DEEPFLOYD_UNET_SIZE, DEEPFLOYD_UNET_SIZE):
            images = torch.nn.functional.interpolate(
                images,
                size=(DEEPFLOYD_UNET_SIZE, DEEPFLOYD_UNET_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Convert from [0, 1] to [-1, 1] (IF expects this range)
        # Keep in float32 for gradient computation
        images = images * 2.0 - 1.0

        return images

    def forward(
        self, images: torch.Tensor, original_images: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute SDS loss for a batch of images.

        The SDS gradient is:
        ∇L = E[w(t) · (ε_pred - ε)]

        We implement this as a pseudo-loss where the gradient flows through
        the input images but not through the UNet.

        Args:
            images: Batch of images in [0, 1] range, shape (B, C, H, W)
            original_images: Unused (for compatibility with VLMLoss signature)

        Returns:
            Scalar loss value
        """
        batch_size = images.shape[0]

        # Preprocess images to [-1, 1] at 64x64
        x0 = self.preprocess_images(images)

        # Sample random timesteps for each image in batch
        timesteps = torch.randint(
            self.min_timestep,
            self.max_timestep,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Sample noise
        noise = torch.randn_like(x0)

        # Add noise to images (forward diffusion process)
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        noisy_images = self.noise_scheduler.add_noise(x0, noise, timesteps)

        # Expand embeddings for batch and CFG
        # For CFG, we need to concatenate negative + positive embeddings
        prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
        negative_embeds = self.negative_prompt_embeds.expand(batch_size, -1, -1)

        # For classifier-free guidance, we run two forward passes
        # One with negative (unconditional) and one with positive (conditional)
        combined_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        combined_noisy = torch.cat([noisy_images, noisy_images], dim=0)
        combined_timesteps = torch.cat([timesteps, timesteps], dim=0)

        # Get noise prediction from UNet (no gradients through UNet)
        # Convert to model dtype (float16) for UNet, then back to float32
        with torch.no_grad():
            noise_pred = self.unet(
                combined_noisy.to(self.dtype),
                combined_timesteps,
                encoder_hidden_states=combined_embeds,
                return_dict=False,
            )[0]

        # DeepFloyd IF predicts both noise (first 3 channels) and variance (last 3 channels)
        # We only need the noise prediction for SDS
        # Convert back to float32 for gradient computation
        noise_pred = noise_pred[:, :3, :, :].float()

        # Split predictions for CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # Compute weighting factor w(t)
        # Common choice: w(t) = sigma_t^2 (related to SNR)
        # We use a simplified constant weighting here
        w = (1.0 - self.alphas_cumprod[timesteps]).float().view(-1, 1, 1, 1)

        # SDS gradient: (noise_pred - noise) * w
        # We create a pseudo-loss that has this gradient
        # grad = w * (noise_pred - noise)
        # loss = 0.5 * ||grad||^2 would give gradient = grad
        # But we want the gradient to be exactly grad, so we use:
        # loss = (noise_pred - noise).detach() * noisy_images * w
        # This gives ∂loss/∂x0 ∝ (noise_pred - noise) via chain rule through noisy_images

        # The trick: noisy_images = f(x0, noise, t)
        # ∂noisy_images/∂x0 = sqrt(alpha_t)
        # So we compute a pseudo-loss that when differentiated gives the SDS gradient

        grad = w * (noise_pred - noise)

        # Target for the loss (detached so no gradient flows through it)
        target = (noisy_images - grad).detach()

        # MSE loss between noisy_images and target
        # ∂loss/∂noisy_images = 2 * (noisy_images - target) = 2 * grad
        # This gradient then flows back to x0 (and thus to the LUT)
        loss = 0.5 * torch.nn.functional.mse_loss(noisy_images, target)

        return loss

    def compute_noise_prediction(
        self, images: torch.Tensor, timestep: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute noise prediction for visualization/debugging.

        Args:
            images: Batch of images in [0, 1] range
            timestep: Fixed timestep to use

        Returns:
            Tuple of (noise_pred, actual_noise)
        """
        batch_size = images.shape[0]
        x0 = self.preprocess_images(images)

        timesteps = torch.full(
            (batch_size,), timestep, device=self.device, dtype=torch.long
        )

        noise = torch.randn_like(x0)
        noisy_images = self.noise_scheduler.add_noise(x0, noise, timesteps)

        prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)

        with torch.no_grad():
            noise_pred = self.unet(
                noisy_images,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        return noise_pred, noise


if __name__ == "__main__":
    # Test the SDS loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("Warning: SDS on CPU will be very slow. Use CUDA if available.")

    # Note: This requires HuggingFace authentication and license acceptance
    # Run: huggingface-cli login
    # And accept the license at: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0

    try:
        sds_loss = SDSLoss(
            prompt="warm golden hour sunlight",
            device=device,
            use_medium_model=True,  # Use smaller model for testing
        )

        # Create dummy batch of images with gradient tracking
        batch_size = 2
        dummy_images = torch.rand(
            batch_size, 3, 256, 256, device=device, requires_grad=True
        )

        # Compute loss
        loss = sds_loss(dummy_images)
        print(f"\nSDS Loss: {loss.item():.6f}")

        # Test gradient flow
        loss.backward()
        print(f"Gradients flowing: {dummy_images.grad is not None}")
        if dummy_images.grad is not None:
            print(f"Gradient magnitude: {dummy_images.grad.abs().mean().item():.6f}")

    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTo use SDS with DeepFloyd IF:")
        print("1. Run: huggingface-cli login")
        print("2. Accept the license at: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0")
        print("3. Ensure you have enough GPU memory (16GB+ recommended)")

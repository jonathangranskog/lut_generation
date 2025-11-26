import torch


def get_device(allow_mps: bool = False) -> torch.device:
    """
    Get the best available device for computation.

    Args:
        allow_mps: Whether to allow MPS (Apple Silicon) backend.
                   Set to False for operations that don't support MPS
                   (e.g., grid_sampler_3d_backward).

    Returns:
        torch.device: The selected device (mps, cuda, or cpu)
    """
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

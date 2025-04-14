import torch


def get_device() -> torch.device:
    """CUDA or MPS if available, CPU otherwise."""
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    return torch.device(dev)

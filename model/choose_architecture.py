"""Pick the best available torch device (CUDA > MPS > CPU)."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

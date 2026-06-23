from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 0) -> None:
    """Match the deterministic seed setup used by the original script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve a user-facing device string without assuming a second GPU exists."""
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(f"Requested {requested}, but CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")
        device = torch.device(value)
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            warnings.warn(
                f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) "
                "are visible; falling back to cuda:0."
            )
            return torch.device("cuda:0")
        return device
    raise ValueError(f"Unsupported device specification: {requested!r}")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

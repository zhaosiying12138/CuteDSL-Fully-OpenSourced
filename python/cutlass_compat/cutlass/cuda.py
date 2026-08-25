"""CUDA context init for the compat surface (host side)."""
from __future__ import annotations


def initialize_cuda_context() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required (sm_120a profile)")
    torch.cuda.init()
    _ = torch.cuda.current_device()

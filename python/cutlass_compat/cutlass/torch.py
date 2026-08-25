"""cutlass.torch — torch tensor adapter (from_dlpack) for the compat surface."""
from __future__ import annotations


class Tensor:
    """Host-side tensor handle. Passes the device pointer to the kernel ABI."""

    def __init__(self, torch_tensor):
        import torch

        if not isinstance(torch_tensor, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(torch_tensor)}")
        if not torch_tensor.is_cuda:
            raise ValueError("tensor must be on CUDA device")
        self._torch = torch_tensor
        self.dtype_name = str(torch_tensor.dtype).replace("torch.", "")

    @property
    def shape(self):
        return tuple(self._torch.shape)

    def data_ptr(self) -> int:
        return self._torch.data_ptr()

    def __repr__(self):
        return f"<self_cutedsl.Tensor {self.dtype_name}{self.shape}>"


def from_dlpack(torch_tensor) -> Tensor:
    return Tensor(torch_tensor)

"""cutlass.torch — torch tensor adapter (from_dlpack) for the compat surface."""
from __future__ import annotations

import torch as _torch_mod  # module-level so _TORCH_OF table can reference it


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


def from_dlpack(torch_tensor):
    """torch tensor -> TensorMeta (device pointer + shape/stride/element)."""
    from self_cutedsl.frontend.meta import TensorMeta, make_tensor_meta

    if isinstance(torch_tensor, TensorMeta):
        return torch_tensor
    return make_tensor_meta(torch_tensor)


_TORCH_OF = {
    "Float32": _torch_mod.float32, "Float16": _torch_mod.float16,
    "BFloat16": _torch_mod.bfloat16, "Int32": _torch_mod.int32,
    "Int64": _torch_mod.int64, "Int8": _torch_mod.int8,
    "Float8E4M3FN": _torch_mod.float8_e4m3fn, "Float8E5M2": _torch_mod.float8_e5m2,
}


def dtype(d):
    """cutlass dtype class -> torch dtype."""
    import torch as _t

    name = getattr(d, "name", None) or getattr(d, "__name__", str(d))
    if name in _TORCH_OF:
        return _TORCH_OF[name]
    if isinstance(d, _t.dtype):
        return d
    raise ValueError(f"no torch dtype for {d}")


def current_stream():
    import torch as _t

    return _t.cuda.current_stream()

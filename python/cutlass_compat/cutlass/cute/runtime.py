"""cutlass.cute.runtime — from_dlpack host entry (compat)."""
from __future__ import annotations


def from_dlpack(x, assumed_align=None, **kw):
    """torch tensor (or object with data_ptr/shape/stride/dtype) -> TensorMeta."""
    from self_cutedsl.frontend.meta import make_tensor_meta, TensorMeta

    if isinstance(x, TensorMeta):
        return x
    return make_tensor_meta(x)


def make_fake_compact_tensor(dtype, shape, stride_order=None, assumed_align=None):
    from cutlass.cute._fi_arch import _mk as _f
    return _f.make_fake_compact_tensor(dtype, shape, stride_order, assumed_align)


def make_fake_stream(use_tvm_ffi_env_stream=False):
    from cutlass.cute._fi_arch import _mk as _f
    return _f.make_fake_stream(use_tvm_ffi_env_stream)

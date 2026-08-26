"""cutlass.cute.runtime — from_dlpack host entry (compat)."""
from __future__ import annotations


def from_dlpack(x, assumed_align=None, **kw):
    """torch tensor (or object with data_ptr/shape/stride/dtype) -> TensorMeta."""
    from self_cutedsl.frontend.meta import make_tensor_meta, TensorMeta

    if isinstance(x, TensorMeta):
        return x
    return make_tensor_meta(x)

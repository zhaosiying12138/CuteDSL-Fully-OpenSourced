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


def from_dlpack(torch_tensor, assumed_align=None, **kw):
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


def matrix(l, m_or_n, k_or_m, major, dtype, gen=None):
    """cutlass_torch.matrix(l, m, k, major, dtype) — synthetic GEMM operand.

    The logical result is always ``(mode0, mode1, l)``.  ``major`` is the
    official boolean ``is_mode0_major`` (string spellings are accepted for
    compatibility) and changes strides, not logical shape.
    """
    import torch as _t

    tdtype = _TORCH_OF.get(getattr(dtype, "name", str(dtype)), _t.float16)
    l = int(l)
    # The GeForce examples also pass their axis label directly.  In their
    # current supported surface, "m" is mode-0-major while "k"/"n" select
    # the row-major path (C uses "n").
    mode0_major = bool(major) if isinstance(major, bool) else major == "m"
    storage_shape = ((l, k_or_m, m_or_n) if mode0_major
                     else (l, m_or_n, k_or_m))
    # CUTLASS RANDOM initialization is integer-valued with an exclusive
    # upper bound; signed floating references use {-2,-1,0,1} by default.
    storage = _t.randint(-2, 2, storage_shape, dtype=_t.int32).to(tdtype)
    return storage.permute(2, 1, 0) if mode0_major else storage.permute(1, 2, 0)


def get_workspace_count(one_workspace_bytes, warmup, iterations):
    return max(1, min(10, iterations))


def default_stream():
    import torch as _t

    return _t.cuda.current_stream().cuda_stream


def current_stream():
    import torch as _t

    return _t.cuda.current_stream().cuda_stream


def cute_tensor_like(data_ref, cutlass_dtype=None, aligned_alloc=False,
                     buffer_align_bytes=16, is_dynamic_layout=False,
                     assumed_align=None, **kw):
    """official signature: cute_tensor_like(ref, dtype, aligned, align) ->
    (cute_tensor, torch_tensor)."""
    import torch as _t
    from cutlass.cute.runtime import from_dlpack

    tt = _t.empty_like(data_ref, device="cuda")
    tt.copy_(data_ref)                     # official: new CUDA tensor w/ data
    if cutlass_dtype is not None:
        dname = getattr(cutlass_dtype, "name", cutlass_dtype)
        if dname == "Float4E2M1FN":
            packed = _pack_fp4(tt)
            # keep the f32 reference grid-aligned with what the kernel
            # consumes (official init is fp4-representable): write the
            # dequantized values back so golden einsums match
            dq = _unpack_fp4(packed, tuple(tt.shape))
            data_ref.copy_(dq)
            from self_cutedsl.frontend.meta import TensorMeta, F4E2M1_
            return TensorMeta(packed, F4E2M1_,
                              tuple(data_ref.shape), tuple(packed.stride())), packed
        elif dname in ("Float8E4M3FN", "Float8E5M2", "Float8E8M0FNU"):
            td = _TORCH_OF.get(dname)
            if td is not None and td != tt.dtype:
                tt = tt.to(td)
            from self_cutedsl.frontend.meta import (TensorMeta, F8E4M3FN_,
                                                    F8E5M2_, F8E8M0_)
            elem = {"Float8E4M3FN": F8E4M3FN_, "Float8E5M2": F8E5M2_,
                    "Float8E8M0FNU": F8E8M0_}[dname]
            ct = from_dlpack(tt)
            return TensorMeta(tt, elem, tuple(tt.shape),
                              tuple(tt.stride())), tt
        else:
            td = _TORCH_OF.get(dname)
            if td is not None and td != tt.dtype:
                tt = tt.to(td)
    ct = from_dlpack(tt)
    return ct, tt


def convert_cute_tensor(src, dst=None, dtype=None, is_dynamic_layout=False,
                        **kw):
    """Elementwise convert src cute/torch tensor into dst (or dtype)."""
    import torch as _t

    if dst is None:
        return getattr(src, "_torch", src)
    st = getattr(src, "_torch", src)
    dt = getattr(dst, "_torch", dst)
    dt.copy_(st)
    return dst


_FP4_CODESTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pack_fp4(f32_tensor):
    """Pack an f32 tensor into E2M1 nibbles (2 per byte, element k even in
    the LOW nibble) — the CUTLASS fp4 storage convention."""
    import torch as _t

    values = f32_tensor.to(_t.float32)
    sign = (values < 0).to(_t.uint8)
    levels = _t.tensor(_FP4_CODESTS, dtype=_t.float32, device=values.device)
    codes = (values.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(_t.uint8)
    codes = codes | (sign << 3)
    # PTX e2m1 operand packing: element with the LOWER k index occupies
    # the HIGH nibble of each byte (cute fp4 convention)
    if codes.ndim >= 2 and codes.shape[1] % 2 == 0:
        return ((codes[:, 0::2, ...] << 4) |
                codes[:, 1::2, ...]).contiguous()
    flat = codes.reshape(-1)
    if flat.numel() % 2:
        flat = _t.cat([flat, _t.zeros(1, dtype=_t.uint8, device=flat.device)])
    return ((flat[0::2] << 4) | flat[1::2]).contiguous()


def _unpack_fp4(packed, shape):
    """Decode CUTLASS E2M1 storage to the exact values consumed by MMA."""
    import torch as _t

    levels = _t.tensor(_FP4_CODESTS, dtype=_t.float32, device=packed.device)

    def decode(nibble):
        magnitude = levels[(nibble & 0x7).long()]
        return _t.where((nibble & 0x8) != 0, -magnitude, magnitude)

    hi = decode(packed >> 4)
    lo = decode(packed & 0xF)
    if len(shape) >= 2 and shape[1] % 2 == 0:
        # (m, k/2, ...) x pair -> (m, k, ...), with lower k first.
        return _t.stack((hi, lo), dim=2).reshape(shape)
    return _t.stack((hi.reshape(-1), lo.reshape(-1)), dim=1).reshape(-1)[:_t.tensor(shape).prod()].reshape(shape)


class TensorInitType:
    RANDOM = "random"
    SPECIAL = "special"


class RandomInitConfig:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val


def create_and_permute_torch_tensor(shape, dtype, permute_order=(0,),
                                    init_type=None, init_config=None):
    """Create a tensor with the official RANDOM integer-value semantics.

    CUTLASS initializes an int32 tensor with ``random_(min, max)`` and then
    converts it to the requested dtype; ``max`` is exclusive.  Using a
    continuous uniform distribution here made the block-scale reference hold
    unquantized values while the device tensor held E4M3-rounded values.
    """
    import torch as _t

    if init_type == TensorInitType.RANDOM or init_config is not None:
        lo = getattr(init_config, "min_val", 0.0)
        hi = getattr(init_config, "max_val", 1.0)
        t = _t.randint(int(lo), int(hi), tuple(shape), dtype=_t.int32).to(dtype)
    else:
        t = _t.zeros(tuple(shape), dtype=dtype)
    return t.permute(*permute_order)

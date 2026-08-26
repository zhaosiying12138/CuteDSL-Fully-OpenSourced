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


def matrix(l, m_or_n, k_or_m, major, dtype, gen=None):
    """cutlass_torch.matrix(l, m, k, major, dtype) — synthetic GEMM operand.

    Layout convention: returns an (m, k) or (k, m) torch CPU tensor per
    the requested major ('k' -> row-major (m,k); 'm' -> (k,m) k-major).
    """
    import torch as _t

    _torch_to_dsl = {v: k for k, v in _TORCH_OF.items()}
    tdtype = _torch_to_dsl.get(getattr(dtype, "name", dtype), _t.float16)
    l = int(l)
    # 'k'-major A/B (m,k) and 'n'-major C (m,n) are both row-major in the
    # (first, second) argument order; only 'm'-major transposes
    if major in ("k", "n"):
        shape = ((m_or_n, k_or_m) if l == 1 else (l, m_or_n, k_or_m))
        t = _t.randn(shape, dtype=tdtype)
    else:
        shape = ((k_or_m, m_or_n) if l == 1 else (l, k_or_m, m_or_n))
        t = _t.randn(shape, dtype=tdtype).contiguous()
    if l == 1:
        t = t.unsqueeze(-1)         # (m,k,1): einsum "mkl" needs 3 dims
    else:
        t = t.permute(1, 2, 0)      # (l,m,k) -> (m,k,l)
    return t


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
            tt = _pack_fp4(tt)          # 2 nibbles/byte, little-nibble first
            from self_cutedsl.frontend.meta import TensorMeta, F4E2M1_
            ct = from_dlpack(tt)
            return TensorMeta(tt, F4E2M1_,
                              tuple(tt.shape), tuple(tt.stride())), tt
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

    flat = f32_tensor.reshape(-1).to(_t.float32)
    sign = (flat < 0).to(_t.uint8)
    mag = flat.abs()
    codes = _t.zeros_like(mag, dtype=_t.uint8)
    for code, st in enumerate(_FP4_CODESTS):
        codes[(mag >= st) & (mag <= (st + 0.25) if code < 7 else mag >= st)] = code
    # nearest of the representable magnitudes
    codes = _t.zeros_like(mag, dtype=_t.uint8)
    m_np = mag.cpu().numpy()
    import numpy as _np
    sts = _np.array(_FP4_CODESTS)
    idx = _np.abs(m_np[:, None] - sts[None, :]).argmin(axis=1)
    codes = _t.from_numpy(idx.astype(_np.uint8)).to(mag.device)
    codes = codes | (sign << 3)
    if codes.numel() % 2:
        codes = _t.cat([codes, _t.zeros(1, dtype=_t.uint8, device=codes.device)])
    packed = (codes[0::2] | (codes[1::2] << 4)).reshape(f32_tensor.shape[:-1] +
                                                        (f32_tensor.shape[-1] // 2,) if f32_tensor.dim() and f32_tensor.shape[-1] % 2 == 0 else (-1,))
    return packed


class TensorInitType:
    RANDOM = "random"
    SPECIAL = "special"


class RandomInitConfig:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val


def create_and_permute_torch_tensor(shape, dtype, permute_order=(0,),
                                    init_type=None, init_config=None):
    """torch tensor of shape, permuted, optionally randomized on [lo, hi]."""
    import torch as _t

    if init_type == TensorInitType.RANDOM or init_config is not None:
        lo = getattr(init_config, "min_val", 0.0)
        hi = getattr(init_config, "max_val", 1.0)
        t = lo + (hi - lo) * _t.rand(tuple(shape), dtype=_t.float32)
        t = t.to(dtype)
    else:
        t = _t.zeros(tuple(shape), dtype=dtype)
    return t.permute(*permute_order)

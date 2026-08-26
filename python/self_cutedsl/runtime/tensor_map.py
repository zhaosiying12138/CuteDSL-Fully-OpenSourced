"""tensor_map.py — CUtensorMap recipe (host side).

The compiler never encodes the opaque TensorMap bit pattern; it emits a
recipe (element type, rank, dims/strides, box, swizzle...) and this
runtime materializes it via cuTensorMapEncodeTiled per the CUDA API.

The descriptor is staged into device global memory (128 bytes,
256B-aligned torch allocation satisfies the 64B requirement); kernels
receive a plain global pointer to it. PTX permits tensormap operands in
global memory (fence.proxy.tensormap only required when the descriptor
is modified after launch).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cu

_DT = cu.CUtensorMapDataType
_TORCH_TO_CU_DATATYPE = {
    torch.float32: _DT.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16: _DT.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16: _DT.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.int32: _DT.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64: _DT.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.int8: _DT.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint8: _DT.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}


@dataclass
class TensorMapRecipe:
    """What the compiler emits instead of an opaque descriptor."""

    dtype: torch.dtype
    shape: tuple               # logical, slowest-first (torch order)
    strides_elems: tuple        # per-dim element strides (slowest-first)
    box: tuple                  # box dims, fastest-first (TMA order)

    @property
    def tile_bytes(self) -> int:
        n = 1
        for b in self.box:
            n *= int(b)
        return n * self.dtype.itemsize


def encode_to_bytes(recipe: TensorMapRecipe, global_address: int) -> bytes:
    """Encode into a fresh 128-byte host buffer; returns raw bytes."""
    dt = _TORCH_TO_CU_DATATYPE[recipe.dtype]
    rank = len(recipe.shape)
    u32, u64 = cu.cuuint32_t, cu.cuuint64_t
    # TMA arrays are fastest-first; torch order is slowest-first.
    gdim = [u64(int(d)) for d in reversed(recipe.shape)]
    # globalStrides: byte strides, fastest-first, innermost dim skipped;
    # torch dims 0..rank-2 (slowest-first) reversed gives the same order
    strides = [u64(int(s) * recipe.dtype.itemsize)
               for s in reversed(recipe.strides_elems[:-1])]
    box = [u32(int(b)) for b in recipe.box]
    estrides = [u32(1) for _ in recipe.shape]
    import os as _os
    if _os.environ.get("DG_TMA_DEBUG"):
        import sys as _s
        print(f"SELF_TMA rank={rank} gdim={[int(d) for d in gdim]} "
              f"gstr={[int(x) for x in strides]} box={[int(b) for b in box]}",
              file=_s.stderr)
    r = cu.cuTensorMapEncodeTiled(
        dt, rank, global_address, gdim, strides, box, estrides,
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    err = r[0] if isinstance(r, tuple) else r
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    return ctypes.string_at(r[1].getPtr(), 128)


class CUtensorMapView:
    """A CUtensorMap staged in device global memory."""

    def __init__(self, recipe: TensorMapRecipe, device_tensor: torch.Tensor):
        assert device_tensor.is_cuda
        self.recipe = recipe
        host_bytes = encode_to_bytes(recipe, device_tensor.data_ptr())
        self.device_copy = torch.frombuffer(bytearray(host_bytes),
                                            dtype=torch.uint8).to("cuda")
        assert self.device_copy.data_ptr() % 64 == 0
        self.source = device_tensor  # keep alive

    def data_ptr(self) -> int:
        return self.device_copy.data_ptr()


def encode_tiled(recipe: TensorMapRecipe, device_tensor: torch.Tensor) -> CUtensorMapView:
    return CUtensorMapView(recipe, device_tensor)

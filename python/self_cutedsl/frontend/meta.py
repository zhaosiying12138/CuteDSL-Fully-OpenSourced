"""meta.py — host-side CuTe meta objects (tensors, tiles, coords, layouts).

All constexpr meta computed at trace time, mirroring official partial
evaluation. `.type` strings approximate the official pretty types for the
`print(f"{x.type}")` lines in upstream examples.
"""
from __future__ import annotations

from .layout import CuteLayout, _flatten, _prod


# ------------------------------------------------------------------ dtypes
class ElementType:
    def __init__(self, name: str, width: int, is_integer: bool, is_float: bool):
        self.name, self.width = name, width
        self.is_integer, self.is_float = is_integer, is_float

    def __repr__(self):
        return self.name

    @property
    def mlir(self) -> str:
        return {"f32": "f32", "f16": "f16", "bf16": "bf16",
                "i32": "i32", "i64": "i64", "i8": "i8"}.get(self.name, "f32")


F32 = ElementType("f32", 32, False, True)
F16 = ElementType("f16", 16, False, True)
BF16 = ElementType("bf16", 16, False, True)
I32 = ElementType("i32", 32, True, False)
I64 = ElementType("i64", 64, True, False)
BOOL = ElementType("bool", 8, False, False)

_TORCH_TO_ELEM = {
    "float32": F32, "float16": F16, "bfloat16": BF16,
    "int32": I32, "int64": I64, "bool": BOOL,
}


# ------------------------------------------------------------------ tensors
class TensorMeta:
    """from_dlpack(torch_tensor)[.mark_layout_dynamic()]"""

    def __init__(self, torch_tensor, elem: ElementType, shape, stride=None):
        self._torch = torch_tensor
        self.element_type = elem
        self.shape = tuple(int(s) for s in shape)
        self.stride = tuple(int(s) for s in stride) if stride else _row_major(self.shape)

    def mark_layout_dynamic(self) -> "TensorMeta":
        return self  # layout dynamism is a specialization hint; same meta here

    @property
    def type(self) -> str:
        return (f"cutlass.Tensor(element={self.element_type.name}, "
                f"shape={self.shape}, stride={self.stride})")

    @property
    def ptr(self) -> int:
        return self._torch.data_ptr()

    def data_ptr(self) -> int:
        return self._torch.data_ptr()


def make_tensor_meta(torch_tensor) -> TensorMeta:
    key = str(torch_tensor.dtype).replace("torch.", "")
    elem = _TORCH_TO_ELEM.get(key)
    if elem is None:
        raise TypeError(f"unsupported torch dtype {key}")
    stride = tuple(int(s) for s in torch_tensor.stride())
    return TensorMeta(torch_tensor, elem, tuple(torch_tensor.shape), stride)


def _row_major(shape) -> tuple:
    out, acc = [], 1
    for d in reversed(shape):
        out.append(acc)
        acc *= d
    return tuple(reversed(out)) if out else ()


# --------------------------------------------------------------- tiling meta
class TiledTensorMeta:
    """zipped_divide(tensor, tiler): ((TileM,TileN),(RestM,RestN)) view."""

    def __init__(self, base: TensorMeta, tiler):
        self.base = base
        self.tile = (int(tiler[0]), int(tiler[1]))
        rest = [-(-s // t) for s, t in zip(base.shape, self.tile)]  # ceil div
        self.rest = tuple(rest)
        self.shape = (self.tile, self.rest)

    @property
    def type(self) -> str:
        return (f"cutlass.Tensor(element={self.base.element_type.name}, "
                f"tile={self.tile}, rest={self.rest})")


class CoordinateTensorMeta:
    """make_identity_tensor(shape): values are their own coordinates."""

    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)

    @property
    def type(self) -> str:
        return f"cutlass.Tensor(element=coord, shape={self.shape}, stride=identity)"

    def tiled(self, tiler) -> "CoordinateTensorMeta":
        rest = tuple(-(-s // t) for s, t in zip(self.shape, tiler))
        return TiledCoordinateMeta(self.shape, (int(tiler[0]), int(tiler[1])), rest)


class TiledCoordinateMeta:
    def __init__(self, global_shape, tile, rest):
        self.global_shape = global_shape
        self.tile, self.rest = tile, rest

    @property
    def type(self) -> str:
        return (f"cutlass.Tensor(element=coord, tile={self.tile}, "
                f"rest={self.rest})")


def zipped_divide(m, tiler):
    if isinstance(m, CoordinateTensorMeta):
        return m.tiled(_flatten(tiler)[:2])
    return TiledTensorMeta(m, _flatten(tiler)[:2])


def make_identity_tensor(shape) -> CoordinateTensorMeta:
    return CoordinateTensorMeta(shape)


# ------------------------------------------------------------- layout utils
def make_ordered_layout(shape, order) -> CuteLayout:
    """Layout with strides following the given dimension order (major-first)."""
    flat = _flatten(shape)[: len(shape)] if not isinstance(shape[0], tuple) else _flatten(shape)
    dims = [int(d) for d in (shape if not isinstance(shape[0], (tuple, list)) else _flatten(shape))]
    order = [int(o) for o in order]
    strides = [0] * len(dims)
    acc = 1
    for o in order:  # order lists dims from major to minor
        strides[o] = acc
        acc *= dims[o]
    return CuteLayout(tuple(dims), tuple(strides))


def cute_size(x, mode=None) -> int:
    """cute.size(tensor_or_layout, mode=[k]) — static at trace time."""
    if isinstance(x, TiledTensorMeta):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            return _prod(x.shape[k])
        return _prod(x.tile) * _prod(x.rest)
    if isinstance(x, CuteLayout):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            group = x.shape[k]  # top-level group k (nested or leaf)
            return _prod(group)
        return x.size
    if isinstance(x, TiledCoordinateMeta):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            return _prod((x.tile, x.rest)[k])
        return _prod(x.tile) * _prod(x.rest)
    from .cute_objects import (Tensor as _HostTensor, FragmentViewA,
                               FragmentViewB, R2SSmemView, _flatten)

    if isinstance(x, (FragmentViewA, FragmentViewB)):
        return x.size(mode)
    if isinstance(x, R2SSmemView):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            return x.shape[k]
        n = 1
        for d in x.shape:
            n *= int(d)
        return n
    if isinstance(x, _HostTensor):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            return _prod(x.layout.shape[k])   # top-level mode entry
        return _prod(_flatten(x.layout.shape))
    if isinstance(x, (int, tuple)):
        return _prod(x)
    raise TypeError(f"cute.size on {type(x)}")

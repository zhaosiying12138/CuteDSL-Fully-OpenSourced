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

    def __eq__(self, other):
        # cutlass._DType instances compare by name (duck identity)
        return getattr(other, "name", None) == self.name

    def __hash__(self):
        return hash(self.name)

    @property
    def mlir(self) -> str:
        return {"f32": "f32", "f16": "f16", "bf16": "bf16",
                "i32": "i32", "i64": "i64", "i8": "i8"}.get(self.name, "f32")


F32 = ElementType("f32", 32, False, True)
F16 = ElementType("f16", 16, False, True)
BF16 = ElementType("bf16", 16, False, True)
I32 = ElementType("i32", 32, True, False)
I64 = ElementType("i64", 64, True, False)
I8_ = ElementType("i8", 8, True, False)     # fp8/fp4 byte storage view
F4E2M1_ = ElementType("Float4E2M1FN", 4, False, True)
F8E4M3FN_ = ElementType("Float8E4M3FN", 8, False, True)
F8E5M2_ = ElementType("Float8E5M2", 8, False, True)
F8E8M0_ = ElementType("Float8E8M0FNU", 8, False, True)
BOOL = ElementType("bool", 8, False, False)

_TORCH_TO_ELEM = {
    "float32": F32, "float16": F16, "bfloat16": BF16,
    "int32": I32, "int64": I64, "bool": BOOL,
    "int8": I8_, "uint8": I8_, "float8_e4m3fn": I8_, "float8_e5m2": I8_,
}


# ------------------------------------------------------------------ tensors
class TensorMeta:
    """from_dlpack(torch_tensor)[.mark_layout_dynamic()]"""

    def __init__(self, torch_tensor, elem: ElementType, shape, stride=None):
        self._torch = torch_tensor
        self.element_type = elem
        self.shape = tuple(int(s) for s in shape)
        self.stride = tuple(int(s) for s in stride) if stride else _row_major(self.shape)
        self._layout_cache = None
        # mode decomposition for grouped views: [(size, stride), ...] or,
        # for a merged mode, (size, [(sub_size, sub_stride), ...])
        self._modes = [(int(a), int(b)) for a, b in zip(self.shape, self.stride)]

    def group_modes_meta(self, i, j):
        """Merge modes i..j into one (index math keeps the sub-strides)."""
        subs = self._modes[i:j + 1]
        g = 1
        for sz, _ in subs:
            g *= sz
        modes = self._modes[:i] + [(g, subs)] + self._modes[j + 1:]
        shp = tuple(m[0] for m in modes)
        strd = tuple(m[1] if isinstance(m[1], int) else m[1][0][1]
                     for m in modes)
        nm = TensorMeta(self._torch, self.element_type, shp, strd)
        nm._modes = modes
        return nm

    def _flat_index(self, coord):
        c = coord if isinstance(coord, tuple) else (coord,)
        idx = 0
        for ci, mode in zip(c, self._modes):
            if isinstance(mode[1], int):
                idx += int(ci) * mode[1]
            else:
                # merged group: decompose the digit over the sub-modes
                rem = int(ci)
                for sz, st in reversed(mode[1]):
                    idx += (rem % sz) * st
                    rem //= sz
        return idx

    def mark_layout_dynamic(self) -> "TensorMeta":
        return self  # layout dynamism is a specialization hint; same meta here

    @property
    def iterator(self):
        """cute.make_tensor(t.iterator, layout) source handle."""
        return self

    @property
    def layout(self):
        """Host-meta Layout view of this tensor's shape/stride."""
        from .cute_objects import Layout as _L

        if self._layout_cache is None:
            self._layout_cache = _L(self.shape, self.stride)
        return self._layout_cache

    def __getitem__(self, coord):
        return self._torch.reshape(-1)[self._flat_index(coord)]

    def __setitem__(self, coord, v):
        self._torch.reshape(-1)[self._flat_index(coord)] = v

    def mark_compact_shape_dynamic(self, *a, **kw):
        return self

    def mark_dynamic(self, *a, **kw):
        return self

    def cuda(self):
        """Host convenience: this meta already wraps a CUDA tensor."""
        return self

    def cpu(self):
        from .meta import make_tensor_meta
        return make_tensor_meta(self._torch.cpu())

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
    if isinstance(x, TensorMeta):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            return _prod(x.shape[k])
        return _prod(x.shape)
    if x is None:
        return 1
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        return len(x)
    if isinstance(x, (int, tuple)):
        return _prod(x)
    raise TypeError(f"cute.size on {type(x)}")

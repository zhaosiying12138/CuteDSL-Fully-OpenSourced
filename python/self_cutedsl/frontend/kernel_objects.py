"""kernel_objects.py — device-side value objects created during tracing."""
from __future__ import annotations

from .emitter import SSA


class KernelTensor:
    """A cute.Tensor kernel param: base pointer + tiling meta."""

    def __init__(self, ptr: SSA, tiled_meta, element_type):
        self.ptr = ptr
        self.meta = tiled_meta          # TiledTensorMeta or TensorMeta
        self._element_type = element_type

    @property
    def shape(self):
        from .meta import TiledTensorMeta

        if isinstance(self.meta, TiledTensorMeta):
            return self.meta.shape
        return self.meta.shape

    @property
    def stride(self):
        return getattr(self.meta, "stride", None)

    @property
    def element_type(self):
        return self._element_type

    @element_type.setter
    def element_type(self, v):
        self._element_type = v

    @property
    def layout(self):
        from self_cutedsl.frontend.cute_objects import Layout

        shp = tuple(self.meta.shape)
        strd = tuple(getattr(self.meta, "stride", None)
                     or _row_major_of(shp))
        return Layout(shp, strd)

    @property
    def iterator(self):
        return _IterPtr(self.ptr, self)

    def __getitem__(self, idx):
        """KernelTensor[i] / [r, c]: gmem element load (flat coordinate
        through the meta strides). Returns a Float32 typed scalar."""
        if not isinstance(idx, (tuple, list)):
            idx = (idx,)
        strides = tuple(getattr(self.meta, "stride", None)
                        or _row_major_of(tuple(self.meta.shape)))
        from .emitter import SSA as _S
        from cutlass._bridge_helpers import _emitter
        from cutlass import Float32
        e = _emitter()
        total = None
        for c, st in zip(idx, strides):
            v = c.ssa if hasattr(c, "ssa") else c
            if isinstance(v, _S):
                if v.type == "index":
                    v = e.ssa("i64", f"arith.index_cast {v.name} : index to i64")
                elif v.type == "i32":
                    v = e.ssa("i64", f"arith.extsi {v.name} : i32 to i64")
                c64 = e.ssa("i64", f"arith.constant {int(st)} : i64")
                term = e.ssa("i64", f"arith.muli {v.name}, {c64.name} : i64") \
                    if int(st) != 1 else v
                total = term if total is None else \
                    e.ssa("i64", f"arith.addi {total.name}, {term.name} : i64")
            else:
                piece = int(v) * int(st)
                if total is None or isinstance(total, int):
                    total = piece if total is None else total + piece
        if total is None:
            total = 0
        if isinstance(total, int):
            total = e.ssa("i64", f"arith.constant {total} : i64")
        elem = getattr(self.meta, "element_type", None)
        ety = {"Float16": "f16", "f16": "f16", "float16": "f16",
               "Float32": "f32", "f32": "f32", "float32": "f32"}.get(
            getattr(elem, "name", ""), "f32")
        p2 = e.ssa("!llvm.ptr<1>",
                   f"llvm.getelementptr {self.ptr.name}[{total.name}] : "
                   f"(!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {ety}")
        if ety == "f16":
            v16 = e.ssa("f16", f"llvm.load {p2.name} : !llvm.ptr<1> -> f16")
            return Float32(e.ssa("f32", f"llvm.fpext {v16.name} : f16 to f32"))
        return Float32(e.ssa("f32", f"llvm.load {p2.name} : !llvm.ptr<1> -> f32"))

    @property
    def is_tiled(self) -> bool:
        from .meta import TiledTensorMeta

        return isinstance(self.meta, TiledTensorMeta)


class BlockTile:
    """blkA = gA[((None,None), bidx)] — one CTA's tile with dynamic origin."""

    def __init__(self, base_ptr: SSA | None, tiled_meta, block_idx: SSA, is_coord=False):
        self.base_ptr = base_ptr        # None for coordinate tensors
        self.meta = tiled_meta
        self.block_idx = block_idx      # SSA index
        self.is_coord = is_coord

    @property
    def type(self) -> str:
        return f"cutlass.Tensor(tile={self.meta.tile})"


class ThreadOrigin:
    """thr origin as SSA pair (row0, col0) within the block tile."""

    def __init__(self, row: SSA, col: SSA, tiled_copy):
        self.row, self.col = row, col
        self.tc = tiled_copy


class ThrPartition:
    """thrA = thr_copy.partition_S(blkA): per-thread view of a tile."""

    def __init__(self, tile: BlockTile, origin: ThreadOrigin, tiled_copy):
        self.tile = tile
        self.origin = origin
        self.tc = tiled_copy

    @property
    def type(self) -> str:
        elem = getattr(getattr(self.tile.meta, "base", None), "element_type", None)
        return (f"cutlass.Tensor(element={elem.name if elem else 'coord'}, "
                f"vals={self.tc.vals_per_thread} per thread)")

    @property
    def shape(self):
        return (self.tc.vals_per_thread,)


class Fragment:
    """Register fragment with SSA slots (lazy)."""

    def __init__(self, count: int, dtype, name: str = "frg"):
        self.count = count
        self.dtype = dtype              # ElementType
        self.slots: dict[int, SSA] = {}
        self.vecs: list = []            # vector-granular values (perf path)
        self.name = name
        self.dims = (count,)

    @property
    def shape(self):
        return self.dims

    def __getitem__(self, i):
        if isinstance(i, (tuple, list)):
            return _FragSlice.from_modes(self, i)
        return self.slots[int(i)]

    def __setitem__(self, i, v):
        if isinstance(i, (tuple, list)):
            dims = getattr(self, "dims", None) or (self.count,)
            lin = 0
            for c, d in zip(i, dims):
                lin = lin * d + int(c)
            i = lin
        self.slots[int(i)] = v

    @property
    def element_type(self):
        return self.elem

    def fill(self, value):
        """accumulators.fill(0.0) — all slots to a constant."""
        if float(value) != 0.0:
            raise NotImplementedError("Fragment.fill(nonzero)")
        from .builtins import _emitter

        e = _emitter()
        for i in range(self.count):
            self.slots[i] = e.ssa("f32", "arith.constant 0.0 : f32")

    def reduce(self, op, init_val=None, reduction_profile=0):
        """Fold per-element values with op into one f32 SSA (init_val is a
        TypedScalar / SSA / plain float identity)."""
        from . import builtins as _b
        from cutlass._bridge_helpers import TypedScalar
        e = _b._emitter()
        if isinstance(init_val, TypedScalar):
            if init_val.ssa is None:
                init_val.ir_value()
            acc = init_val.ssa
        elif isinstance(init_val, SSA):
            acc = init_val
        else:
            acc = e.ssa("f32", f"arith.constant {float(init_val if init_val is not None else 0.0)!r} : f32")
        view = self.load()
        for v in view.values:
            acc = e.ssa(acc.type, f"arith.addf {acc.name}, {v.name} : {acc.type}")
        from cutlass import Float32
        return Float32(acc)

    def load(self) -> "FragmentView":
        if self.vecs:
            return FragmentView([], vecs=list(self.vecs))
        return FragmentView([self.slots[i] for i in range(self.count)])

    def store(self, view: "FragmentView"):
        if view.vecs:
            self.vecs = list(view.vecs)
            return
        for i in range(self.count):
            self.slots[i] = view.values[i]

    @property
    def type(self) -> str:
        return f"cutlass.Tensor(rmem {self.dtype.name} x {self.count})"


class _FragSlice:
    """View retaining leading fragment modes while fixing trailing modes.

    CuTe's ``frag[(None, m, n)]`` keeps the value mode and selects one MMA
    atom.  Fragment slots are stored atom-major with the retained values
    contiguous, so the selected atom starts at ``linear(m, n) * values``.
    """

    def __init__(self, holder, base: int, shape):
        self.holder = holder
        self.base = int(base)
        self.dims = tuple(int(d) for d in shape)
        self.count = _shape_product(self.dims)

    @classmethod
    def from_modes(cls, holder, index, shape=None):
        dims = tuple(int(d) for d in (shape or holder.shape))
        index = tuple(index)
        if len(index) != len(dims):
            raise IndexError(
                f"fragment rank mismatch: index {index} for shape {dims}")

        first_fixed = next(
            (pos for pos, coord in enumerate(index) if coord is not None),
            len(index),
        )
        if any(coord is not None for coord in index[:first_fixed]) or \
                any(coord is None for coord in index[first_fixed:]):
            raise IndexError(
                "fragment mode slices must retain leading modes and fix "
                f"trailing modes, got {index}")

        fixed_dims = dims[first_fixed:]
        fixed_coords = [int(getattr(c, "value", c))
                        for c in index[first_fixed:]]
        linear = 0
        for coord, dim in zip(fixed_coords, fixed_dims):
            if coord < 0 or coord >= dim:
                raise IndexError(
                    f"fragment coordinate {coord} outside mode size {dim}")
            linear = linear * dim + coord
        retained_shape = dims[:first_fixed]
        retained_count = _shape_product(retained_shape)
        return cls(holder, linear * retained_count, retained_shape)

    @property
    def shape(self):
        return self.dims

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        i = int(getattr(i, "value", i))
        if i < 0 or i >= self.count:
            raise IndexError(i)
        return self.holder.slots[self.base + i]

    def __setitem__(self, i, value):
        i = int(getattr(i, "value", i))
        if i < 0 or i >= self.count:
            raise IndexError(i)
        self.holder.slots[self.base + i] = value


def _shape_product(shape):
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


class FragmentView:
    """Result of Fragment.load(): scalar SSAs or vector-granular SSAs."""

    def to(self, dtype):
        from . import builtins as _b

        return _b.frag_to(self, dtype)

    def __init__(self, values=None, vecs=None):
        self.values = values or []
        self.vecs = vecs or []

    def reduce(self, op, init_val=None, reduction_profile=0):
        """Fold element values with op into one f32 value."""
        from . import builtins as _b
        from cutlass._bridge_helpers import TypedScalar
        e = _b._emitter()
        if isinstance(init_val, TypedScalar):
            if init_val.ssa is None:
                init_val.ir_value()
            acc = init_val.ssa
        elif isinstance(init_val, SSA):
            acc = init_val
        else:
            acc = e.ssa("f32",
                        f"arith.constant {float(init_val if init_val is not None else 0.0)!r} : f32")
        for v in self.values:
            acc = e.ssa(acc.type, f"arith.addf {acc.name}, {v.name} : {acc.type}")
        from cutlass import Float32
        return Float32(acc)

    def __add__(self, other: "FragmentView") -> "FragmentView":
        raise NotImplementedError("interpreter handles FragmentView arithmetic")


class CoordPair:
    """A dynamic (row, col) coordinate as SSA pair."""

    def __init__(self, row: SSA, col: SSA):
        self.row, self.col = row, col

    def __repr__(self):
        return f"<coord ({self.row.name}, {self.col.name})>"


def _row_major_of(shape):
    st = [1]
    for d in reversed(shape[1:]):
        st.insert(0, st[0] * d)
    return tuple(st)


class _IterPtr:
    """tensor.iterator + offset — official pointer arithmetic; carries
    .llvm_ptr (raw SSA) for inline-asm address computation."""

    def __init__(self, ptr, kt):
        self.ptr = ptr
        self.kt = kt

    @property
    def llvm_ptr(self):
        return self.ptr

    @property
    def base(self):
        return self.kt

    def _off_ssa(self, other):
        from cutlass._bridge_helpers import _emitter
        e = _emitter()
        if hasattr(other, "ssa"):  # TypedScalar
            v = other.ssa
            if v is None:
                other.ir_value()
                v = other.ssa
        elif hasattr(other, "name"):
            v = other
        else:
            v = e.ssa("i32", f"arith.constant {int(other)} : i32")
        if v.type == "index":
            v = e.ssa("i32", f"arith.index_cast {v.name} : index to i32")
        return v

    def __add__(self, other):
        from cutlass._bridge_helpers import _emitter
        e = _emitter()
        off = self._off_ssa(other)
        elem = getattr(getattr(self.kt, "meta", None), "element_type", None)
        ety = {"Float16": "f16", "Float32": "f32", "Uint8": "i8",
               "Int32": "i32"}.get(getattr(elem, "name", ""), "f16")
        p = e.gep(self.ptr, off, ety)
        return _IterPtr(p, self.kt)

    def __radd__(self, other):
        return self.__add__(other)

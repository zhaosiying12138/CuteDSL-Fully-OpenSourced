"""kernel_objects.py — device-side value objects created during tracing."""
from __future__ import annotations

from .emitter import SSA


class KernelTensor:
    """A cute.Tensor kernel param: base pointer + tiling meta."""

    def __init__(self, ptr: SSA, tiled_meta, element_type):
        self.ptr = ptr
        self.meta = tiled_meta          # TiledTensorMeta or TensorMeta
        self.element_type = element_type

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
        self.name = name

    @property
    def shape(self):
        return (self.count,)

    def load(self) -> "FragmentView":
        return FragmentView([self.slots[i] for i in range(self.count)])

    def store(self, view: "FragmentView"):
        for i in range(self.count):
            self.slots[i] = view.values[i]

    @property
    def type(self) -> str:
        return f"cutlass.Tensor(rmem {self.dtype.name} x {self.count})"


class FragmentView:
    """Result of Fragment.load(): a vector of SSA values."""

    def __init__(self, values: list[SSA]):
        self.values = values

    def __add__(self, other: "FragmentView") -> "FragmentView":
        raise NotImplementedError("interpreter handles FragmentView arithmetic")


class CoordPair:
    """A dynamic (row, col) coordinate as SSA pair."""

    def __init__(self, row: SSA, col: SSA):
        self.row, self.col = row, col

    def __repr__(self):
        return f"<coord ({self.row.name}, {self.col.name})>"

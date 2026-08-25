"""tiled.py — TV layouts, tiled copies, fragments: CuTe canonical partitioning
as host-side meta-objects (M3 first slice).

Everything here is constexpr meta computed at trace time (mirroring the
official partial-evaluation model); the interpreter lowers copy/partition
materialization into per-thread pointer arithmetic + vectorized loads/stores
(CopyUniversalOp path). SMEM staging + ldmatrix paths come with the
selfcute_sm120 layer later in M3.
"""
from __future__ import annotations

from .layout import CuteLayout, _flatten, _prod


def make_layout(shape, stride=None) -> CuteLayout:
    return CuteLayout(shape, stride)


def make_layout_tv(thr_layout: CuteLayout, val_layout: CuteLayout):
    """CuTe make_layout_tv: returns (tiler_mn, tv_layout) for a copy."""
    t_shape = _flatten(thr_layout.shape)
    v_shape = _flatten(val_layout.shape)
    if len(t_shape) != 2 or len(v_shape) != 2:
        raise NotImplementedError("make_layout_tv supports rank-2 thr/val for now")
    tiler = (t_shape[0] * v_shape[0], t_shape[1] * v_shape[1])
    # tv layout: ((thr_v, thr_rest), (val, rest)) — simplified canonical form:
    # threads cover thr_layout, values val_layout within each thread's tile
    tv = CuteLayout((t_shape[0], t_shape[1], v_shape[0], v_shape[1]),
                    (v_shape[0] * v_shape[1] * t_shape[1],
                     v_shape[0] * v_shape[1],
                     v_shape[1],
                     1))
    return tiler, tv


class CopyAtom:
    def __init__(self, op, element_type, bits=None):
        self.op = op                 # e.g. CopyUniversalOp()
        self.element_type = element_type
        self.bits = bits or 32


class CopyUniversalOp:
    """Plain universal load/store (ld.global/st.global)."""


class TiledCopy:
    def __init__(self, atom: CopyAtom, thr_layout: CuteLayout, val_layout: CuteLayout):
        self.atom = atom
        self.thr_layout = thr_layout
        self.val_layout = val_layout
        self.num_threads = _prod(thr_layout.shape)
        self.vals_per_thread = _prod(val_layout.shape)
        t_shape = _flatten(thr_layout.shape)
        v_shape = _flatten(val_layout.shape)
        self.tile_shape = (t_shape[0] * v_shape[0], t_shape[1] * v_shape[1])

    def get_slice(self, tidx: int) -> "ThreadSlice":
        return ThreadSlice(self, int(tidx))

    def partition_S(self, tile) -> "PartitionedTile":
        raise NotImplementedError("partition_S is driven by the interpreter")


class ThreadSlice:
    def __init__(self, tiled_copy: TiledCopy, tidx: int):
        self.tc = tiled_copy
        self.tidx = tidx

    def partition_S(self, tile) -> "PartitionedTile":
        return PartitionedTile(self.tc, self.tidx, tile, is_source=True)

    def partition_D(self, tile) -> "PartitionedTile":
        return PartitionedTile(self.tc, self.tidx, tile, is_source=False)


class PartitionedTile:
    """A per-thread view of a tile: base (row, col) + the value offsets."""

    def __init__(self, tc: TiledCopy, tidx: int, tile, is_source: bool):
        self.tc = tc
        self.tidx = tidx
        self.tile = tile  # interpreter-side tile object (tensor + origin)
        self.is_source = is_source

    def thread_origin(self) -> tuple[int, int]:
        t_shape = _flatten(self.tc.thr_layout.shape)
        v_shape = _flatten(self.tc.val_layout.shape)
        ty, tx = divmod(self.tidx, t_shape[1])
        return (ty * v_shape[0], tx * v_shape[1])

    def value_offsets(self) -> list[tuple[int, int]]:
        v_shape = _flatten(self.tc.val_layout.shape)
        return [(vy, vx) for vy in range(v_shape[0]) for vx in range(v_shape[1])]


def make_copy_atom(op, element_type, **kwargs) -> CopyAtom:
    return CopyAtom(op, element_type, kwargs.get("bits"))


def make_tiled_copy_tv(atom: CopyAtom, thr_layout: CuteLayout,
                       val_layout: CuteLayout) -> TiledCopy:
    return TiledCopy(atom, thr_layout, val_layout)


def make_fragment_like(partitioned, count: int | None = None) -> "Fragment":
    n = count if count is not None else partitioned.tc.vals_per_thread
    return Fragment(n)


class Fragment:
    """Register fragment: SSA values materialized by the interpreter."""

    def __init__(self, count: int):
        self.count = count

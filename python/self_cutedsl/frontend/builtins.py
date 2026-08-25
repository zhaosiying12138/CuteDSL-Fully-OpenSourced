"""builtins.py — cute.* device builtins routed into the active trace.

The interpreter publishes the active KernelInterpreter in `_active`; the
compat `cute` module calls these. State is per-trace, single-threaded.
"""
from __future__ import annotations

from .emitter import SSA
from .interp import InterpError

_active = None  # KernelInterpreter during kernel tracing


def _emitter():
    if _active is None:
        raise InterpError("cute.* device builtin used outside kernel trace")
    return _active.emitter


class _Arch:
    """cute.arch — device intrinsics."""

    def thread_idx(self):
        e = _emitter()
        return (e.thread_id("x"), e.thread_id("y"), e.thread_id("z"))

    def block_idx(self):
        e = _emitter()
        return (e.block_id("x"), e.block_id("y"), e.block_id("z"))

    def block_dim(self):
        e = _emitter()
        return (e.block_dim("x"), e.block_dim("y"), e.block_dim("z"))

    def sync_threads(self):
        _emitter().raw("gpu.barrier")


arch = _Arch()


def printf(fmt: str, *args) -> None:
    """cute.printf('{}-style fmt', values...) -> gpu.printf C-style."""
    e = _emitter()
    pieces = fmt.split("{}")
    assert len(pieces) == len(args) + 1, "placeholder/arg count mismatch"
    c_fmt = ""
    dyn = []
    for i, piece in enumerate(pieces):
        c_fmt += piece
        if i < len(args):
            a = args[i]
            if isinstance(a, SSA):
                # index lowers to i64; %d on an i64 vararg desyncs the rest
                c_fmt += {"i32": "%d", "i64": "%lld", "index": "%lld",
                          "f32": "%f", "f64": "%lf", "i1": "%d"}.get(a.type, "%d")
                dyn.append(a)
            elif isinstance(a, bool):
                c_fmt += "%d" % a
            elif isinstance(a, (int, float)):
                c_fmt += f"{a:g}"
            else:
                c_fmt += str(a)
    e.printf(c_fmt + "\n", dyn)


# ------------------------------------------------------------------ tiled copy
def make_copy_atom(op, element_type, **kwargs):
    from .tiled import CopyAtom

    return CopyAtom(op, element_type, kwargs.get("bits"))


def make_tiled_copy_tv(atom, thr_layout, val_layout):
    from .tiled import TiledCopy

    return TiledCopy(atom, thr_layout, val_layout)


def partition_for_thread(tiled_copy, tile):
    """tiled_copy.get_slice(tidx).partition_S/D(tile) — lowered here so the
    thread-origin math is emitted as SSA index arithmetic."""
    from .kernel_objects import BlockTile, ThreadOrigin, ThrPartition

    e = _emitter()
    assert isinstance(tile, BlockTile), f"partition needs a BlockTile, got {type(tile)}"
    t_shape = _flat2(tiled_copy.thr_layout.shape)
    v_shape = _flat2(tiled_copy.val_layout.shape)
    tidx = _active.tidx_ssa
    ty = e.idx_binop("arith.divsi", tidx, t_shape[1])
    tx = e.idx_binop("arith.remsi", tidx, t_shape[1])
    row0 = e.idx_binop("arith.muli", ty, v_shape[0])
    col0 = e.idx_binop("arith.muli", tx, v_shape[1])
    origin = ThreadOrigin(row0, col0, tiled_copy)
    return ThrPartition(tile, origin, tiled_copy)


def make_fragment_like(part):
    from .kernel_objects import Fragment
    from .meta import F32

    return Fragment(part.tc.vals_per_thread, F32)


def make_rmem_tensor(shape, dtype):
    from .kernel_objects import Fragment
    from .meta import BOOL

    n = 1
    for d in (shape if isinstance(shape, (tuple, list)) else (shape,)):
        n *= int(d)
    return Fragment(n, BOOL if getattr(dtype, "name", "") in ("bool", "Boolean") else dtype)


def elem_less(coord, shape):
    e = _emitter()
    from .kernel_objects import CoordPair

    assert isinstance(coord, CoordPair)
    m, n = int(shape[0]), int(shape[1])
    r_ok = e.cmpi_slt_const(coord.row, m)
    c_ok = e.cmpi_slt_const(coord.col, n)
    return e.ssa("i1", f"arith.andi {r_ok.name}, {c_ok.name} : i1")


def copy(atom, src, dst, pred=None):
    """cute.copy(atom, src, dst, pred=frag) — per-value predicated ld/st."""
    from .kernel_objects import Fragment, ThrPartition

    e = _emitter()
    if isinstance(src, ThrPartition) and isinstance(dst, Fragment):
        _copy_g2r(e, src, dst, pred)
    elif isinstance(src, Fragment) and isinstance(dst, ThrPartition):
        _copy_r2g(e, src, dst, pred)
    else:
        raise TypeError(f"copy: unsupported src/dst {type(src)} -> {type(dst)}")


def _partition_base(e, part) -> tuple:
    """Hoisted (row, col) of this thread's value (0,0) in its block; cached."""
    if getattr(part, "_base_rc", None) is not None:
        return part._base_rc
    tile_meta = part.tile.meta
    tile_m, tile_n = tile_meta.tile
    rest = tile_meta.rest
    if len(rest) >= 2:
        blk_r = e.idx_binop("arith.divsi", part.tile.block_idx, rest[1])
        blk_c = e.idx_binop("arith.remsi", part.tile.block_idx, rest[1])
    else:
        blk_r, blk_c = e.idx_const(0), e.idx_binop("arith.remsi", part.tile.block_idx, rest[0])
    row = e.idx_binop("arith.muli", blk_r, tile_m)
    row = e.idx_binop("arith.addi", row, part.origin.row)
    col = e.idx_binop("arith.muli", blk_c, tile_n)
    col = e.idx_binop("arith.addi", col, part.origin.col)
    part._base_rc = (row, col)
    return part._base_rc


def _partition_element_offset(e, part, j) -> tuple:
    """SSA (row, col) of value j for this thread's partition of its block."""
    row0, col0 = _partition_base(e, part)
    v_shape = _flat2(part.tc.val_layout.shape)
    rj, cj = divmod(j, v_shape[1])
    row = e.idx_binop("arith.addi", row0, rj) if rj else row0
    col = e.idx_binop("arith.addi", col0, cj) if cj else col0
    return row, col


def _coord_value(e, part, j):
    from .kernel_objects import CoordPair

    row, col = _partition_element_offset(e, part, j)
    return CoordPair(row, col)


def _vec_width(part, pred) -> int:
    """Vectorize the per-thread value row when provably 16B-aligned.

    With a predicate, only vectorize when the global shape is statically
    tile-divisible (every predicate is provably true and can be dropped).
    """
    v_cols = _flat2(part.tc.val_layout.shape)[1]
    elem = getattr(getattr(part.tile.meta, "base", None), "element_type", None)
    if elem is None or elem.name != "f32":
        return 1
    if _global_cols(part) * elem.width % 128 != 0:
        return 1  # row starts not 16B-aligned
    if v_cols % 4 != 0:
        return 1
    if pred is not None:
        m, n = part.tile.meta.base.shape
        tm, tn = part.tile.meta.tile
        if m % tm != 0 or n % tn != 0:
            return 1  # real boundary: keep per-lane predication
    return 4


def _copy_g2r(e, part, frag, pred):
    base = part.tile.base_ptr
    n = _global_cols(part)
    vw = _vec_width(part, pred)
    v_cols = _flat2(part.tc.val_layout.shape)[1]
    v_rows = part.tc.vals_per_thread // v_cols
    for r in range(v_rows):
        j0 = r * v_cols
        row, col = _partition_element_offset(e, part, j0)
        flat = e.idx_binop("arith.muli", row, n)
        flat = e.idx_binop("arith.addi", flat, col)
        p = e.gep(base, flat)
        if vw > 1:
            frag.vecs.append(e.load_vec_f32(p, vw))
            continue
        for lane in range(v_cols):
            p_lane = e.gep(base, e.idx_binop("arith.addi", flat, lane))
            v = e.load_f32(p_lane)
            if pred is not None:
                old = frag.slots.get(j0 + lane) or e.ssa("f32", "arith.constant 0.0 : f32")
                sel = e.ssa("f32",
                            f"arith.select {pred.slots[j0 + lane].name}, {v.name}, {old.name} : f32")
                frag.slots[j0 + lane] = sel
            else:
                frag.slots[j0 + lane] = v


def _copy_r2g(e, frag, part, pred):
    n = _global_cols(part)
    vw = _vec_width(part, pred)
    v_cols = _flat2(part.tc.val_layout.shape)[1]
    v_rows = part.tc.vals_per_thread // v_cols
    for r in range(v_rows):
        j0 = r * v_cols
        row, col = _partition_element_offset(e, part, j0)
        flat = e.idx_binop("arith.muli", row, n)
        flat = e.idx_binop("arith.addi", flat, col)
        p = e.gep(part.tile.base_ptr, flat)
        if vw > 1 and frag.vecs:
            e.store_vec_f32(frag.vecs[r], p, vw)
            continue
        for lane in range(v_cols):
            p_lane = e.gep(part.tile.base_ptr, e.idx_binop("arith.addi", flat, lane))
            if pred is not None:
                e.open_if(pred.slots[j0 + lane])
                e.store_f32(frag.slots[j0 + lane], p_lane)
                e.close_if()
            else:
                e.store_f32(frag.slots[j0 + lane], p_lane)


def _global_cols(part) -> int:
    return int(part.tile.meta.base.shape[1])


def _flat2(x):
    out = []

    def rec(v):
        if isinstance(v, (tuple, list)):
            for i in v:
                rec(i)
        else:
            out.append(int(v))

    rec(x)
    if len(out) < 2:
        out = out + [1]
    return out


# expose the thread index SSA recorded by the interpreter
def _get_tidx():
    return _active.tidx_ssa

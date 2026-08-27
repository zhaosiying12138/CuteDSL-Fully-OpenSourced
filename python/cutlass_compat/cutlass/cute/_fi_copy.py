"""tv-layout tiled copy + SmemAllocator + arch/cpasync extensions for the
flashinfer operators (verbatim source support).

The rmsnorm/add_rmsnorm kernels build a ((T,R),(V,B)) thread-value layout
over a column-major (rows x cols) tile and copy global->shared with
cp.async 16B per thread per B-block, predicated on the k coordinate.
"""
from __future__ import annotations

import cutlass
from cutlass._bridge_helpers import _emitter, TypedScalar
from cutlass.cute import _fi_ext as _X
from cutlass.cute._fi_ext import Pointer, ReductionOp, TensorSSA  # noqa: F401


# ---------------------------------------------------------------------------
# cpasync atom
# ---------------------------------------------------------------------------
class CopyG2SOp:
    """cute.nvgpu.cpasync.CopyG2SOp — 16B cp.async global->shared."""

    def __init__(self):
        self.num_bits_per_copy = 128


class _CpAsyncAtom:
    def __init__(self, op, elem, num_bits_per_copy):
        self.op = op
        self.elem = elem
        self.bits = num_bits_per_copy


def make_copy_atom(op, elem=None, num_bits_per_copy=None, **kw):
    if op.__class__.__name__ == "CopyG2SOp":
        return _CpAsyncAtom(op, elem, num_bits_per_copy or 128)
    from self_cutedsl.frontend import builtins as _b
    return _b.make_copy_atom(op, elem, **kw)


class TVCopyAsync:
    """make_tiled_copy(cpasync_atom, tv_layout, tiler_mn) — get_slice /
    partition_S / partition_D produce per-thread coordinate views."""

    def __init__(self, atom, tv_layout, tiler_mn):
        self.atom = atom
        self.tv = tv_layout            # host Layout object
        self.tiler = tiler_mn
        self.tidx = None

    def get_slice(self, tidx):
        self.tidx = tidx
        return self

    def partition_S(self, tensor):
        return TVPartView(self, tensor, "S")

    def partition_D(self, tensor):
        return TVPartView(self, tensor, "D")


class TVPartView:
    """Per-thread partitioned view: subscripts give per-value-unit objects
    (elem coordinate maps); drives copy()/autovec_copy/predicate_k."""

    def __init__(self, tc, tensor, role):
        self.tc = tc
        self.tensor = tensor
        self.role = role

    # -- geometry -------------------------------------------------------
    @property
    def _geom(self):
        tv = self.tc.tv
        (T, R), (V, B) = tv.shape[0], tv.shape[1]
        return int(T), int(R), int(V), int(B)

    @property
    def shape(self):
        T, R, V, B = self._geom
        return (V * B,)

    def thread_base(self):
        """(t, r) decomposition of the bound thread index."""
        T, R, V, B = self._geom
        e = _emitter()
        tid = self.tc.tidx
        t = e.idx_binop("arith.remsi", tid, e.ssa("index", f"arith.constant {T} : index")) \
            if not isinstance(tid, int) else tid % T
        r = e.idx_binop("arith.divsi", tid, e.ssa("index", f"arith.constant {T} : index")) \
            if not isinstance(tid, int) else tid // T
        return t, r

    def coord(self, j):
        """Tile (row, col) for value-unit j of the bound thread (constexpr
        j only — the kernels iterate value units at trace time)."""
        e = _emitter()
        T, R, V, B = self._geom
        t, r = self.thread_base()

        def _c(x):
            return x if hasattr(x, "name") else e.ssa(
                "index", f"arith.constant {int(x)} : index")

        v = j % V
        b = j // V
        if hasattr(t, "name"):
            tbase = e.idx_binop("arith.muli", t,
                                e.ssa("index", f"arith.constant {V} : index"))
            col = e.idx_binop("arith.addi", tbase, e.ssa(
                "index", f"arith.constant {v + b * V * T} : index"))
        else:
            col = int(t) * V + v + b * V * T
        return r, col

    def __getitem__(self, idx):
        return _TVUnit(self, idx if not isinstance(idx, tuple) else idx[0])


class _TVUnit:
    def __init__(self, view, j):
        self.view = view
        self.j = int(j)


# ---------------------------------------------------------------------------
# copy() lowering: cp.async per 16B block with predicate
# ---------------------------------------------------------------------------
def copy_cpasync(atom, tXgX, tXsX, pred):
    """cute.copy(cpasync_atom, tXgX, tXsX, pred=tXpX): one cp.async.cg of
    `bits` per B-block; predicate is a per-unit i1 list (or None)."""
    e = _emitter()
    tc = tXgX.tc
    T, R, V, B = tXgX._geom
    gX = tXgX.tensor          # host/kernel tensor view (gmem)
    sX = tXsX.tensor          # smem tensor view
    H = _tile_cols(tXgX)      # full row extent for gmem addressing
    bidx = _active_bidx()
    bytes_per = atom.bits // 8
    elems = bytes_per // 2    # f16 element size assumed for G2S loads
    for b in range(B):
        row, col0 = tXgX.coord(b * V)
        # gmem address: (bidx*R + row)*H + col0 .. +V
        gr = e.idx_binop("arith.addi",
                         e.idx_binop("arith.muli", bidx,
                                     e.ssa("index", f"arith.constant {R} : index")),
                         row if not isinstance(row, int) else e.ssa("index", f"arith.constant {row} : index"))
        grow_off = e.idx_binop("arith.muli", gr, e.ssa("index", f"arith.constant {H} : index"))
        gcol = col0 if not isinstance(col0, int) else e.ssa("index", f"arith.constant {col0} : index")
        goff = e.idx_binop("arith.addi", grow_off, gcol)
        gp = _gep_gmem(e, gX, goff)
        # smem address: col-major tile -> row + col*R
        scol = gcol
        C_tile = _tile_cols(tXgX)
        row_s = row if not isinstance(row, int) else e.ssa("index", f"arith.constant {row} : index")
        soff = e.idx_binop(
            "arith.addi",
            e.idx_binop("arith.muli", row_s,
                        e.ssa("index", f"arith.constant {C_tile} : index")),
            scol)
        sp = _gep_smem(e, sX, soff)
        p = _pred_at(pred, b)
        e.open_if(p)
        e.raw(f'nvvm.inline_ptx "cp.async.cg.shared.global [$1], [$0], {bytes_per};" '
              f'ro ({gp.name}, {sp.name} : !llvm.ptr<1>, !llvm.ptr<3>)')
        e.close_if()


def _pred_at(pred, b):
    e = _emitter()
    if pred is None:
        return e.ssa("i1", "arith.constant 1 : i1")
    if hasattr(pred, "slots"):  # Fragment of i1 predicates
        return pred.slots[b]
    if isinstance(pred, (list, tuple)):
        return pred[b]
    if hasattr(pred, "name"):
        return pred
    return e.ssa("i1", "arith.constant 1 : i1")


def _tile_cols(view):
    tiler = view.tc.tiler
    return int(tiler[1])


def _active_bidx():
    from self_cutedsl.frontend import builtins as _b
    e = _emitter()
    return e.block_id("x")


def _gep_gmem(e, tensor, off):
    ptr = tensor.base if not hasattr(tensor.base, "ptr") else tensor.base.ptr
    if hasattr(off, "type"):
        if off.type == "index":
            off = e.ssa("i64", f"arith.index_cast {off.name} : index to i64")
        elif off.type == "i32":
            off = e.ssa("i64", f"arith.extsi {off.name} : i32 to i64")
    else:
        off = e.ssa("i64", f"arith.constant {int(off)} : i64")
    return e.ssa("!llvm.ptr<1>",
                 f"llvm.getelementptr {ptr.name}[{off.name}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16")


def _gep_smem(e, tensor, off):
    arr = tensor.base
    if hasattr(arr, "ptr"):
        p = arr.ptr
    else:
        p = arr
    off32 = e.ssa("i64", f"arith.index_cast {off.name} : index to i64")
    return e.ssa("!llvm.ptr<3>",
                 f"llvm.getelementptr {p.name}[{off32.name}] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16")


class CoordTile:
    """local_tile(identity_tensor, tiler, coord): coordinate tile marker."""

    def __init__(self, meta, tiler, coord):
        self.meta = meta
        self.tiler = tiler
        self.coord = coord


class _CoordUnit:
    """tXcX[(0, j), 0, k] -> indexable [0]=row SSA, [1]=col SSA."""

    def __init__(self, view, j):
        from cutlass._bridge_helpers import _emitter
        e = _emitter()
        self.view = view
        T, R, V, B = view._geom
        t, r = view.thread_base()

        def _idx(x):
            return x if hasattr(x, "name") else e.ssa(
                "index", f"arith.constant {int(x)} : index")

        v = j % V
        b = j // V
        # global row for bounds predicates (tile row + block offset)
        gr = r
        if hasattr(r, "name"):
            bid = e.block_id("x")
            gr = e.idx_binop(
                "arith.addi",
                e.idx_binop("arith.muli", bid,
                            e.ssa("index", f"arith.constant {R} : index")),
                r)
        else:
            from cutlass._bridge_helpers import _emitter as _E
            gr = int(_E().block_id("x").name and 0 or 0) + 0  # constexpr fallback below
        self.row = _idx(gr) if hasattr(gr, "name") else _idx(r)
        tbase = e.idx_binop("arith.muli", _idx(t),
                            e.ssa("index", f"arith.constant {V} : index")) \
            if hasattr(t, "name") else _idx(int(t) * V)
        c = e.idx_binop("arith.addi", tbase, _idx(v))
        if b:
            c = e.idx_binop("arith.addi", c,
                            e.ssa("index", f"arith.constant {b * V * T} : index"))
        self.col = c

    def __getitem__(self, which):
        return self.row if int(which) == 0 else self.col


# TVPartView: hierarchical predicate-style indexing over coord partitions
_old_get = TVPartView.__getitem__


def _tvpart_getitem(self, idx):
    if isinstance(idx, tuple) and len(idx) == 3 \
            and isinstance(idx[0], tuple):
        (a, j) = idx[0]
        if int(a) == 0 and isinstance(self.tensor, CoordTile):
            return _CoordUnit(self, j)
    return _old_get(self, idx)


TVPartView.__getitem__ = _tvpart_getitem

_old_shape = TVPartView.shape


@property
def _tvpart_shape(self):
    if isinstance(self.tensor, CoordTile):
        n = _old_shape.fget(self)[0]
        return (n, 1, 1)
    return _old_shape.fget(self)


TVPartView.shape = _tvpart_shape

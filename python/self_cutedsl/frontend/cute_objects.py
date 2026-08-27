"""cute_objects.py — the CuTe object-model layer (M6.1): Layout/Tensor meta,
TMA atom + partition, TiledMma with partition_A/B/C, and fragment gemm.

All meta computed at trace time (official CuTeDSL model); values that
depend on runtime thread/block indices are captured as SSA index
expressions carried on the objects. Emitters lower to the intrinsics
already verified in M4/M5 (ldmatrix, mma.sync, cp.async.bulk.tensor,
mbarrier).
"""
from __future__ import annotations

from .emitter import SSA
from .layout import CuteLayout


def _flatten(x):
    out = []

    def rec(v):
        if isinstance(v, (tuple, list)):
            for i in v:
                rec(i)
        else:
            from .layout import CuteLayout as _CL
            if hasattr(v, "outer") and hasattr(v, "stages"):
                rec(v.outer.shape)      # staged layout: per-stage shape
                return
            if isinstance(v, _CL):
                rec(v.shape)
            else:
                out.append(int(v))

    rec(x)
    return out


def _prod(xs):
    r = 1
    for x in xs:
        r *= int(x)
    return r


# ------------------------------------------------------------------ Layout
class Layout:
    """Hierarchical (shape, stride) pair; strides may nest like shapes."""

    def __init__(self, shape, stride):
        # staged-layout args degrade to their per-stage shape (kernel-side
        # views only consume per-stage geometry; staging rides the window)
        if hasattr(shape, "outer"):
            shape = shape.outer.shape
            stride = stride.outer.stride if hasattr(stride, "outer") else stride
        self.shape = _nest(shape)
        self.stride = _nest(stride)

    @property
    def size(self):
        return _prod(_flatten(self.shape))

    @property
    def rank(self):
        return len(_flatten(self.shape))

    def __repr__(self):
        return f"Layout({_render(self.shape)}:{_render(self.stride)})"

    def __getitem__(self, i):
        if isinstance(i, tuple) and len(i) == 2 and isinstance(i[1], (tuple, list)) \
                and isinstance(self.shape, tuple) and len(self.shape) == 2 \
                and isinstance(self.shape[1], (tuple, list)):
            # ((tile),(rest))[k, (None,...)] — dense_gemm grid-shape idiom:
            # select tile group k, keep the flagged rest modes
            rest_s = _flatten(self.shape[1])
            rest_d = _flatten(self.stride[1])
            keep_s = [s for s, c in zip(rest_s, i[1]) if c is None]
            keep_d = [d for d, c in zip(rest_d, i[1]) if c is None]
            return Layout(tuple(keep_s), tuple(keep_d))
        if isinstance(i, tuple):
            # coord-style get: (int|(None,...), ...) — project modes
            keep = []
            flat_s = _flatten(self.shape)
            flat_d = _flatten(self.stride)
            for s0, d0, c in zip(flat_s, flat_d, i):
                if c is None:
                    keep.append(s0)
            return Layout(tuple(keep), tuple(flat_d[:len(keep)]))
        return _flatten(self.shape)[i]

    def get_hier_coord(self, i):
        """Linear index -> coordinate tuple over the flattened shape."""
        flat = _flatten(self.shape)
        coord, rem = [], int(i)
        for d in reversed(flat):
            coord.append(rem % int(d))
            rem //= int(d)
        return tuple(reversed(coord))

    def __len__(self):
        return len(_flatten(self.shape))

    def get_flat_coord(self, idx):
        """flat index -> flat coord tuple (row-major traversal)."""
        shp = _flatten(self.shape)
        out = []
        for d in reversed(shp):
            out.append(idx % d)
            idx //= d
        return tuple(reversed(out))


def _nest(x):
    if isinstance(x, (tuple, list)):
        return tuple(_nest(i) for i in x)
    return int(x)


def _render(t):
    if isinstance(t, tuple):
        return "(" + ",".join(_render(x) for x in t) + ")"
    return str(t)


def make_layout(shape, stride=None, order=None):
    if isinstance(shape, Layout):
        return shape
    flat = _flatten(shape)
    if stride is None and order is None:
        stride = _row_major_nested(shape)
    elif order is not None:
        # order: dims from major to minor
        strides, acc = [0] * len(flat), 1
        for d in order:
            strides[d] = acc
            acc *= flat[d]
        stride = tuple(strides)
    return Layout(shape, stride)


def _row_major_nested(shape):
    flat = _flatten(shape)
    out, acc = [], 1
    for d in reversed(flat):
        out.append(acc)
        acc *= d
    return tuple(reversed(out))


# ------------------------------------------------------------------ Tensor
class Tensor:
    """A tensor view: base (SSA ptr or SmemArray) + Layout + element type.

    base kinds: gmem SSA ptr ('gmem'), smem ('smem', with name), rmem
    (python-held fragment). Subscripts and partitioning return new views.
    """

    def __init__(self, base, layout, element, origin=None):
        self.base = base              # SSA ptr / SmemArray / None (coord)
        self.layout = layout
        self.element = element
        self.origin = origin or ()    # SSA/index offsets accumulated

    @property
    def shape(self):
        return self.layout.shape

    @property
    def element_type(self):
        return self.element

    @property
    def iterator(self):
        return self

    @property
    def dtype(self):
        return self.element

    @property
    def type(self):
        return f"cutlass.Tensor(element={getattr(self.element, 'name', '?')}, shape={self.shape})"

    def __getitem__(self, idx):
        """Coord selection over flat modes: None keeps, SSA/int selects
        (mode dropped from the layout, offset recorded in coord_offs)."""
        idx = idx if isinstance(idx, tuple) else (idx,)
        flat_shape = list(_flatten(self.layout.shape))
        if len(idx) == len(flat_shape) and all(c is not None for c in idx) \
                and self.base.__class__.__name__ == "SmemArray":
            return self._load_smem_element(idx)
        # ((tile),(rest))[(k, (None,...))] — hierarchical grid idiom:
        # select tile group k, keep the flagged rest modes
        if len(idx) == 2 and isinstance(idx[0], int) \
                and isinstance(idx[1], (tuple, list)) \
                and isinstance(self.layout.shape, tuple) \
                and len(self.layout.shape) == 2 \
                and isinstance(self.layout.shape[1], (tuple, list)):
            rest_s = _flatten(self.layout.shape[1])
            rest_d = _flatten(self.layout.stride[1])
            keep_s = [x for x, c in zip(rest_s, idx[1]) if c is None]
            keep_d = [x for x, c in zip(rest_d, idx[1]) if c is None]
            nv = Tensor(self.base, Layout(tuple(keep_s), tuple(keep_d)),
                        self.element, self.origin)
            nv.coord_offs = dict(getattr(self, "coord_offs", {}))
            nv.swizzle = getattr(self, "swizzle", None)
            return nv
        flat_s = list(_flatten(self.layout.shape))
        flat_d = list(_flatten(self.layout.stride))
        if len(idx) > len(flat_s):
            # trailing stage selectors (sX[None, None, 0]) drop out —
            # staging rides the dedicated window machinery
            idx = idx[:len(flat_s)]
        offs = dict(getattr(self, "coord_offs", {}))
        keep_s, keep_d = [], []
        for m, (s0, d0, c) in enumerate(zip(flat_s, flat_d,
                                            idx + (None,) * (len(flat_s) - len(idx)))):
            if c is None:
                keep_s.append(s0)
                keep_d.append(d0)
            else:
                offs[m] = c
        nv = Tensor(self.base, Layout(tuple(keep_s), tuple(keep_d)),
                    self.element, self.origin)
        nv.coord_offs = offs
        nv.swizzle = getattr(self, "swizzle", None)
        return nv

    def _load_smem_element(self, coord):
        from cutlass._bridge_helpers import _emitter
        from cutlass import Float16, Float32

        e = _emitter()
        total = None
        for value, stride in zip(coord, _flatten(self.layout.stride)):
            if hasattr(value, "ssa"):
                if value.ssa is None:
                    value.ir_value()
                value = value.ssa
            if hasattr(value, "type"):
                if value.type == "index":
                    value = e.ssa(
                        "i64", f"arith.index_cast {value.name} : index to i64")
                elif value.type != "i64":
                    value = e.ssa(
                        "i64", f"arith.extsi {value.name} : {value.type} to i64")
                term = value
                if int(stride) != 1:
                    scale = e.ssa(
                        "i64", f"arith.constant {int(stride)} : i64")
                    term = e.ssa(
                        "i64", f"arith.muli {value.name}, {scale.name} : i64")
            else:
                term = int(value) * int(stride)
            if total is None:
                total = term
            elif isinstance(total, int) and isinstance(term, int):
                total += term
            else:
                if isinstance(total, int):
                    total = e.ssa(
                        "i64", f"arith.constant {total} : i64")
                if isinstance(term, int):
                    term = e.ssa(
                        "i64", f"arith.constant {term} : i64")
                total = e.ssa(
                    "i64", f"arith.addi {total.name}, {term.name} : i64")
        if isinstance(total, int):
            total = e.ssa("i64", f"arith.constant {total} : i64")
        elem_name = getattr(self.element, "name", "")
        elem_type = "f32" if elem_name in ("f32", "Float32") else "f16"
        ptr = e.ssa(
            "!llvm.ptr<3>",
            f"llvm.getelementptr {self.base.ptr.name}[{total.name}] : "
            f"(!llvm.ptr<3>, i64) -> !llvm.ptr<3>, {elem_type}",
        )
        loaded = e.ssa(
            elem_type,
            f"llvm.load {ptr.name} : !llvm.ptr<3> -> {elem_type}",
        )
        return (Float32 if elem_type == "f32" else Float16)(loaded)


def make_tensor(base, layout, element):
    return Tensor(base, layout, element)


# ------------------------------------------------------- algebra (host meta)
def group_modes(tensor_or_layout, i: int, j: int):
    """Collapse modes i..j (inclusive) of the top-level shape into one."""
    lay = tensor_or_layout.layout if isinstance(tensor_or_layout, Tensor) else tensor_or_layout
    shp, str_ = _flatten(lay.shape), _flatten(lay.stride)
    g = _prod(shp[i:j]) if i < j else 1
    new_shape = (g,) + tuple(shp[j:])
    out = Layout(new_shape, _stride_for_group(shp, str_, i, min(j, len(shp) - 1)) + tuple(str_[j:]))
    if isinstance(tensor_or_layout, Tensor):
        nv = Tensor(tensor_or_layout.base, out, tensor_or_layout.element,
                    tensor_or_layout.origin)
        nv.coord_offs = dict(getattr(tensor_or_layout, "coord_offs", {}))
        return nv
    return out


def _stride_for_group(shp, str_, i, j):
    # strides of the group's innermost (contiguous-by-construction in
    # flagship usage): outer stride of the group = stride of mode i
    return (str_[i],)


def zipped_divide(tensor_or_layout, tiler):
    """Divide into ((tile), (rest)) hierarchical layout; modes beyond the
    tiler rank ride the rest group unchanged (the implicit L mode)."""
    lay = tensor_or_layout.layout if isinstance(tensor_or_layout, Tensor) else tensor_or_layout
    tshp = _flatten(tiler)
    shp = _flatten(lay.shape)
    strd = _flatten(lay.stride)
    pad = (1,) * (len(shp) - len(tshp))
    tshp_padded = tuple(tshp) + pad
    rest = tuple(-(-s // t) for s, t in zip(shp, tshp_padded))
    tshp = tuple(tshp) + ()      # tile group keeps the tiler's own rank
    tile_layout = Layout(tuple(tshp), tuple(strd[:len(tshp)]))
    rest_shape = tuple(rest)
    if isinstance(tensor_or_layout, Tensor):
        nv = Tensor(tensor_or_layout.base,
                    Layout((tile_layout.shape, rest_shape),
                           (tile_layout.stride, _row_major_nested(rest_shape))),
                    tensor_or_layout.element, tensor_or_layout.origin)
        nv.coord_offs = dict(getattr(tensor_or_layout, "coord_offs", {}))
        nv.swizzle = getattr(tensor_or_layout, "swizzle", None)
        return nv
    return Layout((tile_layout.shape, rest_shape),
                  (tile_layout.stride, _row_major_nested(rest_shape)))


def local_tile(tensor, tiler, coord):
    """Tile then select rest coords (None keeps the mode, int selects)."""
    tiled = zipped_divide(tensor, tiler)
    return _select_rest(tiled, coord)


def _select_rest(tiled, coord):
    lay = tiled.layout
    tile_shape, tile_stride = lay.shape[0], lay.stride[0]
    rest_shape_f = _flatten(lay.shape[1])
    rest_stride_f = _flatten(lay.stride[1])
    coord_f = coord if isinstance(coord, (tuple, list)) else (coord,)
    keep_shape, keep_stride = [], []
    for r, c in zip(rest_shape_f, coord_f):
        if c is None:
            keep_shape.append(r)
            keep_stride.append(None)  # patched below
    # strides for kept modes: recompute row-major over kept shape
    ks = [k for k in keep_shape]
    strides, acc = [], 1
    for d in reversed(ks):
        strides.append(acc)
        acc *= d
    strides.reverse()
    # flat top-level modes: (tile..., kept rest...) — official profile
    # convention for local_tile (size(mode=[3]) indexes loopK directly)
    new_lay = Layout(tuple(_flatten(tile_shape)) + tuple(keep_shape),
                     tuple(_flatten(tile_stride)) + tuple(strides))
    nv = Tensor(tiled.base, new_lay, tiled.element, tiled.origin)
    nv.coord_offs = dict(getattr(tiled, "coord_offs", {}))
    nv.swizzle = getattr(tiled, "swizzle", None)
    return nv


def slice_(layout_or_tensor, coord):
    """Select sub-modes of a top-level shape with None/int."""
    lay = layout_or_tensor.layout if isinstance(layout_or_tensor, Tensor) else layout_or_tensor
    shp = _flatten(lay.shape)
    strd = _flatten(lay.stride)
    coord_f = coord if isinstance(coord, (tuple, list)) else (coord,)
    keep_s, keep_d = [], []
    for s, d, c in zip(shp, strd, coord_f):
        if c is None:
            keep_s.append(s)
            keep_d.append(d)
        elif not isinstance(c, int):
            raise NotImplementedError("dynamic slice_ coord")
    out = Layout(tuple(keep_s), tuple(keep_d))
    if isinstance(layout_or_tensor, Tensor):
        return Tensor(layout_or_tensor.base, out, layout_or_tensor.element,
                      layout_or_tensor.origin)
    return out


def size(x, mode=None):
    if isinstance(x, Tensor):
        x = x.layout
    if isinstance(x, Layout):
        if mode is not None:
            k = mode[0] if isinstance(mode, (list, tuple)) else mode
            return _prod(_flatten(x.shape[k:k + 1]))
        return x.size
    if isinstance(x, (tuple, list)):
        return _prod(_flatten(x))
    return int(x)


def shape(x):
    return x.shape if isinstance(x, (Tensor, Layout)) else x


# ------------------------------------------------------------------ TMA atom
class TmaAtom:
    """make_tiled_tma_atom result: recipe + cta layout + tma tensor view."""

    def __init__(self, op, tma_tensor, cta_layout, smem_tile_layout, box):
        self.op = op                    # CopyBulkTensorTileG2SOp / S2GOp / ...
        self.tma_tensor = tma_tensor    # gmem tensor viewed in TMA coords
        self.cta_layout = cta_layout    # Layout of participating CTAs
        self.smem_tile_layout = smem_tile_layout
        self.box = box                  # fastest-first box dims


class CopyBulkTensorTileG2SOp:
    pass


class CopyBulkTensorTileS2GOp:
    pass


def make_tiled_tma_atom(op, gmem_tensor, smem_tile_layout, cta_layout=None,
                        tiler=None):
    """Build a TMA atom over a gmem tensor.

    smem_tile_layout: the (tile) layout the SMEM staging uses; its flat
    shape (slowest-first) is the TMA box (fastest-first reversed).
    """
    box_slow = _flatten(smem_tile_layout.shape)[:2] if smem_tile_layout.rank > 1 \
        else _flatten(smem_tile_layout.shape)
    box = tuple(reversed(box_slow))
    # tma tensor: gmem viewed as (box_tile, rest) hierarchy
    tma_tensor = zipped_divide(gmem_tensor, tuple(reversed(box)))
    return TmaAtom(op, tma_tensor, cta_layout or make_layout((1, 1)),
                   smem_tile_layout, box)


class TmaSmemView:
    """tAsA from tma_partition: staged smem destination; [(None, stage)]
    windows the array at stage (stage may be SSA)."""

    def __init__(self, arr, elems_per_stage):
        self.arr = arr                  # SmemArray
        self.elems_per_stage = elems_per_stage

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        slots = [c for c in idx if c is not None]
        assert len(slots) == 1, f"tma smem subscript {idx}"
        from . import builtins as _b

        return _b.smem_stage(self.arr, slots[0], self.elems_per_stage)

    def data_ptr(self):
        return self.arr.ptr


class TmaGmemView:
    """tAgA/bSG_gD from tma_partition: gmem side carrying tile-unit
    offsets per logical dim. Subscripts select progressively: an optional
    leading TMA slot, then one coordinate per remaining dim (None keeps
    the dim, SSA/int records the offset and drops it)."""

    def __init__(self, atom, dim_names, tile_sizes):
        self.atom = atom
        self.dims = tuple(dim_names)    # e.g. ('m','k','l')
        self.tiles = tuple(tile_sizes)  # per-dim tile extent in elements
        self.offs = {}                  # dim -> SSA/int (tile units)

    @property
    def shape(self):
        # ((atom), (rest)) descriptor profile for epilogue tiling queries
        return (self.tiles, (1, 1))

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(idx) == len(self.dims) + 1:
            idx = idx[1:]               # drop the TMA slot
        if len(idx) == 2 and isinstance(idx[0], int) and \
                isinstance(idx[1], (tuple, list)):
            idx = tuple(idx[1])         # (tile_idx, coords) hierarchical form
        if len(idx) == 2 and idx[0] is None and \
                isinstance(idx[1], (tuple, list)):
            grp = tuple(idx[1])         # (None, coords) group form
            # shorter groups map onto the leading dims (epi sub-tiling)
            idx = grp + (None,) * (len(self.dims) - len(grp))
        if len(idx) != len(self.dims):
            raise NotImplementedError(
                f"tma gmem subscript {idx} over dims {self.dims}")
        keep = [(d, t) for d, t, c in zip(self.dims, self.tiles, idx)
                if c is None]
        nv = TmaGmemView(self.atom,
                         tuple(d for d, _ in keep),
                         tuple(t for _, t in keep))
        nv.offs = dict(self.offs)
        for d, c in zip(self.dims, idx):
            if c is None:
                continue
            if d in nv.offs:
                nv.offs[d] = _add_offsets(nv.offs[d], c)
            else:
                nv.offs[d] = c
        return nv


def _add_offsets(a, b):
    """Combine tile-unit offsets (int fold; SSA via i32 arith)."""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    from .emitter import SSA as _SSA

    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    from . import builtins as _b

    e = _b._emitter()

    def _i32(v):
        if isinstance(v, _SSA):
            return v if v.type == "i32" else e.ssa(
                "i32", f"arith.index_cast {v.name} : {v.type} to i32")
        return e.ssa("i32", f"arith.constant {int(v)} : i32")

    return _b.add_i32(_i32(a), _i32(b))


class TmaPartitioned:
    """(smem_view, gmem_view) pair from tma_partition."""

    def __init__(self, smem_view, gmem_view, cta_coord_ssa, box):
        self.smem_view = smem_view
        self.gmem_view = gmem_view
        self.cta_coord_ssa = cta_coord_ssa
        self.box = box

    def __iter__(self):
        return iter((self.smem_view, self.gmem_view))

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return (self.smem_view, self.gmem_view)[i]


def tma_partition(atom, cta_coord, cta_layout, smem_tensor_grouped,
                  gmem_tma_grouped):
    """Project smem/gmem grouped views onto this CTA's tile coordinates.

    cta_coord: SSA i32 (or int) index of this CTA along the atom's
    cta layout; used as the box row coordinate for the TMA copy.
    """
    return TmaPartitioned(smem_tensor_grouped, gmem_tma_grouped,
                          cta_coord, atom.box)


# ------------------------------------------------------------------ TiledMma
class MmaF16BF16OpAtom:
    shape_mnk = (16, 8, 16)


class TiledMma:
    """make_tiled_mma(MmaF16BF16Op, atom_layout) — m16n8k16 warp atom tiled
    over the CTA by atom_layout (M-tiles, N-tiles, _)."""

    def __init__(self, op, atom_layout=None, num_warps=None):
        self.op = op
        self.atom_layout = atom_layout or (1, 1, 1)
        am, an, ak = op.shape_mnk
        self.tile_mn = (am * self.atom_layout[0], an * self.atom_layout[1])
        self.num_warps = num_warps or _prod(_flatten(self.atom_layout))

    def get_slice(self, tidx_ssa):
        return MmaThreadSlice(self, tidx_ssa)

    def make_fragment_A(self, view):
        return FragmentViewA(view)

    def make_fragment_B(self, view):
        return FragmentViewB(view)


class MmaThreadSlice:
    def __init__(self, tiled_mma, tidx_ssa):
        self.mma = tiled_mma
        self.tidx = tidx_ssa

    def partition_A(self, sA):
        return PartitionedMmaOperand("A", sA, self.mma, self.tidx)

    def partition_B(self, sB):
        return PartitionedMmaOperand("B", sB, self.mma, self.tidx)

    def partition_C(self, gC):
        return PartitionedAccumulator(gC, self.mma, self.tidx)


class PartitionedMmaOperand:
    def __init__(self, which, tile, mma, tidx):
        self.which = which
        self.tile = tile
        self.mma = mma
        self.tidx = tidx

    def __getitem__(self, idx):
        return _mma_operand_slice(self, idx)


def _mma_operand_slice(part, idx):
    """tXsA[None, None, None, 0]-style: returns the (atom-k) slice meta."""
    return MmaOperandAtomView(part, idx)


class MmaOperandAtomView:
    def __init__(self, part, idx):
        self.part = part
        self.idx = idx


def _frag_atom_grid(part):
    """(MMA_M, MMA_N, MMA_K) atom grid of a warp's fragment over its tile:
    warp tile from tiled_mma, K extent from the smem tile layout."""
    mma = part.mma
    am, an, ak = getattr(mma.op, "shape_mnk", (16, 8, 16))
    warp_m, warp_n = mma.tile_mn
    tile = part.tile
    lay = tile.layout if hasattr(tile, "layout") else tile
    flat = _flatten(lay.shape)
    k_ext = max(flat) if len(flat) >= 2 else ak   # (m,k)/(n,k): k is 2nd
    mma_k = max(1, k_ext // ak)
    return warp_m // am, warp_n // an, mma_k


class FragmentViewA:
    """make_fragment_A: per-thread operand registers, keyed by k_block.
    slots[k] = [SSA i32] * (4 * MMA_M) — ldmatrix regs, bitcast at mma."""

    def __init__(self, view):
        self.view = view
        self.slots: dict[int, list] = {}
        self.retile_tc = None

    def size(self, mode=None):
        mm, _, mk = _frag_atom_grid(self.view.part)
        grid = (mm, 1, mk)
        return grid[mode[0]] if mode is not None else mm * mk

    @property
    def mma(self):
        return self.view.part.mma

    def __getitem__(self, idx):
        # (None, None, k) — k_block selection for gemm/copy targets
        k = idx[2] if isinstance(idx, tuple) and len(idx) == 3 else idx
        return FragK(self, int(_const_of(k)))


class FragmentViewB(FragmentViewA):
    def size(self, mode=None):
        _, mn, mk = _frag_atom_grid(self.view.part)
        grid = (1, mn, mk)
        return grid[mode[0]] if mode is not None else mn * mk


class FragK:
    """tCrX[None, None, k] — a fragment sliced at one k block."""

    def __init__(self, frag, k):
        self.frag = frag
        self.k = k


def _const_of(v):
    from .emitter import SSA as _SSA

    if isinstance(v, _SSA):
        raise ValueError(f"k-block index must be constexpr, got SSA {v.name}")
    return int(v)


# ------------------------------------------------- r2s epilogue copies
class TiledCopyCAtom:
    """make_tiled_copy_C_atom(stmatrix_atom, tiled_mma)."""

    def __init__(self, atom, mma):
        self.atom = atom
        self.mma = mma


class TiledCopyR2S:
    """make_tiled_copy_S(r2s_atom, tiled_copy_C): rmem->smem epilogue copy
    with the mma accumulator (trait-table C) value layout."""

    def __init__(self, atom, tiled_copy_c):
        self.atom = atom
        self.tiled_copy_c = tiled_copy_c
        self.mma = tiled_copy_c.mma
        self.tidx = None

    def get_slice(self, tidx):
        self.tidx = tidx
        return self

    def partition_D(self, sC):
        return R2SSmemView(self, sC)

    def partition_S(self, sC):
        return R2SSmemView(self, sC)

    def retile(self, accumulators):
        return AccumRetile(self, accumulators)


class R2SSmemView:
    """tRS_sD: epilogue smem tile (subscriptable to the epi buffer)."""

    def __init__(self, tc, tile, stage=0):
        self.tc = tc
        self.tile = tile              # Tensor view over staged sC
        self.stage = stage

    @property
    def shape(self):
        mm, mn = _r2s_grid(self.tc)
        return (4, mm, mn, 1)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        stage = [c for c in idx if c is not None]
        return R2SSmemView(self.tc, self.tile, stage[0] if stage else 0)


class AccumRetile:
    """tRS_rAcc = tiled_copy_r2s.retile(accumulators): the fragment in the
    r2s value order (== the gemm slot order by construction)."""

    def __init__(self, tc, frag):
        self.tc = tc
        self.frag = frag

    def __len__(self):
        return self.frag.count

    @property
    def shape(self):
        mm, mn = _r2s_grid(self.tc)
        return (4, mm, mn)

    @property
    def count(self):
        return self.frag.count

    def __getitem__(self, i):
        if isinstance(i, (tuple, list)):
            from .kernel_objects import _FragSlice

            return _FragSlice.from_modes(self.frag, i, shape=self.shape)
        return self.frag[i]


def _r2s_grid(tc):
    """Return the per-warp accumulator atom grid used by R2S copies."""
    mma = getattr(tc, "mma", None)
    am, an, ak = getattr(getattr(mma, "op", None), "shape_mnk",
                         (16, 8, 16))
    warp_m, warp_n = getattr(mma, "tile_mn", (32, 32))
    # The SM120 block-scaled (m16n8k64) atom layout is four warps along M
    # and two along N over a 128x128 CTA tile: one warp owns 32x64.
    if ak >= 64:
        warp_m, warp_n = 32, 64
    return warp_m // am, warp_n // an


# ------------------------------------------------- smem->rmem tiled copies
class TiledCopyAB:
    """cute.make_tiled_copy_A/B(ldmatrix_atom, tiled_mma): smem->rmem copy
    whose value layout matches the mma operand fragment geometry (the
    lowering emits the trait-table ldmatrix sequence directly)."""

    def __init__(self, which, atom, mma):
        self.which = which          # 'A' | 'B'
        self.atom = atom            # CopyAtom marker (LdMatrix8x8x16bOp)
        self.mma = mma
        self.tidx = None            # bound by get_slice

    def get_slice(self, tidx):
        tc = TiledCopyAB(self.which, self.atom, self.mma)
        tc.tidx = tidx
        # cute.copy is invoked with the ORIGINAL object in dense_gemm —
        # remember the bound thread index on it too
        self.tidx = tidx
        return tc

    def partition_S(self, tile):
        return SmemCopyView(self, tile)

    def retile(self, frag_view):
        frag_view.retile_tc = self
        return frag_view


class SmemCopyView:
    """tCsA_copy_view = thr_copy.partition_S(sA): staged smem operand tile.
    [None, None, None, stage] -> stage window; [None, None, k] -> src."""

    def __init__(self, tc, tile, stage=None):
        self.tc = tc
        self.tile = tile            # Tensor view over SmemArray (staged)
        self.stage = stage          # int | SSA | None

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(idx) == 4:
            return SmemCopyView(self.tc, self.tile, idx[3])
        if len(idx) == 3:
            return SmemCopySrc(self.tc, self.tile, self.stage, idx[2])
        raise NotImplementedError(f"copy view subscript {idx}") 

class SmemCopySrc:
    """Concrete (stage, k_block) source ready for cute.copy lowering."""

    def __init__(self, tc, tile, stage, k):
        self.tc = tc
        self.tile = tile
        self.stage = stage
        self.k = k


class PartitionedAccumulator:
    """thr_mma.partition_C view: per-thread (4, MMA_M, MMA_N) register grid
    (4 = f32 regs per m16n8 atom, from the mma_atoms trait geometry)."""

    def __init__(self, tile, mma, tidx):
        self.tile = tile
        self.mma = mma
        self.tidx = tidx

    @property
    def shape(self):
        am, an, _ = self.mma.op.shape_mnk if hasattr(self.mma.op, "shape_mnk") \
            else (16, 8, 16)
        warp_m = self.mma.tile_mn[0]
        warp_n = self.mma.tile_mn[1]
        return (4, warp_m // am, warp_n // an)

    @property
    def count(self):
        c = 1
        for d in self.shape:
            c *= int(d)
        return c


def make_tiled_mma(op, atom_layout=None, num_warps=None):
    return TiledMma(op, atom_layout, num_warps)


# ------------------------------------------------ in-kernel views (M6.2)
class KernelTensorView:
    """A Tensor view usable INSIDE the traced kernel: hierarchical layout
    (tile, rest) with an SSA-augmented origin. Subscripting with rest
    coords (None/int/SSA) selects a concrete tile base."""

    def __init__(self, base_ssa, layout, element, origin=None, smem=None):
        self.base_ssa = base_ssa    # SSA ptr (gmem or smem) or SmemArray
        self.layout = layout        # Layout((tile),(rest))
        self.element = element
        self.origin = origin or ()  # tuple of SSA/int element offsets
        self.smem = smem            # SmemArray if shared

    @property
    def shape(self):
        return self.layout.shape

    @property
    def type(self):
        return f"cutlass.Tensor(element={getattr(self.element, 'name', '?')}, shape={self.shape})"


def make_kernel_view(base_ssa, layout, element, smem=None):
    return KernelTensorView(base_ssa, layout, element, smem=smem)


def view_select(view, rest_coord):
    """gX[(None,...), (bidx,...)]-style rest selection -> concrete tile view
    whose origin carries the SSA element offset."""
    rest_shape = _flatten(view.layout.shape[1])
    rest_strides_f = _flatten(view.layout.stride[1])
    coord = rest_coord if isinstance(rest_coord, (tuple, list)) else (rest_coord,)
    off_terms = []
    for s, d, c in zip(rest_shape, rest_strides_f, coord):
        if c is None or isinstance(c, int) and c == 0:
            continue
        off_terms.append((c, d))
    return KernelTensorView(view.base_ssa, Layout(view.layout.shape[0],
                                                  view.layout.stride[0]),
                            view.element, origin=tuple(view.origin) +
                            tuple(off_terms), smem=view.smem)

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


def _nest(x):
    if isinstance(x, (tuple, list)):
        return tuple(_nest(i) for i in x)
    return int(x)


def _render(t):
    if isinstance(t, tuple):
        return "(" + ",".join(_render(x) for x in t) + ")"
    return str(t)


def make_layout(shape, stride=None, order=None):
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
    def type(self):
        return f"cutlass.Tensor(element={getattr(self.element, 'name', '?')}, shape={self.shape})"


def make_tensor(base, layout, element):
    return Tensor(base, layout, element)


# ------------------------------------------------------- algebra (host meta)
def group_modes(tensor_or_layout, i: int, j: int):
    """Collapse modes i..j (inclusive) of the top-level shape into one."""
    lay = tensor_or_layout.layout if isinstance(tensor_or_layout, Tensor) else tensor_or_layout
    shp, str_ = _flatten(lay.shape), _flatten(lay.stride)
    if i != 0:
        raise NotImplementedError("group_modes from mode 0 only (as used by flagship)")
    g = _prod(shp[i:j + 1])
    new_shape = (g,) + tuple(shp[j + 1:])
    new_stride = tuple(str_[i:j + 1]) and _stride_for_group(shp, str_, i, j) + tuple(str_[j + 1:])
    out = Layout(new_shape, _stride_for_group(shp, str_, i, j) + tuple(str_[j + 1:]))
    if isinstance(tensor_or_layout, Tensor):
        return Tensor(tensor_or_layout.base, out, tensor_or_layout.element,
                      tensor_or_layout.origin)
    return out


def _stride_for_group(shp, str_, i, j):
    # strides of the group's innermost (contiguous-by-construction in
    # flagship usage): outer stride of the group = stride of mode i
    return (str_[i],)


def zipped_divide(tensor_or_layout, tiler):
    """Divide into ((tile), (rest)) hierarchical layout."""
    lay = tensor_or_layout.layout if isinstance(tensor_or_layout, Tensor) else tensor_or_layout
    tshp = _flatten(tiler)
    shp = _flatten(lay.shape)
    strd = _flatten(lay.stride)
    rest = tuple(-(-s // t) for s, t in zip(shp, tshp))
    tile_layout = Layout(tuple(tshp), tuple(strd[:len(tshp)]))
    rest_shape = tuple(rest)
    if isinstance(tensor_or_layout, Tensor):
        return Tensor(tensor_or_layout.base,
                      Layout((tile_layout.shape, rest_shape),
                             (tile_layout.stride, _row_major_nested(rest_shape))),
                      tensor_or_layout.element, tensor_or_layout.origin)
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
    new_lay = Layout((tile_shape, tuple(keep_shape)),
                     (tile_stride, tuple(strides)))
    return Tensor(tiled.base, new_lay, tiled.element, tiled.origin)


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


class TmaPartitioned:
    """(smem_view, gmem_view) pair from tma_partition: carries the SSA
    coordinate of this CTA within the cluster for box addressing."""

    def __init__(self, smem_view, gmem_view, cta_coord_ssa, box):
        self.smem_view = smem_view
        self.gmem_view = gmem_view
        self.cta_coord_ssa = cta_coord_ssa
        self.box = box


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


class FragmentViewA:
    def __init__(self, view):
        self.view = view


class FragmentViewB:
    def __init__(self, view):
        self.view = view


class PartitionedAccumulator:
    def __init__(self, tile, mma, tidx):
        self.tile = tile
        self.mma = mma
        self.tidx = tidx


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

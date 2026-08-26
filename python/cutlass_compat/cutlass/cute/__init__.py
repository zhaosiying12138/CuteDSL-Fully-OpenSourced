"""cutlass.cute — compat surface backed by the self stack frontend.

The real logic lives in self_cutedsl.frontend; this module exposes the
official-API names (cute.kernel, cute.jit, cute.arch, cute.printf, ...).
"""
from __future__ import annotations



def kernel(fn):
    from self_cutedsl.frontend.jit import KernelFunction
    return KernelFunction(fn)


def jit(fn):
    from self_cutedsl.frontend.jit import JitFunction
    return JitFunction(fn)


def __getattr__(name):
    if name == 'arch':
        from self_cutedsl.frontend import builtins
        return builtins.arch
    if name in ('nvgpu', 'runtime', 'testing'):
        from importlib import import_module
        return import_module(f'cutlass.cute.{name}')
    if name == 'struct':
        # importing the submodule binds cutlass.cute.struct as a MODULE
        # attribute that would shadow this namespace on later accesses —
        # install the namespace as the explicit package attribute
        from cutlass.cute.struct import _StructNamespace
        import cutlass.cute as _cc

        ns = _StructNamespace()
        setattr(_cc, "struct", ns)
        return ns
    raise AttributeError(name)


def printf(fmt: str, *args) -> None:
    from self_cutedsl.frontend import builtins
    builtins.printf(fmt, *args)



def compile(fn, *args, **options):
    from self_cutedsl.frontend.jit import compile_function

    return compile_function(fn, *args, **options)


# ------------------------------------------------------------------ markers
class Tensor:
    """Annotation marker: kernel parameter is a device tensor (ptr ABI)."""


class Shape:
    """Annotation marker: compile-time shape meta parameter."""


class Layout:
    """Annotation marker: compile-time layout meta parameter."""


class Coord:
    pass


class ComposedLayout:
    """Annotation marker: staged/swizzled smem layout meta."""


class CopyAtom:
    """Annotation marker: copy atom object (TMA/CopyUniversal)."""


class TiledMma:
    """Annotation marker: tiled MMA object."""


class Shape:
    pass


class Swizzle:
    pass


class TensorSSA:
    pass


class TmaTensor:
    """Annotation marker: kernel parameter is a TMA descriptor pointer."""


# ------------------------------------------------------------ layout utils
def make_layout(shape, stride=None):
    from self_cutedsl.frontend.layout import CuteLayout

    return CuteLayout(shape, stride)


def make_ordered_layout(shape, order):
    from self_cutedsl.frontend.meta import make_ordered_layout as _m

    return _m(shape, order)


def make_layout_tv(thr_layout, val_layout):
    from self_cutedsl.frontend.tiled import make_layout_tv as _m

    return _m(thr_layout, val_layout)


def size(x, mode=None):
    from self_cutedsl.frontend.kernel_objects import Fragment as _F
    from self_cutedsl.frontend.meta import cute_size

    if isinstance(x, _F):
        return x.count
    return cute_size(x, mode)


# ------------------------------------------------------------ tensor utils
def zipped_divide(m, tiler=None):
    from self_cutedsl.frontend.meta import zipped_divide as _m

    return _m(m, tiler)


def make_identity_tensor(shape):
    from self_cutedsl.frontend.meta import make_identity_tensor as _m

    return _m(shape)


# --------------------------------------------------------------- copies/mma
def make_copy_atom(op, element_type, **kwargs):
    from self_cutedsl.frontend import builtins

    return builtins.make_copy_atom(op, element_type, **kwargs)


def shape(x):
    return getattr(x, "shape", x)


def make_tiled_copy_C_atom(atom, tiled_mma):
    from self_cutedsl.frontend.cute_objects import TiledCopyCAtom

    return TiledCopyCAtom(atom, tiled_mma)


def make_tiled_copy_S(atom, tiled_copy_c):
    from self_cutedsl.frontend.cute_objects import TiledCopyR2S

    return TiledCopyR2S(atom, tiled_copy_c)


def make_tiled_copy(atom, thr_layout, val_shape=None):
    """cute.make_tiled_copy(atom, TV-layout, val-shape): SF smem->rmem copy
    descriptor; lowered at the copy site from the SF trait geometry."""
    return _SF_copy(atom, thr_layout, val_shape)


class _SF_copy:
    def __init__(self, atom, thr_layout, val_shape):
        self.atom = atom
        self.thr_layout = thr_layout
        self.val_shape = val_shape
        self.tidx = None

    def get_slice(self, tidx):
        self.tidx = tidx
        return self

    def partition_S(self, view):
        return _SF_copy_view(self, view)

    def retile(self, frag):
        return frag


class _SF_copy_view:
    """tCsSF staged view + the copy descriptor; subscripts select the
    k/stage window like the operand copy views."""

    def __init__(self, tc, tile, stage=None, k=None):
        self.tc = tc
        self.tile = tile
        self.stage = stage
        self.k = k

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        slots = [c for c in idx if c is not None]
        if len(slots) >= 2:
            return _SF_copy_view(self.tc, self.tile, slots[-2], slots[-1])
        if len(slots) == 1:
            return _SF_copy_view(self.tc, self.tile, self.stage, slots[0])
        return self


def make_tiled_copy_A(atom, tiled_mma):
    from self_cutedsl.frontend.cute_objects import TiledCopyAB

    return TiledCopyAB("A", atom, tiled_mma)


def make_tiled_copy_B(atom, tiled_mma):
    from self_cutedsl.frontend.cute_objects import TiledCopyAB

    return TiledCopyAB("B", atom, tiled_mma)


def make_tiled_copy_tv(atom, thr_layout, val_layout):
    from self_cutedsl.frontend import builtins

    return builtins.make_tiled_copy_tv(atom, thr_layout, val_layout)


def copy(atom, src, dst, pred=None, **kw):
    from self_cutedsl.frontend import builtins
    from self_cutedsl.frontend.cute_objects import TiledCopyAB, TmaAtom

    from self_cutedsl.frontend.cute_objects import TiledCopyR2S

    if type(atom).__name__ == "_SF_copy":
        builtins.copy_sf(atom, src, dst)
        return
    if isinstance(atom, TiledCopyAB):
        builtins.copy_tiled(atom, src, dst)
        return
    if isinstance(atom, TiledCopyR2S):
        builtins.copy_r2s(atom, src, dst)
        return
    if isinstance(atom, TmaAtom):
        _tma_copy(atom, src, dst, **kw)
        return
    builtins.copy(atom, src, dst, pred)


def make_tensor(iterator, layout):
    """cute.make_tensor(iter, layout): rebind a host tensor's layout (the
    iterator is the source meta/tensor view)."""
    from self_cutedsl.frontend.meta import TensorMeta
    from self_cutedsl.frontend.cute_objects import Tensor as _HT

    src = getattr(iterator, "_torch", iterator)
    if isinstance(iterator, TensorMeta):
        nm = TensorMeta(iterator._torch, iterator.element_type,
                        _shape_of(layout), _strides_of(layout, iterator))
        return nm
    v = _HT(getattr(iterator, "base", None), layout,
            getattr(iterator, "element", None))
    return v


def _shape_of(layout):
    from self_cutedsl.frontend.cute_objects import _flatten

    return tuple(_flatten(layout.shape))


def _strides_of(layout, like):
    from self_cutedsl.frontend.cute_objects import _flatten

    try:
        return tuple(_flatten(layout.stride))
    except Exception:
        return None


def filter_zeros(x):
    """Drop zero-stride modes (tma partition views in this model are
    already minimal descriptors — identity for them; layouts keep their
    nonzero-stride modes)."""
    from self_cutedsl.frontend.cute_objects import Layout as _L, Tensor as _T

    if hasattr(x, "outer") and not hasattr(x, "shape"):   # staged layout
        from cutlass.utils import ComposedLayoutStaged as _S

        return _S(filter_zeros(x.outer), x.stages, x.inner)
    if hasattr(x, "shape") and not isinstance(x, (_L, _T)) and \
            hasattr(x, "stride"):
        # host-meta layout-ish object (CuteLayout etc.)
        return x
    if isinstance(x, _L):
        shp = x.shape if isinstance(x.shape, tuple) else (x.shape,)
        strd = x.stride if isinstance(x.stride, tuple) else (x.stride,)
        keep = [(a, b) for a, b in zip(shp, strd)
                if not (isinstance(b, int) and b == 0)]
        return _L(tuple(a for a, _ in keep), tuple(b for _, b in keep))
    return x


def rank(x):
    """Mode count of a tensor/layout's top level."""
    from self_cutedsl.frontend.cute_objects import Layout as _L, Tensor as _T
    from self_cutedsl.frontend.meta import TensorMeta as _M

    if isinstance(x, _T):
        shp = x.layout.shape
    elif isinstance(x, _L):
        shp = x.shape
    elif isinstance(x, _M):
        shp = x.shape
    else:
        shp = x
    return len(shp) if isinstance(shp, tuple) else 1


def append(tup, *vals):
    if not isinstance(tup, tuple):
        tup = (tup,)
    return tup + tuple(vals)


def recast_tensor(tensor, dtype=None, **kw):
    """View a tensor under a different element type (same storage)."""
    t = tensor
    if dtype is not None:
        t.element = dtype
        t.recast_dtype = dtype
    return t


def tile_to_shape(atom_layout, shape, order=None, **kw):
    """Tile a layout atom up to shape (row-major rest by default)."""
    from self_cutedsl.frontend.cute_objects import Layout as _L, _flatten, _prod

    a_shp = _flatten(atom_layout.shape)
    a_str = _flatten(atom_layout.stride)
    tgt = _flatten(shape)
    # trailing stage mode: (..., stages) -> ComposedLayoutStaged
    stages = 1
    if len(tgt) == len(a_shp) + 1:
        stages = int(tgt[-1])
        tgt = tgt[:-1]
    reps = [-(-t // a) for t, a in zip(tgt, a_shp)] +         [1] * (len(a_shp) - len(tgt))
    # atom tiled up to the per-stage tile: (tile, rest-not-used) — the
    # per-stage layout is just the atom repeated over the tile extents
    tile_s = tuple(tgt)
    tile_d = []
    acc = 1
    from self_cutedsl.frontend.cute_objects import _prod as _p
    for a, d in zip(reversed(a_shp), reversed(a_str)):
        tile_d.append(acc if d == _p(a_shp[a_shp.index(a):]) or True else d)
        acc = None if acc is None else acc
    # simple row-major strides over the per-stage shape
    strides, acc = [], 1
    for d in reversed(tile_s):
        strides.append(acc)
        acc *= int(d)
    strides.reverse()
    from cutlass.utils import ComposedLayoutStaged
    return ComposedLayoutStaged(_L(tile_s, tuple(strides)), stages)


def _prod_rest(rest):
    from self_cutedsl.frontend.cute_objects import _flatten

    out, acc = [], 1
    for d in reversed(rest):
        out.append(acc)
        acc *= d
    return list(reversed(out))


def group_modes(x, i, j):
    from self_cutedsl.frontend import cute_objects as co
    from self_cutedsl.frontend.meta import TensorMeta

    if isinstance(x, TensorMeta):
        return x.group_modes_meta(i, j)
    try:
        return co.group_modes(x, i, j)
    except (AttributeError, NotImplementedError):
        return x      # opaque fragment descriptors keep their identity


def _tma_copy(atom, src, dst, tma_bar_ptr=None, mcast_mask=None):
    from self_cutedsl.frontend import builtins

    builtins.tma_copy_partitioned(atom, src, dst, tma_bar_ptr, mcast_mask)


def make_fragment_like(part):
    from self_cutedsl.frontend import builtins

    return builtins.make_fragment_like(part)


def make_rmem_tensor(shape, dtype):
    from self_cutedsl.frontend import builtins

    return builtins.make_rmem_tensor(shape, dtype)


def elem_less(coord, shape):
    from self_cutedsl.frontend import builtins

    return builtins.elem_less(coord, shape)


# --------------------------------------------------------------- smem + mma
def make_smem_array(name, count, element=None):
    from self_cutedsl.frontend import builtins

    return builtins.make_smem_array(name, count, element)


def ldmatrix(smem, row_ssa, col_elems=0, num=4, trans=False):
    from self_cutedsl.frontend import builtins

    return builtins.ldmatrix(smem, row_ssa, col_elems, num, trans)


def make_tiled_mma(atom, atom_layout=None, permutation_mnk=None):
    from self_cutedsl.frontend import builtins

    tm = builtins.make_tiled_mma(atom, atom_layout)
    if permutation_mnk is not None:
        pm = tuple(int(p) for p in permutation_mnk)
        tm.permutation_mnk = pm
        # the permutation defines the warp-level tile extent (e.g. the
        # ldmatrix.x4 n-pairing doubles the n atoms per warp)
        tm.tile_mn = (pm[0], pm[1])
    return tm


def gemm(tiled_mma, acc, a_frag, b_frag, *rest):
    from self_cutedsl.frontend import builtins
    from self_cutedsl.frontend.cute_objects import FragK

    if isinstance(a_frag, (list, tuple)):   # block-scaled [data, SF] form
        builtins.gemm_bs(tiled_mma, acc, a_frag, b_frag)
        return acc
    if isinstance(a_frag, FragK):      # 5-arg dense_gemm form
        builtins.gemm_tv(tiled_mma, acc, a_frag, b_frag)
        return acc
    return builtins.gemm(tiled_mma, acc, a_frag, b_frag)


def extract_frag(ld_res, idx):
    from self_cutedsl.frontend import builtins

    return builtins.extract_frag(ld_res, idx)


def zero_f32():
    from self_cutedsl.frontend.builtins import _emitter

    return _emitter().ssa("f32", "arith.constant 0.0 : f32", "float32")


def sync_threads():
    from self_cutedsl.frontend.builtins import _emitter

    _emitter().barrier()


# --------------------------------------------------------------- TMA / pipeline
def make_mbarrier(name, count):
    from self_cutedsl.frontend import builtins

    return builtins.make_mbarrier(name, count)


def make_smem_tile(name, count, element=None):
    from self_cutedsl.frontend import builtins

    return builtins.make_smem_tile(name, count, element)


def tma_load(tma, smem, bar, coords):
    from self_cutedsl.frontend import builtins

    builtins.tma_load(tma, smem, bar, coords)


def tma_store(tma, smem, coords):
    from self_cutedsl.frontend import builtins

    builtins.tma_store(tma, smem, coords)


def mbarrier_arrive_expect_tx(bar, tx_bytes):
    from self_cutedsl.frontend import builtins

    builtins.mbarrier_arrive_expect_tx(bar, tx_bytes)


def mbarrier_try_wait_parity(bar, phase):
    from self_cutedsl.frontend import builtins

    builtins.mbarrier_try_wait_parity(bar, phase)


def setmaxnreg(value, increase=True):
    from self_cutedsl.frontend import builtins

    builtins.setmaxnreg(value, increase)


def named_barrier_arrive(id_, count):
    from self_cutedsl.frontend import builtins

    builtins.named_barrier_arrive(id_, count)


def named_barrier_sync(id_, count):
    from self_cutedsl.frontend import builtins

    builtins.named_barrier_sync(id_, count)


def smem_stage(smem_arr, stage, elems_per_stage):
    from self_cutedsl.frontend import builtins

    return builtins.smem_stage(smem_arr, stage, elems_per_stage)


def bool_to_i32(b):
    from self_cutedsl.frontend import builtins

    return builtins.bool_to_i32(b)


def add_i32(a, b):
    from self_cutedsl.frontend import builtins

    return builtins.add_i32(a, b)


def rem_i32(a, b):
    from self_cutedsl.frontend import builtins

    return builtins.rem_i32(a, b)


def lt_i32(a, b):
    from self_cutedsl.frontend import builtins

    return builtins.lt_i32(a, b)


def const_i32(v):
    from self_cutedsl.frontend import builtins

    return builtins.const_i32(v)


def idx_to_i32(v):
    from self_cutedsl.frontend import builtins

    return builtins.idx_to_i32(v)


def mul_i32(a, b):
    from self_cutedsl.frontend import builtins

    return builtins.mul_i32(a, b)


def sub_i32(a, b):
    from self_cutedsl.frontend import builtins

    return builtins.sub_i32(a, b)


def mbarrier_inval_and_init(bar, count):
    from self_cutedsl.frontend import builtins

    builtins.mbarrier_inval_and_init(bar, count)


def mbarrier_reinit(bar, count):
    from self_cutedsl.frontend import builtins

    builtins.mbarrier_reinit(bar, count)


def fence_and_sync():
    from self_cutedsl.frontend import builtins

    builtins.fence_and_sync()


def div_i32(a, b):
    from self_cutedsl.frontend import builtins

    return builtins.div_i32(a, b)


def fence_proxy():
    from self_cutedsl.frontend import builtins

    builtins.fence_proxy()


def make_barrier_array(name, count):
    from self_cutedsl.frontend import builtins

    return builtins.make_barrier_array(name, count)


def slice_(x, coord):
    """cute.slice_(layout|tuple|staged, coord): select modes."""
    from self_cutedsl.frontend.layout import CuteLayout, _flatten, _flatten_tuple
    try:
        from cutlass.utils import ComposedLayoutStaged
    except Exception:
        ComposedLayoutStaged = ()
    from self_cutedsl.frontend.emitter import SSA as _SSA
    if isinstance(x, _SSA):
        raise NotImplementedError(
            "slice_ on kernel SSA layouts requires the object-model path")
    if isinstance(x, ComposedLayoutStaged):
        # staged ((tile),(stage)) sliced on the TILE modes
        shp = _flatten_tuple(x.outer.shape)
        strd = _flatten_tuple(x.outer.stride)
        crd = coord if isinstance(coord, (tuple, list)) else (coord,)
        keep_s, keep_d = [], []
        for s0, d0, c in zip(shp, strd, crd):
            if c is None:
                keep_s.append(s0)
                keep_d.append(d0)
        outer = CuteLayout(tuple(keep_s), tuple(keep_d))
        return ComposedLayoutStaged(outer, x.stages, x.inner)
    if isinstance(x, tuple):
        crd = coord if isinstance(coord, (tuple, list)) else (coord,)
        out = tuple(v for v, c in zip(x, crd) if c is None)
        return out
    if isinstance(x, CuteLayout):
        shp = _flatten_tuple(x.shape)
        strd = _flatten_tuple(x.stride)
        crd = coord if isinstance(coord, (tuple, list)) else (coord,)
        keep_s, keep_d = [], []
        for s0, d0, c in zip(shp, strd, crd):
            if c is None:
                keep_s.append(s0)
                keep_d.append(d0)
        return CuteLayout(tuple(keep_s), tuple(keep_d))
    raise TypeError(f"slice_ on {type(x)}")


def cosize(x):
    from self_cutedsl.frontend.layout import CuteLayout, _flatten
    try:
        from cutlass.utils import ComposedLayoutStaged
    except Exception:
        ComposedLayoutStaged = ()
    if isinstance(x, ComposedLayoutStaged):
        shp = _flatten(x.outer.shape)
        n = 1
        for d in shp:
            n *= int(d)
        return n * max(1, int(x.stages))   # staged cosize spans all stages
    if isinstance(x, CuteLayout):
        # cosize = 1 + max index over size elements (row-major flatten)
        leaves = _flatten(x.stride)
        shp = _flatten(x.shape)
        n = 1
        idx = 0
        prod = 1
        # simple static evaluation
        m = 0
        for s0, d0 in zip(shp, leaves):
            m = max(m, d0 * (int(s0) - 1))
        return m + 1
    if isinstance(x, (tuple, list)):
        from self_cutedsl.frontend.layout import _prod
        return _prod(x)
    return int(x)


def size_in_bytes(dtype, layout):
    from self_cutedsl.frontend.layout import CuteLayout, _flatten, _prod
    from self_cutedsl.frontend.cute_objects import Layout as HostLayout

    if isinstance(layout, (CuteLayout, HostLayout)):
        n = _prod(_flatten(layout.shape))
    elif hasattr(layout, "outer"):          # ComposedLayoutStaged per-stage
        def _walk(t):
            if isinstance(t, (tuple, list)):
                r = 1
                for d in t:
                    r *= _walk(d)
                return r
            return int(t)

        n = _walk(layout.outer.shape)
    else:
        n = 1
    return n * int(getattr(dtype, "width", 16) // 8)


def make_layout_image_mask(cta_layout, coord, mode):
    """Multicast mask for the given mode at coord (host meta, cluster=1 ->
    mask 0 in the single-CTA profile)."""
    return 0


def make_rmem_tensor(shape, dtype):
    from self_cutedsl.frontend import builtins

    return builtins.make_rmem_tensor(shape, dtype)


def local_tile(m, tiler, coord):
    """local_tile(mA, (M,K), (None,None,None)): pad 2-mode operand metas
    with the implicit L=1 mode so the result carries (tile..., loopM,
    loopK/loopN, loopL) flat modes like the official profile."""
    from self_cutedsl.frontend import cute_objects as co
    from self_cutedsl.frontend.cute_objects import Tensor as HT, Layout

    base = m if hasattr(m, "layout") else m
    coord = tuple(coord)
    if isinstance(base, HT):
        lay = base.layout
    else:
        from self_cutedsl.frontend.kernel_objects import KernelTensor

        assert isinstance(base, KernelTensor), f"local_tile on {type(base)}"
        shp = tuple(base.shape)
        strd = tuple(getattr(base.meta, "stride", (1,) * len(shp)) or
                     (1,) * len(shp))
        lay = Layout(shp, strd)
        base = HT(None, lay, getattr(base.meta, "element_type", None))
    flat_s = list(co._flatten(lay.shape))
    flat_d = list(co._flatten(lay.stride))
    if len(flat_s) == 2 and len(coord) == 3:
        flat_s.append(1)
        flat_d.append(flat_s[0] * flat_d[0])
        base = HT(base.base, Layout(tuple(flat_s), tuple(flat_d)),
                  base.element, getattr(base, "origin", None))
    tiled = co.zipped_divide(base, tiler)
    return co._select_rest(tiled, coord)


def zipped_divide(m, tiler=None):
    from self_cutedsl.frontend import cute_objects as co

    return co.zipped_divide(m, tiler)

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

    def grid_dim(self):
        e = _emitter()
        return (e.ssa("index", "gpu.grid_dim x"),
                e.ssa("index", "gpu.grid_dim y"),
                e.ssa("index", "gpu.grid_dim z"))

    def warp_idx(self):
        """warp index as (wid, _, _) — tid / 32 (warp-uniform)."""
        e = _emitter()
        w = e.ssa("index", "arith.constant 32 : index")
        tid = e.thread_id("x")
        wid = e.idx_binop("arith.divsi", tid, w)
        return (wid, e.ssa("index", "arith.constant 0 : index"),
                e.ssa("index", "arith.constant 0 : index"))

    def setmaxregister_increase(self, byte_count):
        # warp-specialized register rebalance (constexpr byte count baked in)
        _emitter().raw(
            f'nvvm.inline_ptx "setmaxnreg.inc.sync.aligned.u32 {int(byte_count)} ;"')

    def setmaxregister_decrease(self, byte_count):
        _emitter().raw(
            f'nvvm.inline_ptx "setmaxnreg.dec.sync.aligned.u32 {int(byte_count)} ;"')

    def fence_proxy(self, kind, space=None):
        # cute.arch.fence_proxy("async.shared", space="cta")
        _emitter().fence_proxy_async_shared()

    def make_warp_uniform(self, v):
        # tid/32-derived values are already warp-uniform; the (wid,0,0)
        # thread-index tuple reduces to its linear warp id (both dense_gemm
        # flavors compare `warp_idx == 0` after this call).
        if isinstance(v, (tuple, list)) and v:
            return v[0]
        return v

    def block_idx_in_cluster(self):
        # single-CTA cluster profile: rank is always 0
        return _emitter().ssa("index", "arith.constant 0 : index")

    def cluster_arrive_relaxed(self):
        _emitter().raw('nvvm.inline_ptx "barrier.cluster.arrive.relaxed;"')

    def cluster_wait(self):
        _emitter().raw('nvvm.inline_ptx "barrier.cluster.wait;"')

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

    tc = getattr(part, "tc", None)
    if tc is not None and hasattr(tc, "tv"):  # TVCopyAsync partition
        T, R, V, B = part._geom
        elem = getattr(getattr(part.tensor, "element", None), "name", "f16")
        from .meta import F16
        return Fragment(V * B, F16 if elem in ("f16", "Float16") else F32)
    return Fragment(part.tc.vals_per_thread, F32)


def make_rmem_tensor(shape, dtype):
    from .kernel_objects import Fragment
    from .meta import BOOL

    def _flatten_shape(x):
        out = []
        for d in (x if isinstance(x, (tuple, list)) else (x,)):
            if isinstance(d, (tuple, list)):
                out.extend(_flatten_shape(d))
            elif hasattr(d, "shape"):
                out.extend(_flatten_shape(d.shape))
            else:
                out.append(int(d))
        return out

    dims = tuple(_flatten_shape(shape))
    n = 1
    for d in dims:
        n *= d
    fragment = Fragment(
        n, BOOL if getattr(dtype, "name", "") in ("bool", "Boolean") else dtype)
    fragment.dims = dims
    return fragment


def elem_less(coord, shape):
    e = _emitter()
    from .kernel_objects import CoordPair

    if hasattr(coord, "col"):  # CoordPair / _CoordUnit
        if isinstance(shape, (tuple, list)):
            m, n = int(shape[0]), int(shape[1])
            r_ok = e.cmpi_slt_const(coord.row, m)
            c_ok = e.cmpi_slt_const(coord.col, n)
            return e.ssa("i1", f"arith.andi {r_ok.name}, {c_ok.name} : i1")
        c_ok = e.cmpi_slt_const(coord.col, int(shape))
        return c_ok
    # single SSA/index coordinate vs scalar limit
    if isinstance(shape, (tuple, list)):
        shape = shape[1] if len(shape) > 1 else shape[0]
    return e.cmpi_slt_const(coord, int(shape))

    assert isinstance(coord, CoordPair)
    m, n = int(shape[0]), int(shape[1])
    r_ok = e.cmpi_slt_const(coord.row, m)
    c_ok = e.cmpi_slt_const(coord.col, n)
    return e.ssa("i1", f"arith.andi {r_ok.name}, {c_ok.name} : i1")


def copy(atom, src, dst, pred=None):
    """cute.copy(atom, src, dst, pred=frag) — per-value predicated ld/st."""
    from .kernel_objects import Fragment, ThrPartition

    e = _emitter()
    _t = type(atom).__name__
    if _t == "_SF_copy":                 # SF smem->rmem (blockscaled)
        copy_sf(atom, src, dst)
        return
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
            from .layout import CuteLayout as _CL
            if isinstance(v, _CL):
                rec(v.shape)
            else:
                out.append(int(v))

    rec(x)
    if len(out) < 2:
        out = out + [1]
    return out


# expose the thread index SSA recorded by the interpreter
def _get_tidx():
    return _active.tidx_ssa


# ------------------------------------------------------------------ smem + mma
class SmemArray:
    """cute.make_smem_array(name, count, element) — shared-memory buffer."""

    def __init__(self, ptr, count, elem):
        self.ptr = ptr          # SSA !llvm.ptr<3>
        self.count = count
        self.elem = elem        # ElementType

    def get_tensor(self, layout=None, swizzle=None):
        """Official storage.sX.get_tensor(outer, swizzle=inner): returns the
        SMEM tensor view — a Tensor whose base is this array, so tile
        algebra (group_modes/partition) carries the pointer along."""
        from .cute_objects import Tensor as HostTensor, Layout as HostLayout

        if layout is None:
            return self
        if hasattr(layout, "shape") and hasattr(layout, "stride") \
                and not isinstance(layout, HostLayout):
            from .cute_objects import Layout as _L

            layout = _L(layout.shape, layout.stride)
        if not isinstance(layout, HostLayout):
            from .cute_objects import make_layout as _ml

            layout = _ml(layout)
        view = HostTensor(self, layout, self.elem)
        view.swizzle = swizzle
        return view

    def data_ptr(self):
        return self.ptr


def make_smem_array(name: str, count: int, element=None):
    from .meta import F16

    e = _emitter()
    elem = element or F16
    # declaration must match the gep element used for addressing (interp
    # load/store gep by f16/f32), else f32 arrays get half-size allocations
    ety = {"f16": "f16", "float16": "f16", "f32": "f32", "float32": "f32",
           "i8": "i8", "uint8": "i8"}.get(elem.name.lower(), "f16")
    e.smem_global_declare(f"{name}", int(count), ety)
    ptr = e.smem_ptr(name)
    return SmemArray(ptr, int(count), elem)


def ldmatrix(smem, row_ssa, col_elems: int = 0, num: int = 4, trans: bool = False):
    """ldmatrix from smem at element offset (staged windows honored)."""
    e = _emitter()
    off = row_ssa
    soff = getattr(smem, "stage_offset", None)
    if soff is not None:
        off = e.idx_binop("arith.addi", off, soff)
    p = e.gep_smem(smem.ptr, off)
    return e.ldmatrix(p, num, trans)


def make_tiled_mma(atom, atom_layout=None):
    from .cute_objects import TiledMma, MmaF16BF16OpAtom

    # keep the real op (block-scaled ops carry shape_mnk like (16,8,64)
    # that the copy/gemm paths branch on); f16 call sites pass the plain
    # MmaF16BF16Op marker which has no shape — wrap it for compatibility
    op = atom if hasattr(atom, "shape_mnk") else MmaF16BF16OpAtom()
    return TiledMma(op, atom_layout)


def gemm(tiled_mma, acc, a_frag, b_frag):
    """cute.gemm(mma, acc, aFrags, bFrags) -> new acc (list of 4 SSA f32)."""
    e = _emitter()
    a = [e.bitcast_f16x2(x) for x in a_frag]
    b = [e.bitcast_f16x2(x) for x in b_frag]
    return e.mma_f16(a, b, acc)


def extract_frag(ld_res, idx: int):
    e = _emitter()
    return e.extract_i32(ld_res, idx)


# ------------------------------------------------------------------ TMA / pipeline
class TmaTensor:
    """A cute.Tensor backed by a TMA descriptor (kernel sees ptr to desc)."""

    def __init__(self, desc_ptr: SSA, recipe, smem_name: str):
        self.desc_ptr = desc_ptr      # SSA !llvm.ptr (generic)
        self.recipe = recipe          # runtime TensorMapRecipe
        self.smem_name = smem_name


def make_mbarrier(name: str, count: int):
    e = _emitter()
    bar = e.mbarrier_ptr(name)
    tid = e.thread_id("x")
    e.mbarrier_init_single_thread(bar, count, tid)
    e.fence_mbarrier_init()
    return bar


def tma_load(tma, smem, bar, coords):
    """cute.nvgpu.cpasync TMA G2S: smem tile (SmemArray/staged) + mbarrier."""
    e = _emitter()
    tma = getattr(tma, "desc_ssa", tma)   # kernel param glue
    smem_ptr = getattr(smem, "ptr", smem)
    off = getattr(smem, "stage_offset", None)
    if off is not None:
        elem = getattr(smem, "elem", None)
        ety = "f16" if getattr(elem, "name", "").lower() in ("f16", "float16") else "f32"
        smem_ptr = e.gep_smem(smem_ptr, off, ety)
    tma_ptr = getattr(tma, "desc_ptr", tma)
    e.tma_load(smem_ptr, tma_ptr, bar, coords)


def tma_store(tma, smem, coords):
    e = _emitter()
    smem_ptr = getattr(smem, "ptr", smem)
    tma_ptr = getattr(tma, "desc_ssa", None) or getattr(tma, "desc_ptr", tma)
    e.tma_store(tma_ptr, smem_ptr, coords)


def mbarrier_arrive_expect_tx(bar, tx_bytes: int):
    e = _emitter()
    e.mbarrier_arrive_expect_tx(bar, tx_bytes)


def mbarrier_try_wait_parity(bar, phase: int):
    e = _emitter()
    e.mbarrier_try_wait_parity(bar, phase)


def make_smem_tile(name: str, count: int, element=None):
    from .meta import F32

    e = _emitter()
    elem = element or F32
    mlir = {"f32": "f32", "f16": "f16",
            "float32": "f32", "float16": "f16"}.get(elem.name.lower(), "f32")
    ptr = e.smem_tile_declare(name, int(count), mlir)
    return SmemArray(ptr, int(count), elem)


def setmaxnreg(value: int, increase: bool = True):
    e = _emitter()
    e.setmaxregister(value, increase)


def named_barrier_arrive(id_: int, count: int):
    _emitter().named_barrier_arrive("nb", id_, count)


def named_barrier_sync(id_: int, count: int):
    _emitter().named_barrier_sync(id_, count)


# ------------------------------------------------------------------ TMA atom object model (M6)
def make_tiled_tma_atom(op, gmem_tensor, smem_layout_or_tensor, cta_layout=None):
    """Host-side (jit body) TMA atom construction, dense_gemm-style.

    gmem_tensor: TensorMeta (or CUtensorMapView.source meta) — its shape
    plus the smem tile layout define the box; the kernel-side descriptor
    pointer comes from the TmaTensor kernel param at launch.
    """
    from .cute_objects import TmaAtom, make_layout, _flatten
    from .meta import TensorMeta
    from ..runtime.tensor_map import TensorMapRecipe, CUtensorMapView

    _gw = getattr(getattr(gmem_tensor, "element_type", None), "width", 32) or 32
    if hasattr(smem_layout_or_tensor, "outer"):        # ComposedLayoutStaged
        smem_lay = smem_layout_or_tensor.outer
    elif hasattr(smem_layout_or_tensor, "layout"):
        smem_lay = smem_layout_or_tensor.layout
    else:
        smem_lay = smem_layout_or_tensor
    box_slow = _flatten(smem_lay.shape)[:2]
    if _gw < 8:
        box_slow = (box_slow[0], box_slow[1] // (8 // _gw))  # bytes
    box = tuple(reversed(box_slow))

    meta = gmem_tensor
    shp = tuple(int(d) for d in getattr(gmem_tensor, "shape", ()))
    _w = getattr(getattr(gmem_tensor, "element_type", None), "width", 32) or 32
    is_sf = (_w == 8 and len(shp) == 6 and shp[0] == 32 and shp[1] == 4
             and shp[3] == 4)   # flat ((32,4,rm),(4,rk),l)
    if isinstance(gmem_tensor, CUtensorMapView):
        meta = None
        recipe = gmem_tensor.recipe
    elif is_sf:
        # block-scaled SF in the canonical (32,4)x(4) blocked layout:
        # storage = contiguous 512B blocks; view shape (32,4,rm,4,rk,l)
        # with strides (16,4,rk*512,1,512,l*...).  The rank-4 recipe splits
        # each block into two 256B halves for the TMA box.
        import torch as _t
        strd = tuple(int(x) for x in getattr(gmem_tensor, "stride", ()))
        rm, rk, l = shp[2], shp[4], shp[5]
        # permuted view (32,4,rm,4,rk,l): rm is mode 2 (stride rk*512);
        # storage is the contiguous (l,rm,rk,32,4,4) blocked layout
        rm_str = (strd[2] if strd and strd[2] else None) or rk * 512
        # Split each 512-byte (128 rows x 4 SF) block into two contiguous
        # 256-byte halves.  Non-innermost TMA global strides must describe
        # the real storage (and satisfy CUDA alignment); using 4 here made
        # the dimensions overlap and becomes an invalid descriptor once the
        # logical FP4/SF shapes are represented correctly.
        recipe = TensorMapRecipe(dtype=_t.uint8,
                                 shape=(rm, rk, 2, 256),
                                 strides_elems=(rm_str, 512, 256, 1),
                                 box=(256, 2, 2, 1))
        box = (256, 2, 2, 1)
    elif isinstance(gmem_tensor, TensorMeta):
        import torch as _t
        _n = getattr(gmem_tensor.element_type, "name", "")
        dt = {"f16": _t.float16, "f32": _t.float32, "bf16": _t.bfloat16,
              "i8": _t.uint8, "Float4E2M1FN": _t.uint8,
              "Float8E4M3FN": _t.uint8}.get(_n, _t.float32)
        m, n = gmem_tensor.shape[0], gmem_tensor.shape[1]
        if _w <= 8:
            # fp4/fp8 operand storage: (m, k_bytes) row-major uint8
            row = int(gmem_tensor.stride[0]) if getattr(
                gmem_tensor, "stride", None) else n
            recipe = TensorMapRecipe(dtype=_t.uint8, shape=(m, row),
                                     strides_elems=(row, 1), box=box)
        else:
            recipe = TensorMapRecipe(dtype=dt, shape=(m, n),
                                     strides_elems=(n, 1), box=box)
    else:
        recipe = None
    atom = TmaAtom(op, None, cta_layout or make_layout((1, 1)), smem_lay, box)
    atom.recipe = recipe
    atom.tma_view = _materialize_tma_view(recipe, gmem_tensor)
    return atom, _tma_view(gmem_tensor)


def _materialize_tma_view(recipe, gmem_tensor):
    """Stage the encoded CUtensorMap in device memory at host-trace time
    (single-shot path; repeated launches reuse the same device copy)."""
    if recipe is None:
        return None
    import os as _os
    if _os.environ.get("DG_TMA_DEBUG"):
        import sys as _s
        print(f"DBG tma recipe: dtype={recipe.dtype} shape={recipe.shape} "
              f"strides={recipe.strides_elems} box={recipe.box} "
              f"meta_shape={tuple(getattr(gmem_tensor, 'shape', ()))} "
              f"w={getattr(getattr(gmem_tensor, 'element_type', None), 'width', '?')}",
              file=_s.stderr)
    from ..runtime.tensor_map import CUtensorMapView

    torch_t = getattr(gmem_tensor, "_torch", None)
    if torch_t is None or not getattr(torch_t, "is_cuda", False):
        return None
    return CUtensorMapView(recipe, torch_t)


def _tma_view(gmem_tensor):
    """Coordinate-only gmem view (official TmaTensor analogue): carries the
    global shape/stride for tile algebra; data flows via the TMA descriptor,
    so there is no device pointer behind it."""
    from .cute_objects import Tensor as HostTensor, Layout as HostLayout
    from .meta import TensorMeta

    if not isinstance(gmem_tensor, TensorMeta):
        return None
    shp, str_ = list(gmem_tensor.shape), list(gmem_tensor.stride)
    if len(shp) == 2:                       # (m,k) with implicit L=1 -> (m,k,l)
        shp.append(1)
        str_.append(shp[0] * str_[0])       # L stride = m*k elements
    view = HostTensor(None, HostLayout(tuple(shp), tuple(str_)),
                      gmem_tensor.element_type)
    view.is_tma_view = True
    return view


# ------------------------------------------------ smem->rmem copy lowering
def _ix(e, v):
    """const or SSA -> index-typed SSA"""
    from .emitter import SSA as _SSA

    if isinstance(v, _SSA):
        return v if v.type == "index" else e.ssa(
            "index", f"arith.index_cast {v.name} : i32 to index")
    return e.ssa("index", f"arith.constant {int(v)} : index")


def _ixop(e, op, a, b):
    return e.idx_binop(op, a, b)


def _lane0_predicate():
    """i1 SSA: this thread is lane 0 of its warp (TMA/expect_tx leader)."""
    e = _emitter()
    t = e.thread_id("x")
    w = e.ssa("index", "arith.constant 32 : index")
    lane = e.idx_binop("arith.remsi", t, w)
    z = e.ssa("index", "arith.constant 0 : index")
    return e.ssa("i1", f"arith.cmpi eq, {lane.name}, {z.name} : index")


def copy_tiled(tc, src, dst):
    """cute.copy(smem_tiled_copy_A/B, tCsA_p[...k], tCrA[...k]): emit the
    per-thread ldmatrix sequence from the mma trait geometry."""
    from .cute_objects import SmemCopySrc, FragK, _flatten

    e = _emitter()
    assert isinstance(src, SmemCopySrc) and isinstance(dst, FragK), \
        "tiled copy needs (SmemCopySrc, FragK)"
    tile = src.tile
    arr = tile.base                    # SmemArray
    lay = tile.layout
    flat_shp = [int(d) for d in _flatten(lay.shape)]
    flat_str = [int(d) for d in _flatten(lay.stride)]
    elems_per_stage = flat_shp[0] * flat_shp[1]
    tidx = _ix(e, tc.tidx)

    _ak = getattr(getattr(tc.mma, "op", None), "shape_mnk", (16, 8, 16))[2]
    if _ak >= 64:
        # fp4 arrays are byte-typed: halve the element count for windows
        elems_per_stage = elems_per_stage // 2
    win = smem_stage(arr, src.stage, elems_per_stage)
    soff = win.stage_offset

    mma = tc.mma
    am, an, ak = getattr(mma.op, "shape_mnk", (16, 8, 16))
    warps_m = int(_flatten(mma.atom_layout)[0]) if mma.atom_layout else 1
    warp_m, warp_n = mma.tile_mn
    if ak >= 64:                        # SM120 block-scaled warp grid
        warp_m, warp_n = 32, 64
        warps_m = 4
    mm = warp_m // am
    mn = warp_n // an
    row_str = _ix(e, flat_str[0])      # stride of the non-contiguous dim

    lane = _ixop(e, "arith.remsi", tidx, _ix(e, 32))
    wid = _ixop(e, "arith.divsi", tidx, _ix(e, 32))
    wm = _ixop(e, "arith.remsi", wid, _ix(e, warps_m))
    wn = _ixop(e, "arith.divsi", wid, _ix(e, warps_m))
    k0 = _ixop(e, "arith.muli", _ix(e, src.k), _ix(e, 16))

    if tc.which == "A" and ak >= 64:
        # fp4 packed smem (m, k/2 bytes): per trait ALayout, reg j of atom i:
        # row = base + lane//4 + 8*(j%2); k-octet = 8*(lane%4) + 32*(j//2)
        # -> aligned b32 loads (8 nibbles = 4 bytes each)
        regs = []
        l4 = _ixop(e, "arith.divsi", lane, _ix(e, 4))
        l4m = _ixop(e, "arith.remsi", lane, _ix(e, 4))
        byte_row = _ix(e, flat_str[0] // 2)   # bytes per m-row (fp4)
        kb4 = _ixop(e, "arith.muli", _ix(e, src.k), _ix(e, 32))
        for i in range(mm):
            r0 = _ixop(e, "arith.addi",
                       _ixop(e, "arith.muli", wm, _ix(e, warp_m)),
                       _ix(e, i * 16 + 0))
            r0 = _ixop(e, "arith.addi", r0, l4)
            koct = _ixop(e, "arith.addi",
                         _ixop(e, "arith.muli", l4m, _ix(e, 4)), kb4)
            for j in range(4):
                rj = _ixop(e, "arith.addi", r0,
                           _ix(e, 8 * (j % 2)))
                kj = _ixop(e, "arith.addi", koct, _ix(e, 16 * (j // 2)))
                off = _ixop(e, "arith.addi", soff,
                            _ixop(e, "arith.addi",
                                  _ixop(e, "arith.muli", rj, byte_row), kj))
                p = e.gep_smem(arr.ptr, off, "i8")
                regs.append(e.ssa("i32",
                                  f"llvm.load {p.name} : !llvm.ptr<3> -> i32"))
        dst.frag.slots[dst.k] = regs
    elif tc.which == "A":
        # smem layout (m,k):(k,1): ldmatrix.x4 rows=m (16B = 8 halves of k)
        regs = []
        for i in range(mm):
            m_base = _ixop(e, "arith.addi",
                           _ixop(e, "arith.muli", wm, _ix(e, warp_m)),
                           _ix(e, i * 16))
            m_row = _ixop(e, "arith.addi", m_base,
                          _ixop(e, "arith.remsi", lane, _ix(e, 16)))
            half8 = _ixop(e, "arith.muli",
                          _ixop(e, "arith.divsi", lane, _ix(e, 16)), _ix(e, 8))
            off = _ixop(e, "arith.addi", soff,
                        _ixop(e, "arith.addi",
                              _ixop(e, "arith.muli", m_row, row_str),
                              _ixop(e, "arith.addi", k0, half8)))
            f = e.ldmatrix(e.gep_smem(arr.ptr, off), 4, trans=False)
            regs.extend(e.extract_i32(f, j) for j in range(4))
        dst.frag.slots[dst.k] = regs
    elif ak >= 64:
        # fp4 packed B (n, k/2 bytes): reg j of atom na: col = base +
        # lane//4; k-octet = 8*(lane%4) + 32*j -> aligned b32 loads
        regs = []
        l4 = _ixop(e, "arith.divsi", lane, _ix(e, 4))
        l4m = _ixop(e, "arith.remsi", lane, _ix(e, 4))
        byte_row = _ix(e, flat_str[0] // 2)
        kb4 = _ixop(e, "arith.muli", _ix(e, src.k), _ix(e, 32))
        for na in range(mn):
            c0 = _ixop(e, "arith.addi",
                       _ixop(e, "arith.muli", wn, _ix(e, warp_n)),
                       _ix(e, na * 8))
            c0 = _ixop(e, "arith.addi", c0, l4)
            koct = _ixop(e, "arith.addi",
                         _ixop(e, "arith.muli", l4m, _ix(e, 4)), kb4)
            for j in range(2):
                kj = _ixop(e, "arith.addi", koct, _ix(e, 16 * j))
                off = _ixop(e, "arith.addi", soff,
                            _ixop(e, "arith.addi",
                                  _ixop(e, "arith.muli", c0, byte_row), kj))
                p = e.gep_smem(arr.ptr, off, "i8")
                regs.append(e.ssa("i32",
                                  f"llvm.load {p.name} : !llvm.ptr<3> -> i32"))
        dst.frag.slots[dst.k] = regs
    else:
        # smem layout (n,k):(k,1): ldmatrix.x2 rows=n; mat1 = +8 k elements
        regs = []
        for na in range(mn):
            n_base = _ixop(e, "arith.addi",
                           _ixop(e, "arith.muli", wn, _ix(e, warp_n)),
                           _ix(e, na * 8))
            n_row = _ixop(e, "arith.addi", n_base,
                          _ixop(e, "arith.remsi", lane, _ix(e, 8)))
            kk8 = _ixop(e, "arith.muli",
                        _ixop(e, "arith.divsi", lane, _ix(e, 8)), _ix(e, 8))
            off = _ixop(e, "arith.addi", soff,
                        _ixop(e, "arith.addi",
                              _ixop(e, "arith.muli", n_row, row_str),
                              _ixop(e, "arith.addi", k0, kk8)))
            f = e.ldmatrix(e.gep_smem(arr.ptr, off), 2, trans=False)
            regs.extend(e.extract_i32(f, j) for j in range(2))
        dst.frag.slots[dst.k] = regs


def gemm_tv(mma, acc, a_k, b_k):
    """cute.gemm(tiled_mma, acc_fragment, tCrA[k], tCrB[k], acc_fragment):
    per (m_atom, n_atom) mma.sync; acc slots updated in place."""
    from .cute_objects import FragK

    e = _emitter()
    assert isinstance(a_k, FragK) and isinstance(b_k, FragK)
    aregs = a_k.frag.slots[a_k.k]
    bregs = b_k.frag.slots[b_k.k]
    am, an, _ = getattr(mma.op, "shape_mnk", (16, 8, 16))
    warp_m, warp_n = mma.tile_mn
    mm, mn = warp_m // am, warp_n // an
    for i in range(mm):
        a4 = [e.bitcast_f16x2(aregs[i * 4 + j]) for j in range(4)]
        for na in range(mn):
            b2 = [e.bitcast_f16x2(bregs[na * 2 + j]) for j in range(2)]
            c0 = (i * mn + na) * 4
            c4 = [acc.slots.get(c0 + j) for j in range(4)]
            out = e.mma_f16(a4, b2, c4)
            for j in range(4):
                acc.slots[c0 + j] = out[j]


def copy_r2s(tc, src, dst):
    """cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(..., buf)]): store the
    thread's converted accumulator regs to sC at the trait C coordinates
    (+ epi stage window)."""
    from .cute_objects import R2SSmemView, AccumRetile, _flatten

    e = _emitter()
    frag = src
    if isinstance(src, AccumRetile):
        frag = src.frag
    assert isinstance(dst, R2SSmemView)
    tile = dst.tile
    arr = tile.base                    # SmemArray
    lay = tile.layout
    flat_shp = [int(d) for d in _flatten(lay.shape)]
    flat_str = [int(d) for d in _flatten(lay.stride)]
    elems_per_stage = flat_shp[0] * flat_shp[1]
    win = smem_stage(arr, dst.stage, elems_per_stage)
    soff = win.stage_offset

    mma = tc.mma
    am, an, ak = getattr(mma.op, "shape_mnk", (16, 8, 16))
    warps_m = int(_flatten(mma.atom_layout)[0]) if mma.atom_layout else 1
    warp_m, warp_n = mma.tile_mn
    if ak >= 64:
        warp_m, warp_n = 32, 64
        warps_m = 4
    mm, mn = warp_m // am, warp_n // an
    row_str = _ix(e, flat_str[0])      # (m,n):(n,1) -> row stride = n

    tidx = _ix(e, tc.tidx)
    lane = _ixop(e, "arith.remsi", tidx, _ix(e, 32))
    wid = _ixop(e, "arith.divsi", tidx, _ix(e, 32))
    wm = _ixop(e, "arith.remsi", wid, _ix(e, warps_m))
    wn = _ixop(e, "arith.divsi", wid, _ix(e, warps_m))
    group = _ixop(e, "arith.divsi", lane, _ix(e, 4))    # lane//4
    t2 = _ixop(e, "arith.muli",
               _ixop(e, "arith.remsi", lane, _ix(e, 4)), _ix(e, 2))

    # slot order matches gemm_tv: (i*MMA_N + na)*4 + j
    ty = "f16" if getattr(frag, "dtype", None) is not None and \
        frag.dtype.name in ("f16", "Float16") else \
        ("f16" if flat_str and flat_shp[1] * 2 <= 128 else "f32")
    # element type from the fragment's stored values
    sample = next(iter(frag.slots.values()), None)
    vty = getattr(sample, "type", "f16")
    ety = "f16" if "f16" in vty or frag.dtype.name in ("f16", "Float16") \
        else "f32"
    for i in range(mm):
        row_base = _ixop(e, "arith.addi",
                         _ixop(e, "arith.muli", wm, _ix(e, warp_m)),
                         _ix(e, i * 16))
        for na in range(mn):
            col_base = _ixop(e, "arith.addi",
                             _ixop(e, "arith.muli", wn, _ix(e, warp_n)),
                             _ix(e, na * 8))
            for j in range(4):
                c0 = (i * mn + na) * 4 + j
                v = frag.slots.get(c0)
                if v is None:
                    continue
                # j: 0 (r,c) 1 (r,c+1) 2 (r+8,c) 3 (r+8,c+1) — trait C coords
                r8 = _ix(e, 8 if j >= 2 else 0)
                c1 = _ix(e, 1 if j % 2 == 1 else 0)
                row = _ixop(e, "arith.addi", _ixop(e, "arith.addi",
                                                   row_base, group), r8)
                col = _ixop(e, "arith.addi", col_base,
                            _ixop(e, "arith.addi", t2, c1))
                off = _ixop(e, "arith.addi", soff,
                            _ixop(e, "arith.addi",
                                  _ixop(e, "arith.muli", row, row_str), col))
                p = e.gep_smem(arr.ptr, off, ety)
                e.raw(f"llvm.store {v.name}, {p.name} : {vty}, !llvm.ptr<3>")


def frag_to(view, dtype):
    """FragmentView.to(dtype): elementwise truncf f32->f16."""
    e = _emitter()
    out = []
    for v in view.values:
        if dtype.name in ("f16", "Float16") and v.type == "f32":
            out.append(e.ssa("f16", f"arith.truncf {v.name} : f32 to f16"))
        elif dtype.name in ("f32", "Float32") and v.type == "f16":
            out.append(e.ssa("f32", f"llvm.fpext {v.name} : f16 to f32"))
        else:
            out.append(v)
    from .kernel_objects import FragmentView

    return FragmentView(out)


def copy_sf(tc, src, dst):
    """SF smem->rmem per the SM120 mxf4nvf4 trait layouts (cute BSD):
    one b32 per atom; SFA bytes = SF[row = atom_base + lane//4 +
    8*(lane%2)][kg 0..3]; SFB bytes = SF[col = atom_base + lane//4]
    [kg 0..3] (k-group base = kb*4, nvfp4 sf_vec 16)."""
    from .cute_objects import _flatten

    e = _emitter()
    view = src if hasattr(src, "tile") else dst
    tile = view.tile
    arr = tile.base
    lay = tile.layout
    rows = int(_flatten(lay.shape)[0])
    kgs = int(_flatten(lay.shape)[1])
    kb = int(getattr(view, "k", 0) or 0)

    tidx = _ix(e, tc.tidx)
    lane = _ixop(e, "arith.remsi", tidx, _ix(e, 32))
    wid = _ixop(e, "arith.divsi", tidx, _ix(e, 32))
    wm = _ixop(e, "arith.remsi", wid, _ix(e, 4))
    wn = _ixop(e, "arith.divsi", wid, _ix(e, 4))

    global _SF_SEQ, _SF_KB
    if int(kb) != _SF_KB:
        _SF_KB = int(kb)
        _SF_SEQ = 0
    is_a = _SF_SEQ % 2 == 0
    which = "A" if is_a else "B"
    l4 = _ixop(e, "arith.divsi", lane, _ix(e, 4))      # lane//4
    l2 = _ixop(e, "arith.remsi", lane, _ix(e, 2))      # lane%2 (row select)
    # stage window: rows*kgs bytes per stage (SF arrays are byte-typed)
    win = smem_stage(arr, getattr(view, "stage", 0), rows * kgs)
    soff = win.stage_offset
    regs = []
    # Blocked SMEM order from the SF TMA box.  Each MMA K64 block owns
    # four scale factors per row (sf_vec=16), and all 128 rows precede
    # the next K64 block:
    #   byte = kb * (rows * 4) + (row%32)*16 + (row//32)*4 + kg
    # The old ``kb * 4`` selected the next M32 quadrant instead of the
    # next K64 block, so kg0..3 were applied twice and kg4..7 were never
    # consumed for a 128-wide K tile.
    def _sf_off(row):
        i32p = _ixop(e, "arith.remsi", row, _ix(e, 32))
        i4p = _ixop(e, "arith.divsi", row, _ix(e, 32))
        return _ixop(e, "arith.addi",
                     _ixop(e, "arith.addi",
                           _ixop(e, "arith.muli", i32p, _ix(e, 16)),
                           _ixop(e, "arith.muli", i4p, _ix(e, 4))),
                     _ix(e, int(kb) * rows * 4))

    if is_a:
        for i in range(2):                             # m-atoms (warp 32)
            r = _ixop(e, "arith.addi",
                      _ixop(e, "arith.muli", wm, _ix(e, 32)),
                      _ix(e, i * 16))
            r = _ixop(e, "arith.addi", r, l4)
            r = _ixop(e, "arith.addi", r, _ixop(e, "arith.muli", l2, _ix(e, 8)))
            regs.append(_load_b32(e, arr, _ixop(e, "arith.addi",
                                                soff, _sf_off(r))))
    else:
        for na in range(8):                            # n-atoms (warp 64)
            c = _ixop(e, "arith.addi",
                      _ixop(e, "arith.muli", wn, _ix(e, 64)),
                      _ix(e, na * 8))
            c = _ixop(e, "arith.addi", c, l4)
            regs.append(_load_b32(e, arr, _ixop(e, "arith.addi",
                                                soff, _sf_off(c))))
    _SF_SEQ += 1
    _SF_SLOTS.setdefault(which, {})[int(kb)] = regs


def _bswap(e, v):
    """32-bit byte reversal via inline PTX."""
    return e.ssa(
        "i32",
        'nvvm.inline_ptx "prmt.b32 $0, $1, $1, 0x1032;" ro (' +
        f"{v.name} : i32) -> i32")


def _load_b32(e, arr, off):
    """Aligned 4-byte SF load (4 packed e4m3) from the smem byte array."""
    from .emitter import SSA as _SSA

    if not isinstance(off, _SSA):
        off = e.ssa("index", f"arith.constant {int(off)} : index")
    p = e.gep_smem(arr.ptr, off, "i8")
    return e.ssa("i32", f"llvm.load {p.name} : !llvm.ptr<3> -> i32")


_SF_SLOTS: dict = {}
_SF_SEQ = 0
_SF_KB = -1


def _load_u8(e, arr, off):
    p = e.gep_smem(arr.ptr, off, "i8")
    return e.ssa("i8", f"llvm.load {p.name} : !llvm.ptr<3> -> i8")


def _pack_bytes(e, lo, hi):
    lo32 = e.ssa("i32", f"llvm.zext {lo.name} : i8 to i32")
    hi32 = e.ssa("i32", f"llvm.zext {hi.name} : i8 to i32")
    shift = e.ssa("i32", "arith.constant 8 : i32")
    h = e.ssa("i32", f"llvm.shl {hi32.name}, {shift.name} : i32")
    return e.ssa("i32", f"llvm.or {lo32.name}, {h.name} : i32")


def gemm_bs(mma, acc, a_list, b_list):
    """cute.gemm(mma, acc, [tCrA[k], tCrSFA[k]], [tCrB[k], tCrSFB[k]], acc)
    — block-scaled: kind::mxf4nvf4 per (m_atom, n_atom) over the warp grid.
    First-guess register mapping (calibrate vs PTX): A regs from the fp4
    ldmatrix path, SF bytes from copy_sf slots."""
    e = _emitter()
    a_k, sfa_v = a_list
    b_k, sfb_v = b_list
    aregs = a_k.frag.slots[a_k.k]
    bregs = b_k.frag.slots[b_k.k]
    kb = int(getattr(sfa_v, "k", a_k.k) or 0)
    sfa = _SF_SLOTS.get("A", {}).get(kb) or []
    sfb = _SF_SLOTS.get("B", {}).get(kb) or []
    import os as _os
    if _os.environ.get("DG_GEMM_DEBUG"):
        import sys as _s
        print(f"DBG gemm_bs: kb={kb} sfA={len(sfa)} sfB={len(sfb)} "
              f"aregs={len(aregs)} bregs={len(bregs)} "
              f"slotsA={sorted(_SF_SLOTS.get('A', {}).keys())}",
              file=_s.stderr)
    mma_op = getattr(mma, "op", None)
    atom = getattr(mma_op, "shape_mnk", (16, 8, 64))
    # SM120 blockscaled warp grid: atom_layout (4,2,1) over tile (128,128)
    # -> warp tile (32, 64); keep in lockstep with copy_sf
    warp_m, warp_n = 32, 64
    mm, mn = warp_m // atom[0], warp_n // atom[1]
    zero32 = None
    for i in range(mm):
        for na in range(mn):
            c0 = (i * mn + na) * 4
            c4 = [acc.slots.get(c0 + j) for j in range(4)]
            if zero32 is None:
                zero32 = e.ssa("i32", "arith.constant 0 : i32")
            a4 = [aregs[i * 4 + j] if i * 4 + j < len(aregs) else zero32
                  for j in range(4)]
            b2 = [bregs[na * 2 + j] if na * 2 + j < len(bregs) else zero32
                  for j in range(2)]
            sa1 = [sfa[i] if i < len(sfa) else zero32]
            sb1 = [sfb[na] if na < len(sfb) else zero32]
            r = e.mma_mxf4nvf4(a4, b2, sa1, sb1, c4)
            for j in range(4):
                acc.slots[c0 + j] = r[j]


def tma_partition(atom, cta_coord, cta_layout, smem_grouped, gmem_grouped):
    """Project smem/gmem grouped views onto this CTA's tile coordinates.
    Produces (TmaSmemView, TmaGmemView): dense_gemm subscripts record the
    stage / gmem tile offsets; cute.copy lowers to the bulk-tensor PTX."""
    from .cute_objects import (TmaPartitioned, TmaSmemView, TmaGmemView,
                               Tensor as HT, _flatten)

    arr = smem_grouped.base if isinstance(smem_grouped, HT) else smem_grouped
    lay = smem_grouped.layout if isinstance(smem_grouped, HT) else None
    n = 1
    if lay is not None:
        for d in _flatten(lay.shape):
            n *= int(d)
    else:
        n = int(getattr(arr, "count", 0))
    # sub-byte dtypes: windows address BYTES in the i8-typed smem array
    w = getattr(getattr(arr, "elem", None), "width", 32) or 32
    if w < 8:
        n = n // (8 // w) if w < 8 and 8 % w == 0 else n * w // 8
    elif w == 8:
        pass                             # 1 byte per element already
    sview = TmaSmemView(arr, n)
    box = tuple(int(b) for b in atom.box)       # fastest-first
    gview = TmaGmemView(atom, ("d0", "d1", "d2"), (box[1], box[0], 1))
    # gmem tile coords recorded by view subscripts (flat modes: the loop
    # coords sit at positions 2 (slow dim) and 3 (fast dim) by construction)
    co = dict(getattr(gmem_grouped, "coord_offs", {}))
    if 2 in co:
        gview.offs["d0"] = co[2]
    if 3 in co:
        gview.offs["d1"] = co[3]
    return TmaPartitioned(sview, gview, cta_coord, box)


def tma_copy_partitioned(atom, src, dst, tma_bar_ptr=None, mcast_mask=None):
    """cute.copy(tma_atom, gmem_view, smem_view[, tma_bar_ptr][, mcast_mask])
    -> cp.async.bulk.tensor (G2S or S2G) with coords from the view offsets."""
    from .cute_objects import TmaGmemView
    from .emitter import SSA as _SSA

    e = _emitter()
    if isinstance(src, TmaGmemView):
        gv, smem_win = src, dst            # G2S load
    else:
        gv, smem_win = dst, src            # S2G store
        e.fence_proxy_async_shared()
    box = tuple(int(b) for b in atom.box)  # fastest-first
    offs = gv.offs

    def _elem(dim, tile):
        v = offs.get(dim, 0)
        if not isinstance(v, _SSA):
            return e.ssa("i32", f"arith.constant {int(v) * int(tile)} : i32")
        v32 = v if v.type == "i32" else e.ssa(
            "i32", f"arith.index_cast {v.name} : {v.type} to i32")
        c = e.ssa("i32", f"arith.constant {int(tile)} : i32")
        return e.ssa("i32", f"arith.muli {v32.name}, {c.name} : i32")

    if len(box) == 4:
        # SF rank-4 recipe: (rm, rk, 2, 256), box (256,2,2,1) — coords
        # fastest-first: [inner 0, half 0, rk0 = kt*2, rm]
        coords = [_elem("d1", 2), _elem("d0", 1)]
        coords = [e.ssa("i32", "arith.constant 0 : i32"),
                  e.ssa("i32", "arith.constant 0 : i32")] + coords
    else:
        coords = [_elem("d1", box[0]), _elem("d0", box[1])]
    desc = getattr(atom, "desc_ssa", None)
    if desc is None:
        raise ValueError("tma copy needs atom.desc_ssa (kernel param glue)")
    from ..object_model import tma as _otma

    # TMA bulk ops are leader-issued (lane 0): every thread of the issuing
    # warp reaches cute.copy, but one issue arms the barrier transaction
    e0 = _emitter()
    pred = _lane0_predicate()
    e0.open_if(pred)
    if isinstance(src, TmaGmemView):
        if tma_bar_ptr is None:
            raise ValueError("G2S tma copy needs tma_bar_ptr")
        _otma.tma_issue(smem_win, desc, tma_bar_ptr, coords)
    else:
        _otma.tma_store_issue(desc, smem_win, coords)
    e0.close_if()


def copy(atom_or_view, *rest, mbarrier=None, pred=None):
    """cute.copy dispatch: TMA partitioned pair -> cp.async.bulk.tensor."""
    from .cute_objects import TmaPartitioned, TmaAtom

    e = _emitter()
    if isinstance(atom_or_view, TmaPartitioned):
        p = atom_or_view
        if mbarrier is None:
            raise ValueError("TMA partitioned copy needs mbarrier=")
        coords = []
        c = p.cta_coord_ssa
        coords.append(c if not isinstance(c, int) else e.ssa("i32", f"arith.constant {int(c)} : i32"))
        coords.append(e.ssa("i32", "arith.constant 0 : i32"))
        smem = getattr(p.smem_view, "ptr", None) or getattr(p.smem_view, "base", None)
        desc = getattr(p, "desc_ptr", None)
        if desc is None:
            raise ValueError("partitioned copy needs desc_ptr (set by kernel glue)")
        e.tma_load(smem, desc, mbarrier, coords)
        return
    raise NotImplementedError(f"copy on {type(atom_or_view)}")


def const_f32(v):
    e = _emitter()
    return e.ssa("f32", f"arith.constant {float(v) :.10e} : f32".replace("+", ""))


def exp_f32(v):
    """cute.exp — f32 exp via ex2.approx PTX: exp(x) = ex2(x*log2e)."""
    e = _emitter()
    l2e = e.ssa("f32", "arith.constant 1.4426950408889634 : f32")
    x = e.ssa("f32", f"arith.mulf {v.name}, {l2e.name} : f32")
    return e.ssa(
        "f32",
        'nvvm.inline_ptx "ex2.approx.f32 $0, $1;" ro (' +
        f"{x.name} : f32) -> f32")


def fptrunc_f16(v):
    """f32 -> f16 value conversion (for p/v stores)."""
    e = _emitter()
    return e.ssa("f16", f"llvm.fptrunc {v.name} : f32 to f16")


def fmax_f32(a, b):
    e = _emitter()
    c = e.ssa("i1", f"arith.cmpf ogt, {a.name}, {b.name} : f32")
    return e.ssa("f32", f"arith.select {c.name}, {a.name}, {b.name} : f32")


def min_i32(a, b):
    e = _emitter()
    c = e.ssa("i1", f"arith.cmpi slt, {a.name}, {b.name} : i32")
    return e.ssa("i32", f"arith.select {c.name}, {a.name}, {b.name} : i32")


def smem_stage(smem_arr, stage, elems_per_stage):
    """Window a staged smem array at (stage * elems_per_stage). stage may
    be an SSA index — returns a SmemArray view with SSA offset base."""
    e = _emitter()
    from .emitter import SSA as _SSA
    if isinstance(stage, int):
        off = e.ssa("index", f"arith.constant {stage * int(elems_per_stage)} : index")
    else:
        c = e.ssa("index", f"arith.constant {int(elems_per_stage)} : index")
        prod = e.idx_binop("arith.muli", stage, c) if stage.type == "index" else \
            e.idx_binop("arith.muli", e.ssa("index", f"arith.index_cast {stage.name} : i32 to index"), c)
        off = prod
    v = SmemArray(smem_arr.ptr, int(elems_per_stage), smem_arr.elem)
    v.stage_offset = off
    return v


_mma_cache = {}


def gemm_reg(c_reg, a_frag, b_frag, i):
    """Single-register view of gemm: returns D[i] given C[i]. Emits ONE
    mma.sync per unique (a,b,c0..c3) tuple, caching the 4 outputs."""
    e = _emitter()
    key = (tuple(x.name for x in a_frag), tuple(x.name for x in b_frag),
           _mma_group_key(c_reg, e))
    if key in _mma_cache:
        return _mma_cache[key][int(i)]
    raise RuntimeError("gemm_reg requires group setup")


def _mma_group_key(c_reg, e):
    return c_reg.name


def bool_to_i32(b):
    """i1 SSA -> i32 SSA (0/1) — zero-extend: sign-extending true gives -1."""
    e = _emitter()
    if isinstance(b, bool):
        return e.ssa("i32", f"arith.constant {1 if b else 0} : i32")
    assert getattr(b, "type", "") == "i1"
    return e.ssa("i32", f"arith.extui {b.name} : i1 to i32")


def add_i32(a, b):
    e = _emitter()
    if isinstance(a, int):
        a = e.ssa("i32", f"arith.constant {a} : i32")
    if isinstance(b, int):
        b = e.ssa("i32", f"arith.constant {b} : i32")
    return e.ssa("i32", f"arith.addi {a.name}, {b.name} : i32")


def rem_i32(a, b):
    e = _emitter()
    if isinstance(b, int):
        b = e.ssa("i32", f"arith.constant {b} : i32")
    return e.ssa("i32", f"arith.remsi {a.name}, {b.name} : i32")


def lt_i32(a, b):
    e = _emitter()
    if isinstance(b, int):
        b = e.ssa("i32", f"arith.constant {b} : i32")
    return e.ssa("i1", f"arith.cmpi slt, {a.name}, {b.name} : i32")


def const_i32(v):
    return _emitter().ssa("i32", f"arith.constant {int(v)} : i32")


def idx_to_i32(v):
    e = _emitter()
    if isinstance(v, int):
        return e.ssa("i32", f"arith.constant {v} : i32")
    if v.type == "i32":
        return v
    return e.ssa("i32", f"arith.index_cast {v.name} : {v.type} to i32")


def i32_to_index(v):
    e = _emitter()
    if isinstance(v, int):
        return e.ssa("index", f"arith.constant {v} : index")
    if v.type == "index":
        return v
    return e.ssa("index", f"arith.index_cast {v.name} : {v.type} to index")


def mul_i32(a, b):
    e = _emitter()
    return e.ssa("i32", f"arith.muli {a.name}, {b.name} : i32")


def sub_i32(a, b):
    e = _emitter()
    if isinstance(b, int):
        b = e.ssa("i32", f"arith.constant {int(b)} : i32")
    return e.ssa("i32", f"arith.subi {a.name}, {b.name} : i32")


def mbarrier_reinit(bar, count: int):
    e = _emitter()
    e.mbarrier_init(bar, count)
    e.fence_mbarrier_init()


def fence_and_sync():
    e = _emitter()
    e.fence_mbarrier_init()
    e.barrier()


def mbarrier_inval_and_init(bar, count: int):
    e = _emitter()
    e.mbarrier_init(bar, count)
    e.fence_mbarrier_init()


def div_i32(a, b):
    e = _emitter()
    if isinstance(b, int):
        b = e.ssa("i32", f"arith.constant {int(b)} : i32")
    return e.ssa("i32", f"arith.divsi {a.name}, {b.name} : i32")


def fence_proxy():
    """cute.arch.fence_proxy — async-proxy fence (shared::cta)."""
    _emitter().fence_proxy_async_shared()


def make_barrier_array(name: str, count: int):
    """Int64 smem array for mbarrier storage (driver objects)."""
    e = _emitter()
    e.smem_globals.append(
        f"    llvm.mlir.global internal @{name}() "
        f"{{addr_space = 3 : i32, alignment = 8 : i64}} "
        f": !llvm.array<{int(count)} x i64>")

    class _B:
        pass
    b = _B()
    b.ptr = e.ssa("!llvm.ptr<3>",
                  f"llvm.mlir.addressof @{name} : !llvm.ptr<3>")
    return b


def make_rmem_tensor(shape, dtype):
    """cute.make_rmem_tensor(shape, dtype) — register fragment tensor."""
    from .kernel_objects import Fragment
    from .meta import BOOL, F32, F16

    def _flat_sh(x):
        out = []
        for d in (x if isinstance(x, (tuple, list)) else (x,)):
            if isinstance(d, (tuple, list)):
                out.extend(_flat_sh(d))
            elif hasattr(d, "shape"):
                out.extend(_flat_sh(d.shape))
            else:
                out.append(int(d))
        return out

    dims = tuple(_flat_sh(shape))
    n = 1
    for d in dims:
        n *= d
    elem = {"Boolean": BOOL, "Float32": F32, "Float16": F16}.get(
        getattr(dtype, "name", ""), F32)
    fragment = Fragment(n, elem)
    fragment.dims = dims
    return fragment

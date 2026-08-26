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


# ------------------------------------------------------------------ smem + mma
class SmemArray:
    """cute.make_smem_array(name, count, element) — shared-memory buffer."""

    def __init__(self, ptr, count, elem):
        self.ptr = ptr          # SSA !llvm.ptr<3>
        self.count = count
        self.elem = elem        # ElementType


def make_smem_array(name: str, count: int, element=None):
    from .meta import F16

    e = _emitter()
    elem = element or F16
    e.smem_global_declare(f"{name}", int(count), "f16")
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

    return TiledMma(MmaF16BF16OpAtom(), atom_layout)


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
    from .kernel_objects import Fragment  # noqa: F401
    e = _emitter()
    bar = e.mbarrier_ptr(name)
    e.mbarrier_init(bar, count)
    e.fence_mbarrier_init()
    return bar


def tma_load(tma, smem, bar, coords):
    """cute.nvgpu.cpasync TMA G2S: smem tile (SmemArray/staged) + mbarrier."""
    e = _emitter()
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
    tma_ptr = getattr(tma, "desc_ptr", tma)
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

    if hasattr(smem_layout_or_tensor, "layout"):
        smem_lay = smem_layout_or_tensor.layout
    else:
        smem_lay = smem_layout_or_tensor
    box_slow = _flatten(smem_lay.shape)[:2]
    box = tuple(reversed(box_slow))

    meta = gmem_tensor
    if isinstance(gmem_tensor, CUtensorMapView):
        meta = None
        recipe = gmem_tensor.recipe
    elif isinstance(gmem_tensor, TensorMeta):
        import torch as _t
        dt = {"f16": _t.float16, "f32": _t.float32,
              "bf16": _t.bfloat16}.get(gmem_tensor.element_type.name, _t.float32)
        m, n = gmem_tensor.shape[0], gmem_tensor.shape[1]
        recipe = TensorMapRecipe(dtype=dt, shape=(m, n),
                                 strides_elems=(n, 1), box=box)
    else:
        recipe = None
    atom = TmaAtom(op, None, cta_layout or make_layout((1, 1)), smem_lay, box)
    atom.recipe = recipe
    return atom, None  # (atom, tma_tensor placeholder — host view only)


def tma_partition(atom, cta_coord, cta_layout, smem_grouped, gmem_grouped):
    from .cute_objects import TmaPartitioned

    return TmaPartitioned(smem_grouped, gmem_grouped, cta_coord, atom.box)


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
    """i1 SSA -> i32 SSA (0/1) via arith.extsi."""
    e = _emitter()
    if isinstance(b, bool):
        return e.ssa("i32", f"arith.constant {1 if b else 0} : i32")
    assert getattr(b, "type", "") == "i1"
    return e.ssa("i32", f"arith.extsi {b.name} : i1 to i32")


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


def mul_i32(a, b):
    e = _emitter()
    return e.ssa("i32", f"arith.muli {a.name}, {b.name} : i32")


def sub_i32(a, b):
    e = _emitter()
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
    return e.ssa("i32", f"arith.divsi {a.name}, {b.name} : i32")

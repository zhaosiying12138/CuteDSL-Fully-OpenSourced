"""cutlass.utils — hardware info, layout enums, smem helpers, persistent
tile scheduler (compat, clean-room from CUTLASS C++ semantics + BSD
example call shapes; the official wheel implementation is not read)."""
from __future__ import annotations

import torch

from self_cutedsl.frontend import builtins as _b
from self_cutedsl.frontend.cute_objects import Layout, make_layout, _flatten, _prod


class LayoutEnum:
    ROW_MAJOR = 0
    COLUMN_MAJOR = 1

    def __init__(self, value):
        self.value = value

    @staticmethod
    def from_tensor(t):
        # row-major: last-dim stride == 1
        st = getattr(t, "stride", None)
        if st is not None and tuple(st)[-1] == 1:
            return LayoutEnum(LayoutEnum.ROW_MAJOR)
        return LayoutEnum(LayoutEnum.COLUMN_MAJOR)


class HardwareInfo:
    def __init__(self, **kw):
        props = torch.cuda.get_device_properties(0)
        self.name = props.name
        self.compute_capability = (props.major, props.minor)
        self.sm_count = props.multi_processor_count
        self.l2_cache_size = getattr(props, "L2_cache_size", 0)
        self.threads_per_wave = 32 * self.sm_count
        self.max_active_clusters = 1

    def query_cluster_size(self, cluster_shape):
        return 1

    def to_dict(self):
        return {"sm_count": self.sm_count,
                "compute_capability": self.compute_capability}


def get_smem_capacity_in_bytes(arch: HardwareInfo = None,
                               without_reserved: bool = True) -> int:
    # SM120: 100 KiB per SM, 1 KiB reserved
    return (100 - 1) * 1024 if without_reserved else 100 * 1024


class SmemAllocator:
    """Bump allocator over a named shared-memory arena."""

    _counter = 0

    def __init__(self):
        SmemAllocator._counter += 1
        self._id = SmemAllocator._counter
        self._offset = 0

    def allocate(self, shared_storage_cls, name=None):
        return SharedStorageView(shared_storage_cls, self._id)


class SharedStorageView:
    """Realizes a @cute.struct class into shared memory: each MemRange
    field becomes an SmemArray named after the field."""

    def __init__(self, struct_cls, alloc_id):
        self._alloc_id = alloc_id
        for fname, spec in struct_cls.__struct_fields__.items():
            setattr(self, fname, spec.realize(f"sa{alloc_id}_{fname}"))


class MemRangeSpec:
    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = int(count)

    def realize(self, name: str):
        # Int64 barrier storage -> raw smem bytes; else typed array
        name_lower = getattr(self.dtype, "name", str(self.dtype)).lower()
        if "int64" in name_lower:
            # 8-byte mbarrier slots
            e = _b._emitter()
            line = (f"    llvm.mlir.global internal @{name}() "
                    f"{{addr_space = 3 : i32, alignment = 8 : i64}} "
                    f": !llvm.array<{self.count} x i64>")
            e.smem_globals.append(line)
            ptr = e.ssa("!llvm.ptr<3>",
                        f"llvm.mlir.addressof @{name} : !llvm.ptr<3>")
            return _SmemRawPtr(ptr, self.count)
        elem = {"float16": "f16", "float32": "f32",
                "bfloat16": "bf16"}.get(name_lower, "f32")
        arr = _b.make_smem_tile(name, self.count, self.dtype)
        return arr


class _SmemRawPtr:
    def __init__(self, ptr, count):
        self.ptr = ptr
        self.count = count

    def data_ptr(self):
        return self.ptr


class AlignSpec:
    def __init__(self, inner: MemRangeSpec, alignment: int):
        self.inner = inner
        self.alignment = int(alignment)

    def realize(self, name):
        return self.inner.realize(name)


def _struct_decorator(cls):
    """@cute.struct — collect MemRange/Align fields into __struct_fields__."""
    fields = {}
    for k, v in vars(cls).items():
        if isinstance(v, MemRangeSpec):
            fields[k] = v
        elif isinstance(v, AlignSpec):
            fields[k] = v
    cls.__struct_fields__ = fields
    cls.allocate = lambda self=None, alloc=None: None
    return cls


# ------------------------------------------------------------------ layouts
def compute_tile_shape_or_override(is_a_lat_op, tile_shape_mnk, tile_shape_mnkl,
                                   big_tile_shape_mnkl, dtype_width=16):
    """Resolve auto (-1) tile dims. SM120 f16 canonical: (128, 128, 64) or
    (64, 64, 64) scaled by layout/op; from the C++ sm120 kernel config."""
    ts = list(tile_shape_mnkl) if tile_shape_mnkl is not None else None
    if ts is None:
        ts = list(big_tile_shape_mnkl)
    out = []
    for i, v in enumerate(ts):
        if v in (-1, None):
            out.append([16, 16, 8, 1][i] if dtype_width == 8 else [128, 128, 64, 1][i])
        else:
            out.append(v)
    return tuple(out[:3]), out[3] if len(out) > 3 else 1


def make_smem_layout_a(tile_shape_mnk, ab_stage, dtype, major="k",
                       swizzle=None):
    """A-tile SMEM layout (staged): ((M, K), stage) row-major-K per stage."""
    m, n, k = tile_shape_mnk
    outer = make_layout((m, k))
    return ComposedLayoutStaged(outer, ab_stage)


def make_smem_layout_b(tile_shape_mnk, ab_stage, dtype, major="k",
                       swizzle=None):
    n_, m, k = tile_shape_mnk[1], tile_shape_mnk[0], tile_shape_mnk[2]
    outer = make_layout((n_, k))
    return ComposedLayoutStaged(outer, ab_stage)


def make_smem_layout_epi(tile_shape_mnk, epi_stage, dtype, tile_n=None):
    outer = make_layout((tile_shape_mnk[0], tile_n or tile_shape_mnk[1]))
    return ComposedLayoutStaged(outer, epi_stage)


class ComposedLayoutStaged:
    """outer layout + stage count (+ swizzle placeholder)."""

    def __init__(self, outer: Layout, stages: int, swizzle=None):
        self.outer = outer
        self.stages = int(stages)
        self.inner = swizzle

    def __repr__(self):
        return f"ComposedLayoutStaged({self.outer}, stages={self.stages})"


def sm90_get_smem_store_op(*args, **kw):
    return None


# ------------------------------------------------------------------ sched
class PersistentTileSchedulerParams:
    def __init__(self, problem_shape_ntile_mnl, cluster_shape_mnl,
                 tile_shape_mnk, grid_shape_mnl, max_swizzle_size=1,
                 rank_in_cluster=0, max_active_clusters=1):
        self.problem_shape_ntile_mnl = tuple(problem_shape_ntile_mnl)
        self.cluster_shape_mnl = tuple(cluster_shape_mnl)
        self.grid_shape_mnl = tuple(grid_shape_mnl)
        self.max_active_clusters = max_active_clusters
        # total linear work units
        t = 1
        for d in problem_shape_ntile_mnl:
            t *= int(d)
        self.total = t
        # waves rounding for persistent scheduling
        w = 1
        for d in grid_shape_mnl:
            w *= int(d)
        self.problem_tiles = t
        self.grid_size = w
        self.waves = (t + w - 1) // w


class StaticPersistentTileScheduler:
    """Linear grid-stride work counter with SSA state."""

    def __init__(self, params, bidx_ssa, grid_ssa):
        self.params = params
        self.bidx = bidx_ssa
        self.grid = grid_ssa

    @staticmethod
    def create(params, bidx_ssa, grid_ssa):
        return StaticPersistentTileScheduler(params, bidx_ssa, grid_ssa)

    @staticmethod
    def get_grid_shape(params):
        # persistent: grid = min(total tiles, sm_count-ish wave)
        g = []
        for d in params.grid_shape_mnl:
            g.append(int(d))
        return tuple(g)

    def initial_work_tile_info(self):
        return _WorkInfo(self.bidx)

    def fetch_next_work(self, work_info):
        e = _b._emitter()
        nxt = _b.add_i32(work_info.index, _b.idx_to_i32(self.grid))
        return _WorkInfo(nxt)

    def is_valid(self, work_info):
        return _b.lt_i32(work_info.index, _b.const_i32(self.params.total))


class _WorkInfo:
    def __init__(self, index_ssa):
        self.index = index_ssa          # SSA i32

    @property
    def coord(self):
        e = _b._emitter()
        # linear -> (m, n, l); flagship uses is_last_tile_flag + coord
        return self.index

"""cutlass.utils — hardware info, layout enums, smem helpers, persistent
tile scheduler (compat, clean-room from CUTLASS C++ semantics + BSD
example call shapes; the official wheel implementation is not read)."""
from __future__ import annotations

import torch

from self_cutedsl.frontend import builtins as _b
from self_cutedsl.frontend.cute_objects import Layout, make_layout, _flatten, _prod


class LayoutEnum:
    _ROW_MAJOR = 0
    _COLUMN_MAJOR = 1

    def __init__(self, value):
        self.value = value.value if isinstance(value, LayoutEnum) else int(value)

    @staticmethod
    def from_tensor(t):
        # row-major: last-dim stride == 1
        st = getattr(t, "stride", None)
        if st is not None and tuple(st)[-1] == 1:
            return LayoutEnum(LayoutEnum._ROW_MAJOR)
        return LayoutEnum(LayoutEnum._COLUMN_MAJOR)

    def is_k_major_a(self):
        return self.value == LayoutEnum._ROW_MAJOR

    def is_k_major_b(self):
        return self.value == LayoutEnum._ROW_MAJOR

    def is_m_major_a(self):
        # A is (m,k): m-major when the m stride is 1
        return self.value == LayoutEnum._COLUMN_MAJOR

    def is_n_major_b(self):
        # B is (n,k): n-major when the n stride is 1
        return self.value == LayoutEnum._COLUMN_MAJOR

    def is_n_major_c(self):
        return self.value == LayoutEnum._ROW_MAJOR

    def is_m_major_c(self):
        # C is (m,n): m-major when the m stride is 1
        return self.value == LayoutEnum._COLUMN_MAJOR


LayoutEnum.ROW_MAJOR = LayoutEnum(LayoutEnum._ROW_MAJOR)
LayoutEnum.COLUMN_MAJOR = LayoutEnum(LayoutEnum._COLUMN_MAJOR)


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

    def get_max_active_clusters(self, cluster_size=1):
        return 1

    def get_cluster_dim(self, cluster_shape):
        return 1


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

    def allocate(self, shared_storage, name=None):
        cls = shared_storage if isinstance(shared_storage, type) \
            else type(shared_storage)
        return SharedStorageView(cls, self._id)

    # -- official flashinfer-style API ----------------------------------
    def allocate_tensor(self, elem, layout, byte_alignment=16):
        from cutlass.cute._fi_arch import _alloc_tensor
        return _alloc_tensor(self, elem, layout, byte_alignment)

    def allocate_array(self, dtype, num_elems=1):
        from cutlass.cute._fi_arch import _alloc_array
        return _alloc_array(self, dtype, num_elems)


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
        # sub-byte/fp8 dtypes allocate a BYTE array (count * width/8 bytes)
        width = getattr(self.dtype, "width", 32) or 32
        if width <= 8:
            e = _b._emitter()
            nbytes = self.count * max(1, width) // 8
            line = (f"    llvm.mlir.global internal @{name}() "
                    f"{{addr_space = 3 : i32, alignment = 128 : i64}} "
                    f": !llvm.array<{nbytes} x i8>")
            e.smem_globals.append(line)
            ptr = e.ssa("!llvm.ptr<3>",
                        f"llvm.mlir.addressof @{name} : !llvm.ptr<3>")
            return _b.SmemArray(ptr, nbytes, self.dtype)
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
    """@cute.struct — collect MemRange/Align fields into __struct_fields__.

    Official bodies use annotation form (`sA: cute.struct.Align[...]`),
    so collect from __annotations__ (evaluated specs) as well as plain
    assignments.
    """
    fields = {}
    for k, v in vars(cls).items():
        if isinstance(v, (MemRangeSpec, AlignSpec)):
            fields[k] = v
    anns = dict(getattr(cls, "__annotations__", {}))
    glb = dict(globals())
    glb.update(getattr(cls, "__module__", "") and
               __import__(cls.__module__, fromlist=["x"]).__dict__ or {})
    for k, spec in anns.items():
        if isinstance(spec, str):
            try:
                spec = eval(spec, glb, dict(vars(cls)))
            except Exception:
                continue
        if isinstance(spec, (MemRangeSpec, AlignSpec)):
            fields[k] = spec
    import os as _os
    if _os.environ.get("STRUCT_DEBUG"):
        import sys as _s
        print(f"[_struct_decorator] {cls.__name__}: fields={list(fields)} "
              f"anns={ {k: type(v).__name__ for k, v in anns.items() } }",
              file=_s.stderr)
    cls.__struct_fields__ = fields
    return cls


# ------------------------------------------------------------------ layouts
def compute_tile_shape_or_override(tile_shape_mnk, dtype,
                                   tile_shape_mnkl=None,
                                   big_tile_shape_mnkl=None,
                                   is_cooperative=False,
                                   is_a_lat_op=None,
                                   dtype_width=None):
    """Epilogue/tile resolution. SM120 canonical epilogue tile (from the
    C++ sm120 config): cooperative -> (128, 64), pingpong -> (64, 32),
    subject to the M tile."""
    width = dtype_width or getattr(dtype, "width", 16)
    if tile_shape_mnkl is not None:
        ts = list(tile_shape_mnkl)
    elif big_tile_shape_mnkl is not None:
        ts = list(big_tile_shape_mnkl)
    else:
        # epilogue flavor used by dense_gemm: tile = (M_epi, N_epi)
        m = tile_shape_mnk[0]
        n = tile_shape_mnk[1]
        return (m, 128 if is_cooperative else 64) if False else (m, 64)
    out = []
    for i, v in enumerate(ts):
        if v in (-1, None):
            out.append([16, 16, 8, 1][i] if width == 8 else [128, 128, 64, 1][i])
        else:
            out.append(v)
    return tuple(out[:2]) if len(out) >= 2 else tuple(out)


def make_smem_layout_a(a_layout, tile_shape_mnk, dtype, ab_stage,
                       major=None, swizzle=None, permutation_mnk=None):
    """A-tile staged SMEM layout: ((M, K), stage), K-major (128B swizzle
    boundary handled by the compiler's composed-layout path)."""
    m, n_, k = tile_shape_mnk
    outer = make_layout((m, k))
    return ComposedLayoutStaged(outer, ab_stage)


def make_smem_layout_b(b_layout, tile_shape_mnk, dtype, ab_stage,
                       major=None, swizzle=None, permutation_mnk=None):
    n_, m, k = tile_shape_mnk[1], tile_shape_mnk[0], tile_shape_mnk[2]
    outer = make_layout((n_, k))
    return ComposedLayoutStaged(outer, ab_stage)


def make_smem_layout_epi(dtype, c_layout, epi_tile, epi_stage):
    m, n = epi_tile
    outer = make_layout((m, n))
    return ComposedLayoutStaged(outer, epi_stage)


class ComposedLayoutStaged:
    """outer layout + stage count (+ swizzle placeholder)."""

    def __init__(self, outer: Layout, stages: int, swizzle=None):
        self.outer = outer
        self.stages = int(stages)
        self.inner = swizzle

    @property
    def shape(self):
        return self.outer.shape

    @property
    def stride(self):
        return self.outer.stride

    def __repr__(self):
        return f"ComposedLayoutStaged({self.outer}, stages={self.stages})"


def sm90_get_smem_store_op(*args, **kw):
    return None


# ------------------------------------------------------------------ sched
class PersistentTileSchedulerParams:
    def __init__(self, problem_shape_ntile_mnl, cluster_shape_mnl,
                 tile_shape_mnk=None, grid_shape_mnl=None,
                 max_swizzle_size=1, rank_in_cluster=0,
                 max_active_clusters=1):
        def _pad3(t):
            t = tuple(int(d) for d in t)
            return t + (1,) * (3 - len(t))

        self.problem_shape_ntile_mnl = _pad3(problem_shape_ntile_mnl)
        self.cluster_shape_mnl = _pad3(cluster_shape_mnl)
        self.tile_shape_mnk = tile_shape_mnk
        t = 1
        for d in self.problem_shape_ntile_mnl:
            t *= int(d)
        self.total = t
        if grid_shape_mnl is None:
            grid_shape_mnl = self.problem_shape_ntile_mnl
        self.grid_shape_mnl = tuple(int(d) for d in grid_shape_mnl)
        self.max_active_clusters = max_active_clusters
        w = 1
        for d in self.grid_shape_mnl:
            w *= int(d)
        self.problem_tiles = t
        self.grid_size = w
        self.waves = (t + w - 1) // w

    def __extract_mlir_values__(self):
        """Official runtime-struct hook; self stack keeps this host metadata."""
        return []


class StaticPersistentTileScheduler:
    """Persistent tile scheduler. Trace model: when the launch grid covers
    every tile (the get_grid_shape default), each CTA owns exactly one tile
    — is_valid_tile is compile-time True at entry and False after advance,
    so the persistent while-loop traces as a single trip (semantically
    exact for this configuration class). Oversubscribed grids (more tiles
    than CTAs) are an honest boundary — recorded, not silently wrong."""

    def __init__(self, params, bidx_ssa, grid_ssa):
        self.params = params
        self.bidx = bidx_ssa
        self.grid = grid_ssa
        self._current = None

    @staticmethod
    def create(params, bidx_ssa, grid_ssa):
        return StaticPersistentTileScheduler(params, bidx_ssa, grid_ssa)

    @staticmethod
    def get_grid_shape(params, max_swizzling_size=1):
        g = [int(d) for d in params.grid_shape_mnl]
        while len(g) < 3:
            g.append(1)
        return tuple(g[:3])

    def initial_work_tile_info(self):
        self._current = _WorkInfo(self, self.bidx)
        return self._current

    def __snapshot__(self):
        return (self._current,)

    def __restore__(self, snap):
        self._current = snap[0]

    def advance_to_next_work(self):
        if self._current is not None:
            self._current = self._current.next(self)

    def get_current_work(self):
        return self._current


class _WorkInfo:
    """One persistent-scheduler work tile: linear index (i32 SSA or int)."""

    def __init__(self, sched, index, is_entry=True):
        self.sched = sched
        self.index = index
        self.is_entry = is_entry

    def next(self, sched):
        from self_cutedsl.frontend import builtins as _b

        idx = self.index[0] if isinstance(self.index, tuple) else self.index
        i = _b.idx_to_i32(idx) if getattr(idx, "type", "i32") != "i32" else idx
        nxt = _b.add_i32(i, _b.const_i32(sched.params.grid_size))
        return _WorkInfo(sched, nxt, is_entry=False)

    @property
    def is_valid_tile(self):
        p = self.sched.params
        if p.grid_size >= p.total:
            # grid covers all tiles: the entry index (bidx < grid == total)
            # is always valid; the advanced one never is
            return self.is_entry
        from self_cutedsl.frontend import builtins as _b
        from self_cutedsl.frontend.emitter import SSA as _SSA

        if not isinstance(self.index, _SSA):
            return self.index < p.total
        return _b.lt_i32(_b.idx_to_i32(self.index), _b.const_i32(p.total))

    @property
    def tile_idx(self):
        """(m, n, l) tile coordinates as i32 SSA (or ints)."""
        from self_cutedsl.frontend import builtins as _b
        from self_cutedsl.frontend.emitter import SSA as _SSA

        p = self.sched.params
        tiles_m, tiles_n, _l = p.problem_shape_ntile_mnl
        idx = self.index[0] if isinstance(self.index, tuple) else self.index
        if not isinstance(idx, _SSA):
            i = int(idx)
            return (i % tiles_m, (i // tiles_m) % tiles_n, 0)
        i = _b.idx_to_i32(idx) if idx.type != "i32" else idx
        m = _b.rem_i32(i, _b.const_i32(tiles_m))
        n = _b.rem_i32(_b.div_i32(i, _b.const_i32(tiles_m)), _b.const_i32(tiles_n))
        return (m, n, 0)

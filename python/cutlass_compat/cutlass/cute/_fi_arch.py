"""arch extensions + SmemAllocator + runtime fakes for the flashinfer ops."""
from __future__ import annotations

import cutlass
from cutlass._bridge_helpers import _emitter, TypedScalar
from cutlass import dtypes as _dt


# ---------------------------------------------------------------------------
# arch methods (installed onto the compat arch instance)
# ---------------------------------------------------------------------------
def lane_idx(self):
    e = _emitter()
    tid = e.thread_id("x")
    return e.idx_binop("arith.remsi", tid,
                       e.ssa("index", "arith.constant 32 : index"))


def barrier(self):
    _emitter().raw("gpu.barrier")


def fmax(a, b):
    e = _emitter()
    va = a.ssa if isinstance(a, TypedScalar) else a
    vb = b.ssa if isinstance(b, TypedScalar) else b
    c = e.ssa("i1", f"arith.cmpf ogt, {va.name}, {vb.name} : f32")
    return _dt.Float32(e.ssa("f32", f"arith.select {c.name}, {va.name}, {vb.name} : f32"))


def shuffle_sync_bfly(self, val, offset, mask=0xFFFFFFFF, **kw):
    e = _emitter()
    v = val.ssa if isinstance(val, TypedScalar) else val
    off = offset.ssa if isinstance(offset, TypedScalar) else None
    if off is None:
        off = e.ssa("i32", f"arith.constant {int(offset)} : i32")
    vi = e.ssa("i32", f"llvm.bitcast {v.name} : f32 to i32")
    ri = e.ssa(
        "i32",
        'nvvm.inline_ptx "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" '
        f'ro ({vi.name}, {off.name} : i32, i32) -> i32')
    r = e.ssa("f32", f"llvm.bitcast {ri.name} : i32 to f32")
    return _dt.Float32(r)


def cp_async_commit_group(self):
    _emitter().raw('nvvm.inline_ptx "cp.async.commit_group;"')


def cp_async_wait_group(self, n):
    _emitter().raw(f'nvvm.inline_ptx "cp.async.wait_group {int(n)};"')


def mbarrier_init(ptr, count):
    e = _emitter()
    c = e.ssa("i32", f"arith.constant {int(count)} : i32")
    p = ptr.ssa if hasattr(ptr, "ssa") else ptr
    e.raw(f"nvvm.mbarrier.init {p.name}, {c.name} : !llvm.ptr<3>, i32")


def mbarrier_init_fence(self):
    _emitter().raw('nvvm.inline_ptx "fence.mbarrier_init.release.cluster;"')


def griddepcontrol_wait(self):
    _emitter().raw('nvvm.inline_ptx "griddepcontrol.wait;"')


def griddepcontrol_launch_dependents(self):
    _emitter().raw('nvvm.inline_ptx "griddepcontrol.launch_dependents;"')


def install(arch_instance):
    for name, fn in [
        ("lane_idx", lane_idx),
        ("shuffle_sync_bfly", shuffle_sync_bfly),
        ("barrier", barrier),
        ("fmax", fmax),
        ("cp_async_commit_group", cp_async_commit_group),
        ("cp_async_wait_group", cp_async_wait_group),
        ("mbarrier_init_fence", mbarrier_init_fence),
        ("griddepcontrol_wait", griddepcontrol_wait),
        ("griddepcontrol_launch_dependents", griddepcontrol_launch_dependents),
    ]:
        if not hasattr(arch_instance, name):
            setattr(arch_instance, name, fn.__get__(arch_instance))
    # free-function forms used by fp4_common
    import types as _t
    ns = _t.SimpleNamespace(
        shuffle_sync_bfly=shuffle_sync_bfly, fmax=fmax)
    return ns


# ---------------------------------------------------------------------------
# SmemAllocator (cutlass.utils.SmemAllocator)
# ---------------------------------------------------------------------------
def _flat_geom(layout):
    def fl(x):
        out = []
        for e in x:
            if isinstance(e, (tuple, list)):
                out.extend(fl(e))
            else:
                out.append(int(e))
        return out
    st = layout.stride if hasattr(layout, "stride") else layout[1]
    return tuple(fl(layout.shape if hasattr(layout, "shape") else layout[0])), \
        tuple(fl(st))


def _alloc_tensor(allocator, elem, layout, byte_alignment=16):
    from self_cutedsl.frontend import builtins as _b
    shp, _strd = _flat_geom(layout)
    n = 1
    for x in shp:
        n *= x
    SmemAllocator_n[0] += 1
    name = f"fi_smem_{SmemAllocator_n[0]}"
    arr = _b.make_smem_array(name, n, element=elem)
    return arr.get_tensor(layout)


def _alloc_array(allocator, dtype, num_elems=1):
    from self_cutedsl.frontend import builtins as _b
    from cutlass.cute._fi_ext import Pointer
    SmemAllocator_n[0] += 1
    name = f"fi_sarr_{SmemAllocator_n[0]}"
    arr = _b.make_smem_array(name, int(num_elems), element=dtype)
    return Pointer(arr.ptr)


SmemAllocator_n = [0]


# ---------------------------------------------------------------------------
# runtime fakes
# ---------------------------------------------------------------------------
def make_fake_compact_tensor(dtype, shape, stride_order=None, assumed_align=None):
    """Compile-time stand-in: a tiny real CUDA tensor with sym dims -> 1.
    Row-major strides come from the concrete shape; pointers are replaced
    by the real tensors at call time."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    shp = tuple(int(s) for s in shape)
    tmap = {"Float16": torch.float16, "BFloat16": torch.bfloat16,
            "Float32": torch.float32, "Uint8": torch.uint8, "Int32": torch.int32}
    tt = tmap.get(getattr(dtype, "name", ""), torch.float32)
    t = torch.empty(shp, dtype=tt, device="cuda")
    return from_dlpack(t)


def make_fake_stream(use_tvm_ffi_env_stream=False):
    class _FakeStream:
        pass
    return _FakeStream()


# namespace re-exported by cute.runtime
import types as _types
_mk = _types.SimpleNamespace(make_fake_compact_tensor=make_fake_compact_tensor,
                             make_fake_stream=make_fake_stream)

# install arch extensions on first import
def _install_once():
    from cutlass.cute import __getattr__ as _g  # noqa: F401
    import cutlass.cute as _cc
    arch = getattr(_cc, "_arch_instance", None)
    if arch is None:
        # compat exposes arch lazily through builtins.arch
        from self_cutedsl.frontend.builtins import arch as _arch
        arch = _arch
    install(arch)
    return arch


_arch_installed = None
def get_arch():
    global _arch_installed
    if _arch_installed is None:
        _arch_installed = _install_once()
    return _arch_installed

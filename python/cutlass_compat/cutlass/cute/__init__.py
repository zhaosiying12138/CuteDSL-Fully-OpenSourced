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
    if name in ('nvgpu', 'runtime', 'testing', 'struct'):
        from importlib import import_module
        return import_module(f'cutlass.cute.{name}')
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


def make_tiled_copy_tv(atom, thr_layout, val_layout):
    from self_cutedsl.frontend import builtins

    return builtins.make_tiled_copy_tv(atom, thr_layout, val_layout)


def copy(atom, src, dst, pred=None):
    from self_cutedsl.frontend import builtins

    builtins.copy(atom, src, dst, pred)


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


def make_tiled_mma(atom, atom_layout=None):
    from self_cutedsl.frontend import builtins

    return builtins.make_tiled_mma(atom, atom_layout)


def gemm(tiled_mma, acc, a_frag, b_frag):
    from self_cutedsl.frontend import builtins

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

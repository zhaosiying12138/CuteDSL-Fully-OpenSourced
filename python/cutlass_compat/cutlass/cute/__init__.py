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

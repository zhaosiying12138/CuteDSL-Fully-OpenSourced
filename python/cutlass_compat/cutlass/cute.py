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
    raise AttributeError(name)


def printf(fmt: str, *args) -> None:
    from self_cutedsl.frontend import builtins
    builtins.printf(fmt, *args)

# Frontend: AST tracing for @cute.jit / @cute.kernel programs.
# Lazy exports to avoid import cycles with the cutlass compat namespace.
from importlib import import_module

__all__ = ["JitFunction", "KernelFunction", "DynamicHostValue", "builtins"]


def __getattr__(name):
    if name in ("JitFunction", "KernelFunction", "DynamicHostValue"):
        return getattr(import_module(".jit", __package__), name)
    if name == "builtins":
        return import_module(".builtins", __package__)
    raise AttributeError(name)

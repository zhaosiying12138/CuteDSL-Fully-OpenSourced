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

"""jit.py — @cute.kernel / @cute.jit decorators + compile/launch driver.

M2 flow:
  JitFunction.__call__(*args)
    -> binds params from annotations (Constexpr -> python value,
       typed scalars -> DynamicHostValue)
    -> re-executes the host body with a controlled namespace so kernel
       launches are recorded (straight-line host code = plain Python)
    -> for each recorded kernel: AST-trace the device body to MLIR text
    -> compile via selfcute pipeline (cutlass-compiler passes, sm_120a PTX)
    -> load with DriverJit, pack dynamic args, launch, synchronize
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..compiler import compile_mlir_to_ptx, entry_names
from ..runtime import DriverJit, LaunchManifest
from . import builtins as _builtins
from .interp import InterpError, KernelInterpreter, KernelParam


class DynamicHostValue:
    """A runtime scalar known only at launch (kernel ABI argument)."""

    def __init__(self, name: str, dtype):
        self.name, self.dtype = name, dtype

    def __repr__(self):
        return f"<dyn {self.name}:{self.dtype.name}>"


@dataclass
class _KernelRecord:
    emitter: object
    grid: tuple
    block: tuple
    dynamic_params: list = field(default_factory=list)  # DynamicHostValue order


class KernelFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self._params = _scan_params(fn)

    def __call__(self, *args, **kwargs):
        if _host_trace.get("active") is None:
            raise InterpError("kernel called outside @cute.jit trace")
        return _KernelCallStub(self, args, kwargs)


class _KernelCallStub:
    def __init__(self, kf, args, kwargs):
        self.kf, self.args, self.kwargs = kf, args, kwargs

    def launch(self, *, grid, block):
        kf = self.kf
        arg_values = {}
        for p, v in zip(kf._params, self.args):
            arg_values[p.name] = v
        # trace the device code now
        interp = KernelInterpreter(kf.fn, kf._params, arg_values)
        prev = _builtins._active
        _builtins._active = interp
        try:
            emitter = interp.run()
        finally:
            _builtins._active = prev
        dynamic = [arg_values[p.name] for p in kf._params if p.kind == "dynamic"]
        _host_trace["records"].append(_KernelRecord(emitter, grid, block, dynamic))


class JitFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self._params = _scan_params(fn)

    def __call__(self, *args):
        if _host_trace.get("active") is not None:
            raise InterpError("nested @cute.jit call unsupported")

        # 1. bind parameters
        bound = {}
        dynamic_names = []
        for p, v in zip(self._params, args):
            if p.kind == "constexpr":
                bound[p.name] = _as_constexpr(v)
            else:
                bound[p.name] = DynamicHostValue(p.name, p.dtype or _i32())
                dynamic_names.append(p.name)
                _host_trace_runtime[p.name] = v

        # 2. execute the host body with launches recorded
        import inspect
        import textwrap

        src = textwrap.dedent(_strip_decorators(inspect.getsource(self.fn)))
        ns = dict(self.fn.__globals__)
        ns.update(bound)
        _host_trace.update(active=self, records=[])
        try:
            exec(compile(src, f"<jit:{self.__name__}>", "exec"), ns)
            # exec only *defines* the function; run the body with the args
            ns[self.fn.__name__](*args)
            records = list(_host_trace["records"])
        finally:
            _host_trace.update(active=None, records=[])

        # 3. compile + launch each recorded kernel
        for rec in records:
            ptx = compile_mlir_to_ptx(rec.emitter.module_text())
            entries = entry_names(ptx)
            assert entries, "no kernel entry in emitted PTX"
            jit = DriverJit(ptx)
            manifest = LaunchManifest(
                entry=entries[0],
                args=[{"name": getattr(d, "name", f"arg{i}"), "type": "i32"}
                      for i, d in enumerate(rec.dynamic_params)],
                grid=tuple(rec.grid),
                block=tuple(rec.block),
            )
            launch_vals = [
                _host_trace_runtime[d.name] if isinstance(d, DynamicHostValue) else d
                for d in rec.dynamic_params
            ]
            jit.launch(manifest, *launch_vals)
            jit.synchronize()


# runtime values fed to the final launch (set by JitFunction.__call__)
_host_trace: dict = {"active": None, "records": []}
_host_trace_runtime: dict = {}


def _as_constexpr(v):
    return getattr(v, "value", v)


def _i32():
    from cutlass.dtypes import Int32

    return Int32


def _strip_decorators(src: str) -> str:
    lines = src.splitlines()
    while lines and lines[0].strip().startswith("@"):
        lines = lines[1:]
    return "\n".join(lines)


def _scan_params(fn) -> list[KernelParam]:
    from cutlass.dtypes import Constexpr, ConstexprAnnotation, _DType

    import inspect

    sig = inspect.signature(fn)
    out = []
    for name, param in sig.parameters.items():
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else None
        dtype = None
        kind = "dynamic"
        if ann is Constexpr or (isinstance(ann, type) and issubclass(ann, Constexpr)):
            kind = "constexpr"
        elif isinstance(ann, ConstexprAnnotation):
            kind, dtype = "constexpr", ann.dtype
        elif isinstance(ann, _DType):
            dtype = ann
        out.append(KernelParam(name, kind, dtype, param.default))
    return out

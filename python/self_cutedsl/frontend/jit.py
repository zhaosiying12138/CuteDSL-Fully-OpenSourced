"""jit.py — @cute.kernel / @cute.jit decorators + compile/launch driver.

Flow:
  JitFunction.__call__(*args)
    -> binds params from annotations (Constexpr/meta -> python value,
       typed scalars -> DynamicHostValue, tensors -> TensorMeta)
    -> re-executes the host body so kernel launches are recorded
       (straight-line host code = plain Python)
    -> for each recorded kernel: AST-trace the device body to MLIR text
    -> compile via the selfcute pipeline (cutlass-compiler passes,
       sm_120a PTX) and load with DriverJit
    -> pack dynamic args, launch, synchronize

A launch cache keyed on the specialization (constexpr values + tensor
meta) skips tracing/compiling on repeated calls with identical shapes.
"""
from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field

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
    # dynamic ABI entries in signature order:
    #   ("jit", name, mlir_type)  scalar from the jit call
    #   ("tensor", name)          device pointer from the jit call's tensor
    abi: list = field(default_factory=list)


class KernelFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self._name_prefix = ""
        self._params = _scan_params(fn)

    def set_name_prefix(self, prefix: str) -> "KernelFunction":
        self._name_prefix = prefix
        return self

    def __call__(self, *args, **kwargs):
        if _host_trace.get("active") is None:
            raise InterpError("kernel called outside @cute.jit trace")
        return _KernelCallStub(self, args, kwargs)


class _KernelCallStub:
    def __init__(self, kf, args, kwargs):
        self.kf, self.args, self.kwargs = kf, args, kwargs

    def launch(self, *, grid, block):
        kf = self.kf
        arg_values = dict(zip((p.name for p in kf._params), self.args))
        interp = KernelInterpreter(kf.fn, kf._params, arg_values)
        prev = _builtins._active
        _builtins._active = interp
        try:
            emitter = interp.run()
        finally:
            _builtins._active = prev
        abi = []
        for p in kf._params:
            if p.kind == "tensor":
                v = arg_values[p.name]
                if _is_coord_meta(v):
                    continue  # compile-time coordinate meta: not an ABI arg
                names = _host_trace.get("tensor_names", {})
                base = getattr(v, "base", v)  # TiledTensorMeta -> its base meta
                jit_name = names.get(id(v)) or names.get(id(base)) or p.name
                abi.append(("tensor", jit_name))
            elif p.kind == "dynamic":
                mlir_ty = p.dtype.mlir if p.dtype else "i32"
                abi.append(("jit", p.name, mlir_ty))
        _host_trace["records"].append(
            _KernelRecord(emitter, tuple(grid), tuple(block), abi))


class _CachedLaunch:
    """PTX + loaded module + launch plan for one specialization."""

    def __init__(self, ptx: str, plans: list):
        self.ptx = ptx
        self.plans = plans  # [(DriverJit, LaunchManifest)]


class JitFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self._name_prefix = ""
        self._params = _scan_params(fn)
        self._cache: dict = {}

    def set_name_prefix(self, prefix: str) -> "JitFunction":
        self._name_prefix = prefix
        return self

    # ------------------------------------------------------------- helpers
    def _bind(self, args):
        bound = {}
        key_parts = []
        tensors = {}
        params = self._params
        for i, p in enumerate(params):
            v = args[i] if i < len(args) else p.default
            if p.kind == "constexpr":
                bound[p.name] = _as_constexpr(v)
                key_parts.append((p.name, repr(bound[p.name])))
            elif p.kind == "tensor" or _is_tensor_value(v):
                # official jit params are often unannotated; bind by value
                v = _as_tensor_meta(v)
                bound[p.name] = v
                tensors[p.name] = v
                _host_trace.setdefault("tensor_names", {})[id(v)] = p.name
                key_parts.append((p.name, v.shape, v.stride, v.element_type.name))
            else:
                bound[p.name] = DynamicHostValue(p.name, p.dtype or _i32())
                _host_trace_runtime[p.name] = v
        return bound, tuple(key_parts), tensors

    # ------------------------------------------------------------- __call__
    def __call__(self, *args):
        if _host_trace.get("active") is not None:
            raise InterpError("nested @cute.jit call unsupported")

        bound, key, tensors = self._bind(args)
        cached = self._cache.get(key)

        if cached is None:
            records = self._trace(bound, args)
            plans = []
            for rec in records:
                ptx = compile_mlir_to_ptx(rec.emitter.module_text())
                entries = entry_names(ptx)
                assert entries, "no kernel entry in emitted PTX"
                jit = DriverJit(ptx)
                manifest = LaunchManifest(
                    entry=entries[0],
                    args=[{"name": name, "type": _abi_mlir_type(entry)}
                          for entry in rec.abi
                          for name in [entry[1]]],
                    grid=rec.grid,
                    block=rec.block,
                )
                manifest.uses_printf = rec.emitter.uses_printf
                plans.append((jit, manifest, rec.abi))
            cached = plans
            self._cache[key] = plans

        needs_sync = False
        for jit, manifest, abi in cached:
            vals = []
            for entry in abi:
                if entry[0] == "tensor":
                    vals.append(tensors[entry[1]])
                else:
                    vals.append(_host_trace_runtime[entry[1]])
            jit.launch(manifest, *vals)
            needs_sync = needs_sync or getattr(manifest, "uses_printf", False)
        if needs_sync:
            # device printf FIFO flushes at the next context sync point
            DriverJit.synchronize_ctx()
        # NOTE: no ctx-wide sync otherwise — launches are stream-ordered on
        # the default stream; readback paths synchronize explicitly.

    def _trace(self, bound, args):
        src = textwrap.dedent(_strip_decorators(inspect.getsource(self.fn)))
        ns = dict(self.fn.__globals__)
        ns.update(bound)
        _host_trace.update(active=self, records=[])
        try:
            exec(compile(src, f"<jit:{self.__name__}>", "exec"), ns)
            # exec only defines; run the body with the BOUND values so the
            # body sees TensorMeta/DynamicHostValue, not raw torch objects
            ns[self.fn.__name__](*[bound[p.name] for p in self._params])
            return list(_host_trace["records"])
        finally:
            _host_trace.update(active=None, records=[])


def _abi_mlir_type(entry) -> str:
    if entry[0] == "tensor":
        return "ptr"
    ty = entry[2]
    return "f32" if ty.startswith("f") else ("i64" if ty == "i64" else "i32")


def _is_coord_meta(v) -> bool:
    from self_cutedsl.frontend.meta import CoordinateTensorMeta, TiledCoordinateMeta

    b = getattr(v, "base", v)
    return isinstance(v, (CoordinateTensorMeta, TiledCoordinateMeta)) or \
        isinstance(b, CoordinateTensorMeta)


def _is_tensor_value(v) -> bool:
    from .meta import TensorMeta

    if isinstance(v, TensorMeta):
        return True
    return hasattr(v, "data_ptr") and hasattr(v, "shape") and hasattr(v, "dtype")


def _as_tensor_meta(v):
    from .meta import TensorMeta, make_tensor_meta

    if isinstance(v, TensorMeta):
        return v
    if hasattr(v, "data_ptr") and hasattr(v, "shape"):
        return make_tensor_meta(v)  # raw torch tensor (dynamic-layout path)
    raise TypeError(f"expected tensor, got {type(v)}")


def compile_function(fn, *args, **options):
    """cute.compile(fn, *args, options=...) -> cached callable."""
    if not isinstance(fn, JitFunction):
        raise TypeError("cute.compile expects a @cute.jit function")
    fn._bind(args)  # prime specialization (compile happens on first call)
    return fn


# trace state (single-threaded)
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
        elif getattr(ann, "__module__", "") in ("cutlass.cute", "cutlass.dtypes") \
                and getattr(ann, "__name__", "") in ("Tensor", "Shape", "Layout", "Coord"):
            kind = "tensor" if ann.__name__ == "Tensor" else "constexpr"
        out.append(KernelParam(name, kind, dtype, param.default))
    return out

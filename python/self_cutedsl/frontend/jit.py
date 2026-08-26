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

import inspect
import textwrap
from dataclasses import dataclass, field

_DEBUG_BIND = False

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
    def __init__(self, fn, _self=None):
        self.fn = fn
        self.__name__ = fn.__name__
        self._name_prefix = ""
        self._params = _scan_params(fn)
        self._bound_self = _self

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        bound = KernelFunction(self.fn, _self=obj)
        bound._name_prefix = self._name_prefix
        return bound

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

    def launch(self, *, grid, block, cluster=None, stream=None, **kw):
        if cluster is not None and tuple(cluster) not in ((1, 1, 1),):
            raise NotImplementedError(
                "cluster launch unsupported in the single-CTA profile")
        kf = self.kf
        arg_values = dict(zip((p.name for p in kf._params), self.args))
        if getattr(kf, "_bound_self", None) is not None:
            arg_values["self"] = kf._bound_self
        interp = KernelInterpreter(kf.fn, kf._params, arg_values)
        prev = _builtins._active
        _builtins._active = interp
        try:
            emitter = interp.run()
        finally:
            _builtins._active = prev
        abi = []
        for p in kf._params:
            if p.kind == "tma":
                v = arg_values[p.name]
                names = _host_trace.get("tensor_names", {})
                jit_name = names.get(id(v), p.name)
                view = getattr(v, "tma_view", None)
                if view is None and hasattr(v, "recipe"):
                    from self_cutedsl.frontend import builtins as _bb
                    from .meta import TensorMeta as _TM

                    view = _bb._materialize_tma_view(
                        v.recipe, getattr(v, "gmem_meta", None))
                if view is not None:
                    _host_trace_tma[jit_name] = view
                else:
                    pass
                abi.append(("tma", jit_name))
            elif p.kind == "tensor":
                v = arg_values[p.name]
                if _is_coord_meta(v) or _is_tma_view(v):
                    continue  # compile-time coordinate/tma meta: not an ABI arg
                names = _host_trace.get("tensor_names", {})
                base = getattr(v, "base", v)  # TiledTensorMeta -> its base meta
                jit_name = names.get(id(v)) or names.get(id(base)) or p.name
                abi.append(("tensor", jit_name))
            elif p.kind == "dynamic":
                mlir_ty = p.dtype.mlir if p.dtype else "i32"
                abi.append(("jit", p.name, mlir_ty))
        # persistent schedulers address CTAs by linear ctaid.x — flatten
        # an (x,y,z) grid request into x-major
        gx = int(grid[0]) * int(grid[1]) * int(grid[2]) if len(grid) == 3             else int(grid[0])
        _host_trace["records"].append(
            _KernelRecord(emitter, (gx, 1, 1), tuple(block), abi))


class _CachedLaunch:
    """PTX + loaded module + launch plan for one specialization."""

    def __init__(self, ptx: str, plans: list):
        self.ptx = ptx
        self.plans = plans  # [(DriverJit, LaunchManifest)]


class JitFunction:
    def __init__(self, fn, _self=None):
        self.fn = fn
        self.__name__ = fn.__name__
        self._name_prefix = ""
        self._params = _scan_params(fn)
        self._cache: dict = {}
        self._has_self = getattr(self._params, "has_self", False)
        self._bound_self = _self

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # bound-method access on the decorated METHOD — one bound clone
        # per instance so its specialization cache survives across calls
        cached = getattr(obj, "_jit_bound_self", None)
        if cached is not None and cached.__name__ == self.__name__:
            return cached
        bound = JitFunction(self.fn, _self=obj)
        bound._name_prefix = self._name_prefix
        try:
            obj._jit_bound_self = bound
        except Exception:
            pass
        return bound

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
            if _DEBUG_BIND:
                import sys as _s
                print(f"  bind {p.name!r} kind={p.kind} <- {type(v).__name__}",
                      file=_s.stderr)
            if p.kind == "constexpr":
                bound[p.name] = _as_constexpr(v)
                key_parts.append((p.name, repr(bound[p.name])))
            elif p.kind == "tma" or _is_tma_view(v):
                bound[p.name] = v
                tm = _as_tma_host(v)
                _host_trace_tma[p.name] = tm["view"]
                _host_trace.setdefault("tensor_names", {})[id(v)] = p.name
                key_parts.append((p.name, "tma", repr(tm["recipe"])))
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
    def __call__(self, *args, **kwargs):
        if _host_trace.get("active") is not None and self is not _host_trace.get("active"):
            # method invocation (obj(...) via type(obj).__call__) while a
            # host trace runs: tolerate by running nested (host-only)
            return self._call_nested(*args, **kwargs)
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        if _host_trace.get("active") is not None:
            raise InterpError("nested @cute.jit call unsupported")
        if self._has_self and self._bound_self is None \
                and len(args) == len(self._params) + 1:
            # obj(...) via type(obj).__call__(obj, ...) — bind the instance
            self._bound_self = args[0]
            args = args[1:]
        if kwargs:
            # constexpr kwargs: fill trailing params by name
            by_name = {p.name: p for p in self._params}
            for k, v in kwargs.items():
                if k not in by_name:
                    raise TypeError(f"unexpected kwarg {k!r}")
                by_name[k].default = _as_constexpr(v)
        args = list(args)
        while len(args) < len(self._params):
            args.append(self._params[len(args)].default)

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
                elif entry[0] == "tma":
                    vals.append(_host_trace_tma[entry[1]].device_copy)
                else:
                    vals.append(_host_trace_runtime[entry[1]])
            jit.launch(manifest, *vals)
            needs_sync = needs_sync or getattr(manifest, "uses_printf", False)
        if needs_sync:
            # device printf FIFO flushes at the next context sync point
            DriverJit.synchronize_ctx()
        # NOTE: no ctx-wide sync otherwise — launches are stream-ordered on
        # the default stream; readback paths synchronize explicitly.

    def _call_nested(self, *args, **kwargs):
        """Method jit invoked as obj(...) while a trace is active: unwrap
        the implicit instance Python prepends via type(obj).__call__."""
        if self._has_self and self._bound_self is None \
                and len(args) == len(self._params) + 1:
            args = args[1:]
        # host-level execution without a new trace context
        bound, key, tensors = self._bind(list(args))
        return self._trace_with(bound, list(args))

    def _trace_with(self, bound, args):
        src = textwrap.dedent(_strip_decorators(inspect.getsource(self.fn)))
        ns = dict(self.fn.__globals__)
        ns.update(bound)
        if self._has_self and self._bound_self is not None:
            ns["self"] = self._bound_self
        elif self._has_self:
            ns["self"] = None
        _host_trace.update(active=self, records=[])
        try:
            exec(compile(src, f"<jit:{self.__name__}>", "exec"), ns)
            call_vals = [bound[p.name] for p in self._params]
            if self._has_self:
                call_vals = [self._bound_self] + call_vals
            ns[self.fn.__name__](*call_vals)
            return list(_host_trace["records"])
        finally:
            _host_trace.update(active=None, records=[])

    def _trace(self, bound, args):
        src = textwrap.dedent(_strip_decorators(inspect.getsource(self.fn)))
        ns = dict(self.fn.__globals__)
        ns.update(bound)
        if self._bound_self is not None:
            ns["self"] = self._bound_self
        _host_trace.update(active=self, records=[])
        try:
            exec(compile(src, f"<jit:{self.__name__}>", "exec"), ns)
            # exec only defines; run the body with the BOUND values so the
            # body sees TensorMeta/DynamicHostValue, not raw torch objects
            call_vals = [bound[p.name] for p in self._params]
            if self._has_self:
                if self._bound_self is None:
                    raise InterpError("method jit called without instance")
                call_vals = [self._bound_self] + call_vals
            ns[self.fn.__name__](*call_vals)
            return list(_host_trace["records"])
        finally:
            _host_trace.update(active=None, records=[])


def _abi_mlir_type(entry) -> str:
    if entry[0] in ("tensor", "tma"):
        return "ptr"
    ty = entry[2]
    return "f32" if ty.startswith("f") else ("i64" if ty == "i64" else "i32")


def _is_tma_view(v) -> bool:
    from self_cutedsl.frontend.cute_objects import Tensor as HostTensor
    from self_cutedsl.runtime.tensor_map import CUtensorMapView

    return isinstance(v, CUtensorMapView) or (
        isinstance(v, HostTensor) and getattr(v, "is_tma_view", False))


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
    if isinstance(fn, JitFunction):
        fn._bind(args)  # prime specialization (compile on first call)
        return fn
    if callable(fn):
        # Official compile accepts any callable (kernel objects whose
        # __call__ is typically @cute.jit). Semantics: compile-time args
        # are captured; later calls pass only the DYNAMIC args (tensors)
        # and the captured constexpr/stream values are reused.
        fn(*args)
        return _CompiledCallable(fn, args)
    raise TypeError("cute.compile expects a callable")


class _CompiledCallable:
    """Post-cute.compile runner: later calls replace the tensor args of
    the priming signature, reusing captured constexpr scalars."""

    def __init__(self, fn, prime_args):
        self._fn = fn
        self._prime = list(prime_args)
        self._tensor_slots = [i for i, a in enumerate(prime_args)
                              if hasattr(a, "data_ptr") and hasattr(a, "shape")]

    def __call__(self, *dyn_args):
        merged = list(self._prime)
        for slot, new in zip(self._tensor_slots, dyn_args):
            merged[slot] = new
        return self._fn(*merged)


# trace state (single-threaded)
_host_trace: dict = {"active": None, "records": []}
_host_trace_runtime: dict = {}
_host_trace_tma: dict = {}


def _as_tma_host(v):
    """Normalize a host-side TMA argument to {name, recipe, view}."""
    from self_cutedsl.runtime.tensor_map import CUtensorMapView

    if isinstance(v, CUtensorMapView):
        return {"name": getattr(v, "name", "tma"), "recipe": v.recipe, "view": v}
    view = getattr(v, "tma_view", None)      # kernel-glue TmaAtom
    if isinstance(view, CUtensorMapView):
        return {"name": getattr(v, "name", "tma"), "recipe": v.recipe,
                "view": view}
    if isinstance(v, dict):
        return v
    raise TypeError(f"bad TMA argument {type(v)}")


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

    if isinstance(fn, JitFunction):
        fn = fn.fn  # inspect the wrapped def, not the __call__ shim
    sig = inspect.signature(fn)

    class _Params(list):
        has_self = False

    out = _Params()
    first = True
    for name, param in sig.parameters.items():
        if first and name == "self":
            first = False
            out.has_self = True  # decorated METHOD: skip the receiver
            continue
        first = False
        if name in ("args", "kwargs"):
            raise TypeError(
                f"@cute.jit function '{fn.__name__}' must have named "
                f"parameters (got *{name})")
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"@cute.jit function '{fn.__name__}' has *args/**kwargs; "
                f"named parameters required")
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
                and getattr(ann, "__name__", "") in (
                    "Tensor", "Shape", "Layout", "Coord", "TmaTensor",
                    "ComposedLayout", "CopyAtom", "TiledMma"):
            kind = {"Tensor": "tensor", "TmaTensor": "tma",
                    "CopyAtom": "tma"}.get(ann.__name__, "constexpr")
        elif str(getattr(ann, "__module__", "")).startswith("cutlass.") \
                and isinstance(ann, type) and not isinstance(ann, _DType):
            # cutlass.utils scheduler params & friends: compile-time objects
            kind = "constexpr"
        out.append(KernelParam(name, kind, dtype, param.default))
    return out

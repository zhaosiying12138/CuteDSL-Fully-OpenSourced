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
_compile_only = False

from ..compiler import compile_mlir_to_ptx, entry_names
from ..runtime import DriverJit, LaunchManifest
from . import builtins as _builtins
from .interp import InterpError, KernelInterpreter, KernelParam


class DynGridExpr:
    """Lazy launch-grid arithmetic over runtime scalar ABI arguments."""

    def __init__(self, fn, desc):
        self.fn = fn          # dict name->value  -> int
        self.desc = desc

    def __call__(self, vals):
        return int(self.fn(vals))

    def __repr__(self):
        return f"<dyn-grid {self.desc}>"


class DynamicHostValue:
    """A runtime scalar known only at launch (kernel ABI argument)."""

    def __init__(self, name: str, dtype):
        self.name, self.dtype = name, dtype

    def __repr__(self):
        return f"<dyn {self.name}:{self.dtype.name}>"

    def _expr(self, op, other=None, rev=False):
        n = self.name

        def _i(v):
            if hasattr(v, "value") and not hasattr(v, "name"):
                return int(v.value)
            if hasattr(v, "ssa") and hasattr(v, "value"):
                return int(v.value)
            return int(v)

        if op == "ceil_div":
            return DynGridExpr(lambda v, n=n, b=int(other), _i=_i: -(-_i(v[n]) // b),
                               f"ceil_div({n},{other})")
        if op == "neg":
            return DynGridExpr(lambda v, n=n, _i=_i: -_i(v[n]), f"-{n}")
        raise TypeError(op)

    def __neg__(self):
        return self._expr("neg")

    def __floordiv__(self, b):
        return DynGridExpr(lambda v, n=self.name, b=int(b), _i=None: int(v[n]) // b,
                           f"{self.name}//{b}")

    def __rfloordiv__(self, a):
        return DynGridExpr(lambda v, a=int(a), n=self.name: a // int(v[n]),
                           f"{a}//{self.name}")


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
            elif p.kind == "pointer":
                v = arg_values[p.name]
                names = _host_trace.get("tensor_names", {})
                jit_name = names.get(id(v), p.name)
                abi.append(("pointer", jit_name))
            elif p.kind == "composite":
                v = arg_values[p.name]
                names = _host_trace.get("tensor_names", {})
                for field_name, field_value in vars(v).items():
                    if not _is_tensor_value(field_value):
                        continue
                    base = getattr(field_value, "base", field_value)
                    jit_name = names.get(id(field_value)) or \
                        names.get(id(base)) or f"{p.name}_{field_name}"
                    abi.append(("tensor", jit_name))
            elif p.kind == "dynamic":
                mlir_ty = p.dtype.mlir if p.dtype else "i32"
                abi.append(("jit", p.name, mlir_ty))
        requested_grid = tuple(grid) + (1,) * (3 - len(grid))
        if getattr(grid, "flatten_x", False):
            linear = 1
            for dimension in requested_grid[:3]:
                linear *= int(dimension)
            launch_grid = (linear, 1, 1)
        else:
            launch_grid = requested_grid[:3]
        _host_trace["records"].append(
            _KernelRecord(emitter, launch_grid, tuple(block), abi))


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
    def _bind(self, args, kwargs=None):
        bound = {}
        key_parts = []
        tensors = {}
        params = self._params
        if _host_trace.get("active") is not None:
            # nested @cute.jit call during an active trace: values are
            # trace-time objects (views, fragments, plain ints) — bind
            # verbatim; the body executes inline into the outer trace.
            kw = kwargs or {}
            for i, p in enumerate(params):
                if p.name in kw:
                    bound[p.name] = kw.pop(p.name)
                else:
                    bound[p.name] = args[i] if i < len(args) else p.default
            if getattr(self, "_bound_self", None) is not None:
                bound["self"] = self._bound_self
            return bound, (), {}
        for i, p in enumerate(params):
            v = args[i] if i < len(args) else p.default
            if _DEBUG_BIND:
                import sys as _s
                print(f"  bind {p.name!r} kind={p.kind} <- {type(v).__name__}",
                      file=_s.stderr)
            forced_constexpr = getattr(self, "_forced_constexpr_names", ())
            if p.name in forced_constexpr:
                bound[p.name] = _as_constexpr(v)
                key_parts.append((p.name, repr(bound[p.name])))
            elif p.kind == "constexpr":
                bound[p.name] = _as_constexpr(v)
                key_parts.append((p.name, repr(bound[p.name])))
            elif p.kind == "pointer":
                bound[p.name] = v
                tensors[p.name] = v
                _host_trace.setdefault("tensor_names", {})[id(v)] = p.name
                key_parts.append((
                    p.name,
                    "pointer",
                    getattr(getattr(v, "dtype", None), "name", None),
                    int(getattr(v, "memspace", 0)),
                    int(getattr(v, "_assumed_align", 1)),
                ))
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
            elif p.name == "stream" or hasattr(v, "cuda_stream") or \
                    type(v).__name__ in ("_FakeStream", "CUstream"):
                bound[p.name] = v  # stream handle: launch-time, not an ABI arg
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

        bound, key, tensors = self._bind(args, kwargs)
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
            # A host-only @cute.jit (for example the SF layout scatter)
            # performs its work while _trace executes and has no launch plan.
            # Caching [] would skip the Python body on every same-shaped call;
            # the blockscaled example then initialized SFA but left SFB zero.
            if plans:
                self._cache[key] = plans

        if _compile_only:
            return

        needs_sync = False
        for jit, manifest, abi in cached:
            vals = []
            for entry in abi:
                if entry[0] in ("tensor", "pointer"):
                    vals.append(tensors[entry[1]])
                elif entry[0] == "tma":
                    vals.append(_host_trace_tma[entry[1]].device_copy)
                else:
                    vals.append(_host_trace_runtime[entry[1]])
            rec_emitter_printf = getattr(manifest, "uses_printf", False)
            rt = {e[1]: vals[i] for i, e in enumerate(abi)
                  if e[0] not in ("tensor", "tma")}
            g = manifest.grid
            if any(hasattr(x, "fn") for x in g):
                g = tuple(x(rt) if hasattr(x, "fn") else x for x in g)
                manifest = LaunchManifest(entry=manifest.entry,
                                          args=manifest.args, grid=g,
                                          block=manifest.block)
                manifest.uses_printf = rec_emitter_printf
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
        bound, key, tensors = self._bind(list(args), kwargs)
        active_interp = getattr(_builtins, "_active", None)
        if isinstance(active_interp, KernelInterpreter):
            return active_interp.inline_call(self.fn, bound)
        return self._trace_with(bound, list(args), nested=True)

    def _trace_with(self, bound, args, nested=False):
        src = textwrap.dedent(_strip_decorators(inspect.getsource(self.fn)))
        ns = dict(self.fn.__globals__)
        ns.update(bound)
        if self._has_self and self._bound_self is not None:
            ns["self"] = self._bound_self
        elif self._has_self:
            ns["self"] = None
        # nested calls inline into the OUTER trace: preserve its state and
        # propagate the callee's return value to the caller's trace
        outer_active = _host_trace.get("active")
        outer_records = _host_trace.get("records")
        records = outer_records if outer_records is not None else []
        _host_trace.update(active=self, records=records)
        ret_box = []
        try:
            exec(compile(src, f"<jit:{self.__name__}>", "exec"), ns)
            call_vals = [bound[p.name] for p in self._params]
            if self._has_self:
                call_vals = [self._bound_self] + call_vals

            def _invoke():
                ret_box.append(ns[self.fn.__name__](*call_vals))

            _invoke()
            if nested:
                return ret_box[0] if ret_box else None
            return list(_host_trace["records"])
        finally:
            if nested:
                _host_trace.update(active=outer_active,
                                   records=records)
            else:
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
    if entry[0] in ("tensor", "pointer", "tma"):
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
        bound_call = getattr(fn, "__call__", None)
        params = getattr(bound_call, "_params", ())
        forced_names = {
            param.name
            for param, value in zip(params, args)
            if param.kind == "dynamic"
            and getattr(param.dtype, "is_integer", False)
            and isinstance(getattr(value, "value", value), int)
            and not isinstance(getattr(value, "value", value), bool)
        }
        if any(param.kind == "pointer" for param in params) and forced_names:
            return _DeferredCompiledCallable(fn, args, forced_names)

        global _compile_only
        previous = _compile_only
        _compile_only = True
        try:
            fn(*args)
        finally:
            _compile_only = previous
        return _CompiledCallable(fn, args)
    raise TypeError("cute.compile expects a callable")


class _CompiledCallable:
    """Post-cute.compile runner: later calls replace the tensor args of
    the priming signature, reusing captured constexpr scalars."""

    def __init__(self, fn, prime_args):
        self._fn = fn
        self._prime = list(prime_args)

    @staticmethod
    def _is_stream(value):
        return hasattr(value, "cuda_stream") or \
            type(value).__name__ in ("_FakeStream", "CUstream")

    @classmethod
    def _compatible(cls, old, new):
        if hasattr(old, "data_ptr") and hasattr(old, "shape"):
            return hasattr(new, "data_ptr") and hasattr(new, "shape")
        if hasattr(old, "_pointer"):
            return isinstance(new, int) or hasattr(new, "_pointer")
        if cls._is_stream(old):
            return cls._is_stream(new)
        if isinstance(old, bool):
            return isinstance(new, bool)
        if isinstance(old, str):
            return isinstance(new, str)
        if hasattr(old, "value") and not hasattr(old, "data_ptr"):
            dtype_name = getattr(getattr(old, "dtype", None), "name", "")
            if "Float" in dtype_name or dtype_name.startswith("f"):
                return isinstance(new, (int, float)) and not isinstance(new, bool)
            return isinstance(new, int) and not isinstance(new, bool)
        if isinstance(old, int):
            return isinstance(new, int) and not isinstance(new, bool)
        if isinstance(old, float):
            return isinstance(new, (int, float)) and not isinstance(new, bool)
        return isinstance(new, type(old))

    @classmethod
    def _captured_slot(cls, value):
        return cls._is_stream(value) or isinstance(value, (bool, int, float, str)) or \
            (hasattr(value, "value") and not hasattr(value, "data_ptr"))

    def _merge(self, dyn_args):
        merged = list(self._prime)
        # Runtime args preserve source order and may omit captured constexpr
        # values.  Bind compatible slots directly; skip a captured slot only
        # when its type cannot accept the next runtime value.  This keeps
        # runtime M/eps scalars positional while still skipping a constexpr
        # launch bound placed immediately before a runtime stream.
        prime_index = 0
        for new in dyn_args:
            while prime_index < len(merged) and not self._compatible(
                    merged[prime_index], new):
                old = merged[prime_index]
                if self._captured_slot(old):
                    prime_index += 1
                    continue
                raise TypeError(
                    f"runtime argument {type(new).__name__} cannot replace "
                    f"compiled slot {prime_index} ({type(old).__name__})")
            if prime_index >= len(merged):
                raise TypeError("too many runtime arguments for compiled callable")
            old = merged[prime_index]
            if isinstance(new, int) and hasattr(old, "_pointer"):
                new = type(old)(
                    new,
                    old.dtype,
                    old.memspace,
                    assumed_align=getattr(old, "_assumed_align", None),
                )
            merged[prime_index] = new
            prime_index += 1
        return merged

    def __call__(self, *dyn_args):
        return self._fn(*self._merge(dyn_args))


class _DeferredCompiledCallable(_CompiledCallable):
    """Pointer-based wrapper specialized after runtime layout sizes arrive."""

    def __init__(self, fn, prime_args, forced_names):
        super().__init__(fn, prime_args)
        self._forced_names = frozenset(forced_names)

    def __call__(self, *dyn_args):
        merged = self._merge(dyn_args)
        bound_call = getattr(self._fn, "__call__")
        bound_call._forced_constexpr_names = self._forced_names
        return bound_call(*merged)


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
    try:
        resolved_annotations = inspect.get_annotations(fn, eval_str=True)
    except (NameError, TypeError):
        resolved_annotations = {}

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
        ann = resolved_annotations.get(
            name,
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else None,
        )
        dtype = None
        kind = "dynamic"
        if ann is Constexpr or (isinstance(ann, type) and issubclass(ann, Constexpr)):
            kind = "constexpr"
        elif isinstance(ann, ConstexprAnnotation):
            kind, dtype = "constexpr", ann.dtype
        elif isinstance(ann, _DType):
            dtype = ann
        elif getattr(ann, "__name__", "") == "Pointer" and \
                str(getattr(ann, "__module__", "")).startswith("cutlass.cute"):
            kind = "pointer"
        elif isinstance(ann, type) and hasattr(ann, "__extract_mlir_values__"):
            kind = "composite"
        elif getattr(ann, "__module__", "") in ("cutlass.cute", "cutlass.dtypes") \
                and getattr(ann, "__name__", "") in (
                    "Tensor", "Shape", "Layout", "Coord", "TmaTensor",
                    "ComposedLayout", "CopyAtom", "TiledMma", "MmaAtom",
                    "ThrMma", "TiledCopy", "Tile"):
            kind = {"Tensor": "tensor", "TmaTensor": "tma",
                    "CopyAtom": "tma"}.get(ann.__name__, "constexpr")
        elif str(getattr(ann, "__module__", "")).startswith("cutlass.") \
                and isinstance(ann, type) and not isinstance(ann, _DType):
            # cutlass.utils scheduler params & friends: compile-time objects
            kind = "constexpr"
        out.append(KernelParam(name, kind, dtype, param.default))
    return out

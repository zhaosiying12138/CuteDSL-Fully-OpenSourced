"""Load the UNMODIFIED vendored flashinfer CuTe-DSL operators into a process
whose `cutlass` is the self stack compat package.

Nothing under third_party/flashinfer-src is edited. We only assemble the
import environment in sys.modules:

- `flashinfer.*` submodules that the operator files import relatively
  (api_logging, utils, trace.*) are imported from the vendored tree by
  file path, with a lightweight package shell around them.
- `flashinfer.jit.*` (nvcc JIT ext-loading, unused at FLASHINFER_LOGLEVEL=0)
  gets tiny stub modules.

The operator sources themselves (cute_dsl/rmsnorm_fp4quant.py, fp4_common.py,
...) are the vendored files, byte-for-byte.
"""
import importlib.util
import sys
import types
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[3] / "third_party/flashinfer-src/flashinfer"


def _load(name: str, path: Path, pkg: str):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[] if pkg else None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _package(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def ensure_loaded() -> types.ModuleType:
    """Idempotently assemble `flashinfer` around the self stack; returns the
    cute_dsl subpackage."""
    if "flashinfer" in sys.modules and hasattr(sys.modules["flashinfer"], "cute_dsl"):
        return sys.modules["flashinfer"].cute_dsl

    # package shell (light __init__: version only)
    fi = types.ModuleType("flashinfer")
    fi.__path__ = [str(VENDOR)]
    sys.modules["flashinfer"] = fi
    _load("flashinfer.version", VENDOR / "version.py", pkg=False)
    from flashinfer.version import __version__, __git_version__
    fi.__version__, fi.__git_version__ = __version__, __git_version__

    # jit stubs (nvcc ext loading is never used by the cute_dsl path here)
    jit = _stub("flashinfer.jit")

    def _no_build(self):
        raise RuntimeError("flashinfer jit stub: build_and_load called")

    class _NoSpec:
        def build_and_load(self):
            _no_build(self)

    def _gen(*a, **k):
        return _NoSpec()

    _stub("flashinfer.jit.spdlog", gen_spdlog_module=_gen)
    _stub("flashinfer.jit.api_log_stats", gen_api_log_stats_module=_gen)
    _stub("flashinfer.jit.env", has_flashinfer_jit_cache=lambda *a, **k: False,
          has_flashinfer_cubin=lambda *a, **k: False)

    # real (unmodified) support modules the operators import
    _load("flashinfer.api_logging", VENDOR / "api_logging.py", pkg=False)
    _load("flashinfer.utils", VENDOR / "utils.py", pkg=False)
    trace = _load("flashinfer.trace", VENDOR / "trace/__init__.py", pkg=True)
    trace.__path__ = [str(VENDOR / "trace")]
    tmpl = _load("flashinfer.trace.templates", VENDOR / "trace/templates/__init__.py", pkg=True)
    tmpl.__path__ = [str(VENDOR / "trace/templates")]
    _load("flashinfer.trace.template", VENDOR / "trace/template.py", pkg=False)
    _load("flashinfer.trace.solution", VENDOR / "trace/solution.py", pkg=False)
    _load("flashinfer.trace.templates.norm", VENDOR / "trace/templates/norm.py", pkg=False)

    # cute_dsl package: real __init__ pulls blockscaled/gemm etc. — instead a
    # shell that lazily exposes the operator modules themselves (unmodified)
    cd = types.ModuleType("flashinfer.cute_dsl")
    cd.__path__ = [str(VENDOR / "cute_dsl")]
    sys.modules["flashinfer.cute_dsl"] = cd
    fi.cute_dsl = cd
    return cd


def load_operator(module_name: str):
    """Import an UNMODIFIED vendored cute_dsl operator module by name."""
    cd = ensure_loaded()
    full = f"flashinfer.cute_dsl.{module_name}"
    if full not in sys.modules:
        _load(full, VENDOR / "cute_dsl" / f"{module_name}.py", pkg=False)
        setattr(cd, module_name, sys.modules[full])
    return sys.modules[full]


def load_b12x_operator():
    """Load the unmodified SM12x MoE entry and its real NVFP4 kernel stack."""
    ensure_loaded()
    load_operator("fp4_common")
    load_operator("utils")

    # API tracing and CUDA-version discovery are host integrations outside the
    # kernel path under test. Keep them inert, as for the existing JIT stubs.
    _stub(
        "flashinfer.trace.templates.moe",
        b12x_fused_moe_trace=None,
        b12x_moe_wrapper_run_trace=None,
    )
    _stub(
        "flashinfer.jit.cpp_ext",
        get_cuda_version=lambda: types.SimpleNamespace(major=13, minor=3),
    )

    gemm = VENDOR / "gemm"
    fused = VENDOR / "fused_moe"
    fused_cute = fused / "cute_dsl"
    sm12x = fused_cute / "blackwell_sm12x"
    _package("flashinfer.gemm", gemm)
    _package("flashinfer.gemm.kernels", gemm / "kernels")
    _package("flashinfer.fused_moe", fused)
    _package("flashinfer.fused_moe.cute_dsl", fused_cute)
    _package("flashinfer.fused_moe.cute_dsl.blackwell_sm12x", sm12x)

    dense_name = "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x"
    if dense_name not in sys.modules:
        _load(
            dense_name,
            gemm / "kernels/dense_blockscaled_gemm_sm120_b12x.py",
            pkg=False,
        )
    for module_name in (
        "moe_activation",
        "moe_static_kernel",
        "moe_micro_kernel",
        "moe_dynamic_kernel",
    ):
        full = f"flashinfer.fused_moe.cute_dsl.blackwell_sm12x.{module_name}"
        if full not in sys.modules:
            _load(full, sm12x / f"{module_name}.py", pkg=False)

    dispatch_name = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
    if dispatch_name not in sys.modules:
        _load(dispatch_name, sm12x / "moe_dispatch.py", pkg=False)
    entry_name = "flashinfer.fused_moe.cute_dsl.b12x_moe"
    if entry_name not in sys.modules:
        _load(entry_name, fused_cute / "b12x_moe.py", pkg=False)
    return sys.modules[entry_name]

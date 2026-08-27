"""perf_common.py — shared helpers for the dual-stack performance suite.

Every benchmark in tools/perf/ runs the SAME unmodified operator sources in
two environments:

  self      — this repo's fully open stack (python/ + python/cutlass_compat)
  official  — the official nvidia-cutlass-dsl wheel (.venv-reference)

Timing methodology (uniform across families):
  * per-iteration CUDA events (median / p10 / p90 reported), or the operator's
    own ``testing.benchmark`` harness where the demo entry point provides it;
  * fixed warmup before measurement;
  * GPU state (utilization, clocks, power, memory) recorded next to every
    result so measurements taken on a shared GPU are honestly labeled.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ensure_stack() -> str:
    """Import the right ``cutlass`` and return which stack is active.

    Prefers whatever ``import cutlass`` resolves to; if nothing is importable,
    sets up the self stack paths from this repository.
    """
    try:
        import cutlass  # noqa: F401
        if getattr(cutlass, "__self_cutedsl__", False):
            return "self"
        return "official"
    except ImportError:
        pass
    for p in (ROOT / "python", ROOT / "python/cutlass_compat"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import cutlass  # noqa: F401
    assert getattr(cutlass, "__self_cutedsl__", False), \
        "cutlass resolved to something that is not the self stack"
    return "self"


def init_cuda_context() -> None:
    import torch
    try:
        import cutlass
        cutlass.cuda.initialize_cuda_context()
    except Exception:
        torch.zeros(1, device="cuda")
    torch.cuda.synchronize()


_SMI_CANDIDATES = ("/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi")
_QUERY = ("name,driver_version,utilization.gpu,memory.used,"
          "clocks.sm,clocks.mem,power.draw,temperature.gpu")


def gpu_metadata() -> dict:
    """Best-effort GPU state snapshot (works on WSL2 via the lib path)."""
    for smi in _SMI_CANDIDATES:
        try:
            out = subprocess.run(
                [smi, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                vals = [v.strip() for v in
                        out.stdout.strip().splitlines()[0].split(",")]
                meta = dict(zip(_QUERY.split(","), vals))
                meta["source"] = smi
                return meta
        except Exception:
            continue
    return {"source": "unavailable"}


def cuda_time_us(fn, warmup: int, iters: int) -> dict:
    """Per-iteration CUDA-event timing with robust statistics (µs)."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))

    def pct(q: float) -> float:
        k = min(iters - 1, max(0, int(round(q * (iters - 1)))))
        return ms[k]

    return {
        "median_us": round(pct(0.5) * 1000.0, 2),
        "p10_us": round(pct(0.1) * 1000.0, 2),
        "p90_us": round(pct(0.9) * 1000.0, 2),
        "min_us": round(ms[0] * 1000.0, 2),
        "warmup": warmup,
        "iters": iters,
    }


def envelope(family: str, stack: str, params: dict) -> dict:
    """Result envelope shared by every bench script."""
    import torch
    return {
        "family": family,
        "stack": stack,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "gpu_meta": gpu_metadata(),
        "params": params,
        "cases": [],
    }


def write_result(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    print(f"[perf] wrote {path}")


def median_of(values):
    return statistics.median(values) if values else None

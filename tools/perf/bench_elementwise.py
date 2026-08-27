#!/usr/bin/env python3
"""bench_elementwise.py — official Ampere elementwise_add demo (FP32)
performance, verbatim operator source, dual-stack.

Imports the UNMODIFIED demo module and drives its own ``testing.benchmark``
harness (CUDA events, returns µs/iteration).

Usage:
  .venv-self/bin/python tools/perf/bench_elementwise.py [--warmup 5] [--iters 300]
  .venv-reference/bin/python tools/perf/bench_elementwise.py
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perf"))

import perf_common as P

SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (8192, 8192),
]

_TIME_RE = re.compile(r"Kernel execution time:\s*([\d.]+)\s*ms")
_THR_RE = re.compile(r"Achieved memory throughput:\s*([\d.]+)\s*GB/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/perf"))
    args = ap.parse_args()

    stack = P.ensure_stack()
    sys.path.insert(0, str(
        ROOT / ("third_party/cutlass/examples/python/CuTeDSL/cute/ampere/"
                "kernel/elementwise")))

    import cutlass
    import elementwise_add as EW

    P.init_cuda_context()
    cases = []
    for m, n in SHAPES:
        label = f"elementwise {m}x{n}"
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                us = EW.run_elementwise_add(
                    m, n, cutlass.Float32,
                    warmup_iterations=args.warmup,
                    iterations=args.iters,
                    skip_ref_check=True, benchmark=True)
            out = buf.getvalue()
            if us is None:
                ms = _TIME_RE.findall(out)
                us = float(ms[-1]) * 1000.0 if ms else None
            gbs = _THR_RE.findall(out)
            gbs = float(gbs[-1]) if gbs else 3 * m * n * 4 / (us * 1e-6) / 1e9
            cases.append({"shape": [m, n], "median_us": round(us, 1),
                          "gb_s": round(gbs, 1)})
            print(f"[{stack}] {label}: {us:.1f} us ({gbs:.1f} GB/s)",
                  flush=True)
        except Exception:
            err = traceback.format_exc(limit=6)
            print(f"[{stack}] {label}: FAIL\n{err}", flush=True)
            cases.append({"shape": [m, n], "status": "FAIL",
                          "error_tail": err[-800:]})

    params = {"warmup": args.warmup, "iters": args.iters, "dtype": "float32",
              "shapes": SHAPES}
    payload = P.envelope("elementwise", stack, params)
    payload["kernel"] = ("ampere/kernel/elementwise/elementwise_add.py "
                         "UNMODIFIED")
    payload["cases"] = cases
    P.write_result(Path(args.out_dir) / f"elementwise_{stack}.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""bench_blockscaled.py — official blackwell_geforce blockscaled NVFP4
cooperative GEMM performance, verbatim operator source, dual-stack.

Imports the UNMODIFIED demo module and drives its own ``testing.benchmark``
harness. ``skip_ref_check=True``: performance only; golden verification lives
in tests/python/test_blockscaled_verbatim.py.

Usage:
  .venv-self/bin/python tools/perf/bench_blockscaled.py [--warmup 10] [--iters 50]
  .venv-reference/bin/python tools/perf/bench_blockscaled.py
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perf"))

import perf_common as P

# (M, N, K, L)
SHAPES = [
    (1024, 1024, 1024, 1),
    (1024, 4096, 4096, 1),
    (4096, 4096, 4096, 1),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/perf"))
    args = ap.parse_args()

    stack = P.ensure_stack()
    for p in (
        ROOT / "third_party/cutlass/examples/python/CuTeDSL",
        ROOT / ("third_party/cutlass/examples/python/CuTeDSL/cute/"
                "blackwell_geforce/kernel/blockscaled_gemm"),
    ):
        sys.path.insert(0, str(p))

    import cutlass
    import dense_blockscaled_gemm_persistent_cooperative as bs

    P.init_cuda_context()
    cases = []
    for mnkl in SHAPES:
        label = f"blockscaled {mnkl[:3]}"
        try:
            us = bs.run(
                mnkl,
                cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
                cutlass.Float8E4M3FN, 16, cutlass.Float16,
                a_major="k", b_major="k", c_major="n",
                tile_shape_mnk=(128, 128, 128), epi_tile=(128, 128),
                tolerance=0.1,
                warmup_iterations=args.warmup, iterations=args.iters,
                skip_ref_check=True)
            m, n, k, _ = mnkl
            tflops = 2 * m * n * k / (us * 1e-6) / 1e12
            cases.append({"shape": list(mnkl), "median_us": round(us, 1),
                          "tflops": round(tflops, 1)})
            print(f"[{stack}] {label}: {us:.1f} us ({tflops:.1f} TF/s)",
                  flush=True)
        except Exception:
            err = traceback.format_exc(limit=6)
            print(f"[{stack}] {label}: FAIL\n{err}", flush=True)
            cases.append({"shape": list(mnkl), "status": "FAIL",
                          "error_tail": err[-800:]})

    params = {"warmup": args.warmup, "iters": args.iters,
              "tile_shape_mnk": [128, 128, 128], "epi_tile": [128, 128],
              "dtypes": "nvfp4xnvfp4->fp16, sf e4m3 sv16",
              "shapes": SHAPES}
    payload = P.envelope("blockscaled_gemm", stack, params)
    payload["kernel"] = ("blackwell_geforce/kernel/blockscaled_gemm/"
                         "dense_blockscaled_gemm_persistent_cooperative.py "
                         "UNMODIFIED")
    payload["cases"] = cases
    P.write_result(Path(args.out_dir) / f"blockscaled_{stack}.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

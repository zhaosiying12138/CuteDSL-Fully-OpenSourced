#!/usr/bin/env python3
"""bench_torch_baselines.py — community-default context column (op level).

Not a CuTeDSL comparison: this measures what mainstream PyTorch pays for the
FUNCTION-ALIGNED part of each family on the same GPU, so the report can show
the self stack and the official CuTeDSL wheel against the ecosystem default:

  * norm family: torch eager and torch.compile implementations of the
    RMSNorm core (fp32 reduction, fp16 in/out — WITHOUT the NVFP4
    quantization step; the fused operators do strictly more work);
  * GEMM family: torch.matmul (cuBLAS) at the dense_gemm shapes, fp16;
  * MoE family: no aligned community kernel is installed in this
    environment (production W4A16 MoE serving uses triton/flashinfer CUDA
    kernels outside this comparison) — recorded as N/A, not measured.

Usage:
  .venv-self/bin/python tools/perf/bench_torch_baselines.py
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perf"))

import perf_common as P

NORM_SHAPES = [
    (1024, 2048),
    (16384, 2048),
    (4096, 5120),
]

GEMM_SHAPES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4104, 2056, 512),
]


def _rmsnorm_core(x, w, eps: float):
    h = x.float()
    rstd = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return ((h * rstd) * w.float()).to(x.dtype)


def run_norm_cases(torch, stack_label: str, args) -> list:
    cases = []
    compiled = torch.compile(_rmsnorm_core)
    for tokens, hidden in NORM_SHAPES:
        torch.manual_seed(42)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)
        w = torch.randn(hidden, device="cuda", dtype=torch.float16)
        for name, fn in (("torch_eager", _rmsnorm_core), ("torch_compile", compiled)):
            label = f"{name} B={tokens} H={hidden}"
            try:
                stats = P.cuda_time_us(lambda: fn(x, w, 1e-6),
                                       args.warmup, args.iters)
                per_row = 2 + 2  # read x, write out (fp16)
                stats["gb_s"] = round(
                    per_row * tokens * hidden /
                    (stats["median_us"] * 1e-6) / 1e9, 1)
                stats["impl"] = name
                stats["shape"] = [tokens, hidden]
                cases.append(stats)
                print(f"[{stack_label}] {label}: {stats['median_us']} us "
                      f"({stats['gb_s']} GB/s)", flush=True)
            except Exception:
                err = traceback.format_exc(limit=4)
                print(f"[{stack_label}] {label}: FAIL\n{err}", flush=True)
                cases.append({"impl": name, "shape": [tokens, hidden],
                              "status": "FAIL", "error_tail": err[-500:]})
    return cases


def run_gemm_cases(torch, stack_label: str, args) -> list:
    cases = []
    for m, n, k in GEMM_SHAPES:
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        label = f"torch.matmul {m}x{n}x{k} fp16"
        try:
            stats = P.cuda_time_us(lambda: a @ b, args.warmup, args.iters)
            stats["tflops"] = round(
                2 * m * n * k / (stats["median_us"] * 1e-6) / 1e12, 1)
            stats["impl"] = "cublas"
            stats["shape"] = [m, n, k]
            cases.append(stats)
            print(f"[{stack_label}] {label}: {stats['median_us']} us "
                  f"({stats['tflops']} TF/s)", flush=True)
        except Exception:
            err = traceback.format_exc(limit=4)
            print(f"[{stack_label}] {label}: FAIL\n{err}", flush=True)
            cases.append({"impl": "cublas", "shape": [m, n, k],
                          "status": "FAIL", "error_tail": err[-500:]})
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/perf"))
    args = ap.parse_args()

    global torch
    import torch
    P.init_cuda_context()

    payload = P.envelope("community_baselines", "torch",
                         {"warmup": args.warmup, "iters": args.iters,
                          "norm_shapes": NORM_SHAPES,
                          "gemm_shapes": GEMM_SHAPES,
                          "torch": torch.__version__})
    payload["notes"] = [
        "norm baselines cover the RMSNorm core only (no NVFP4 quantize); "
        "the fused CuTeDSL operators do strictly more work per row",
        "MoE: no aligned community kernel in this environment (N/A)",
    ]
    payload["cases"] = run_norm_cases(torch, "torch", args) + \
        run_gemm_cases(torch, "torch", args)
    P.write_result(Path(args.out_dir) / "community_baselines.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

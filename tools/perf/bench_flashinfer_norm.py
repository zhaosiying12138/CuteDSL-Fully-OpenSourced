#!/usr/bin/env python3
"""bench_flashinfer_norm.py — rmsnorm_fp4quant + add_rmsnorm_fp4quant perf.

Runs the UNMODIFIED vendored flashinfer operators
(third_party/flashinfer-src @ pinned commit) in whichever stack this
interpreter is set up as (self via tools/perf/run_all_perf.sh, official via
.venv-reference). Performance only — no golden verification here (that is
tests/python/test_flashinfer_norm_verbatim.py).

Shapes: (tokens, hidden) pairs representative of mainstream LLM serving —
Qwen3-30B-A3B hidden 2048 (prefill- and decode-scale token counts) and a
5120-hidden config (DeepSeek-class).  All use the documented NVFP4 recipe
(block_size 16, fixed global scale) exactly as the correctness tests do.

Usage:
  .venv-self/bin/python tools/perf/bench_flashinfer_norm.py [--warmup 10] [--iters 50]
  .venv-reference/bin/python tools/perf/bench_flashinfer_norm.py
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perf"))
sys.path.insert(0, str(ROOT / "tests/python/support"))

import perf_common as P

SHAPES = [
    (1024, 2048),    # prefill-scale batch, Qwen3-30B-A3B hidden
    (16384, 2048),   # large prefill, Qwen3-30B-A3B hidden
    (4096, 5120),    # DeepSeek-class hidden
]

GLOBAL_SCALE = (448 * 448 * 6.0) ** 0.5 / 448.0


def run_family(ops, op_name: str, stack: str, args) -> list:
    import torch
    cases = []
    for tokens, hidden in SHAPES:
        torch.manual_seed(42)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)
        w = torch.randn(hidden, device="cuda", dtype=torch.float16)
        gs_t = torch.tensor([GLOBAL_SCALE], device="cuda", dtype=torch.float32)
        y = torch.empty(tokens, hidden // 2, device="cuda", dtype=torch.uint8)
        s = torch.empty(tokens, hidden // 16, device="cuda", dtype=torch.uint8)

        if op_name == "rmsnorm_fp4quant":
            call = lambda: ops.rmsnorm_fp4quant(  # noqa: E731
                x, w, y, s, global_scale=gs_t, eps=1e-6, block_size=16)
        else:
            residual = torch.randn(
                tokens, hidden, device="cuda", dtype=torch.float16)
            call = lambda: ops.add_rmsnorm_fp4quant(  # noqa: E731
                x, residual, w, y, s,
                global_scale=gs_t, eps=1e-6, block_size=16)

        label = f"{op_name} B={tokens} H={hidden}"
        try:
            stats = P.cuda_time_us(call, args.warmup, args.iters)
            # bytes actually crossed: x (f16 in), residual (f16 in+out for the
            # add variant), w (f16, negligible), packed fp4 y (H/2 bytes/row),
            # scale factors (H/16 bytes/row)
            per_row = {"rmsnorm_fp4quant": 2 + 0.5 + 0.0625,
                       "add_rmsnorm_fp4quant": 2 + 2 + 2 + 0.5 + 0.0625}[op_name]
            stats["gb_s"] = round(
                per_row * tokens * hidden /
                (stats["median_us"] * 1e-6) / 1e9, 1)
            stats["shape"] = [tokens, hidden]
            cases.append(stats)
            print(f"[{stack}] {label}: {stats['median_us']} us "
                  f"({stats['gb_s']} GB/s)", flush=True)
        except Exception:
            err = traceback.format_exc(limit=4)
            print(f"[{stack}] {label}: FAIL\n{err}", flush=True)
            cases.append({"shape": [tokens, hidden], "status": "FAIL",
                          "error_tail": err[-600:]})
    return cases


def parse_shapes(text: str):
    shapes = []
    for part in text.split(","):
        tokens, hidden = (int(v) for v in part.strip().split("x"))
        shapes.append((tokens, hidden))
    return shapes


def main() -> int:
    global SHAPES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--shapes", type=parse_shapes, default=None,
                    help="override shape matrix, e.g. 1024x2048,16384x2048")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/perf"))
    args = ap.parse_args()
    if args.shapes:
        SHAPES = args.shapes

    stack = P.ensure_stack()
    P.init_cuda_context()

    import flashinfer_verbatim as FV
    params = {"warmup": args.warmup, "iters": args.iters,
              "block_size": 16, "global_scale": GLOBAL_SCALE,
              "dtype": "float16", "shapes": SHAPES}

    for op_name, module in (
        ("rmsnorm_fp4quant", "rmsnorm_fp4quant"),
        ("add_rmsnorm_fp4quant", "add_rmsnorm_fp4quant"),
    ):
        ops = FV.load_operator(module)
        payload = P.envelope(f"flashinfer_{op_name}", stack, params)
        payload["kernel"] = f"flashinfer cute_dsl/{module}.py UNMODIFIED"
        payload["cases"] = run_family(ops, op_name, stack, args)
        P.write_result(Path(args.out_dir) / f"norm_{op_name}_{stack}.json",
                       payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

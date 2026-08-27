#!/usr/bin/env python3
"""bench_flashinfer_b12x.py — B12x fused MoE (W4A16 NVFP4) performance.

Runs the UNMODIFIED vendored flashinfer b12x MoE entry
(flashinfer/fused_moe/cute_dsl/b12x_moe.py @ pinned commit) in whichever
stack this interpreter is set up as. Performance only — the golden
verification lives in tests/python/test_flashinfer_b12x_verbatim.py.

Shapes are model-anchored MoE configurations:
  * Qwen3-30B-A3B (E=128, hidden 2048, intermediate 768, top-8) at a
    decode-scale and a prefill-scale token count;
  * a 32-expert / 4096-hidden / 1024-intermediate configuration.

Weights are random NVFP4-packed bytes and scale-factor tensors carry a
constant representable value (2^-6): MMA latency on NVIDIA GPUs is
data-independent, so timing is unaffected while the encodings stay legal.

Usage:
  .venv-self/bin/python tools/perf/bench_flashinfer_b12x.py [--warmup 3] [--iters 20]
  .venv-reference/bin/python tools/perf/bench_flashinfer_b12x.py
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

# (experts, hidden, intermediate, topk, tokens)
SHAPES = [
    (128, 2048, 768, 8, 64),     # Qwen3-30B-A3B decode batch
    (128, 2048, 768, 8, 2048),   # Qwen3-30B-A3B prefill batch
    (32, 4096, 1024, 4, 4096),   # large-expert prefill configuration
]

SCALE_VALUE = 0.015625  # 2^-6, exactly representable in e4m3


def _scales(experts: int, rows: int, k: int, device):
    """Constant-filled scale tensor in the operator's six-mode MMA layout."""
    import torch
    m_tiles = (rows + 127) // 128
    k_tiles = (k // 16 + 3) // 4
    storage = torch.full(
        (experts, m_tiles, k_tiles, 32, 4, 4),
        SCALE_VALUE, dtype=torch.float8_e4m3fn, device=device)
    return storage.permute(3, 4, 1, 5, 2, 0)


def run_cases(ops, stack: str, args) -> list:
    import torch
    cases = []
    for experts, hidden, inter, topk, tokens in SHAPES:
        torch.manual_seed(7)
        x = (torch.randn(tokens, hidden, device="cuda") * 0.2).to(torch.bfloat16)
        w1 = torch.randint(
            0, 256, (experts, 2 * inter, hidden // 2),
            device="cuda", dtype=torch.uint8)
        w2 = torch.randint(
            0, 256, (experts, hidden, inter // 2),
            device="cuda", dtype=torch.uint8)
        s1 = _scales(experts, 2 * inter, hidden, x.device)
        s2 = _scales(experts, hidden, inter, x.device)
        ids = torch.randint(
            0, experts, (tokens, topk), device="cuda", dtype=torch.int32)
        routes = torch.full(
            (tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32)
        alpha = torch.ones(experts, device="cuda", dtype=torch.float32)
        fc2_in_scale = torch.ones(1, device="cuda", dtype=torch.float32)

        def call():
            return ops.b12x_fused_moe(
                x, w1, s1, w2, s2, ids, routes, experts, topk,
                w1_alpha=alpha, w2_alpha=alpha,
                fc2_input_scale=fc2_in_scale, quant_mode="nvfp4")

        label = (f"E={experts} H={hidden} I={inter} topk={topk} "
                 f"tokens={tokens}")
        try:
            stats = P.cuda_time_us(call, args.warmup, args.iters)
            flops = 3 * tokens * topk * hidden * inter  # fc1 + gate + fc2
            stats["tflops"] = round(
                flops / (stats["median_us"] * 1e-6) / 1e12, 1)
            stats["shape"] = {
                "experts": experts, "hidden": hidden, "intermediate": inter,
                "topk": topk, "tokens": tokens}
            cases.append(stats)
            print(f"[{stack}] b12x {label}: {stats['median_us']} us "
                  f"({stats['tflops']} TF/s useful)", flush=True)
        except Exception:
            err = traceback.format_exc(limit=6)
            print(f"[{stack}] b12x {label}: FAIL\n{err}", flush=True)
            cases.append({"shape": label, "status": "FAIL",
                          "error_tail": err[-800:]})
    return cases


def parse_shapes(text: str):
    shapes = []
    for part in text.split(","):
        e, h, i, topk, tokens = (int(v) for v in part.strip().split(":"))
        shapes.append((e, h, i, topk, tokens))
    return shapes


def main() -> int:
    global SHAPES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", type=parse_shapes, default=None,
                    help="override matrix, e.g. 8:1024:512:2:256")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/perf"))
    args = ap.parse_args()
    if args.shapes:
        SHAPES = args.shapes

    stack = P.ensure_stack()
    P.init_cuda_context()

    import flashinfer_verbatim as FV
    ops = FV.load_b12x_operator()

    params = {"warmup": args.warmup, "iters": args.iters,
              "quant_mode": "nvfp4", "act_dtype": "bfloat16", "shapes": SHAPES}
    payload = P.envelope("flashinfer_b12x_moe", stack, params)
    payload["kernel"] = ("flashinfer fused_moe/cute_dsl/b12x_moe.py + "
                         "blackwell_sm12x/* UNMODIFIED")
    payload["cases"] = run_cases(ops, stack, args)
    P.write_result(Path(args.out_dir) / f"b12x_moe_{stack}.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

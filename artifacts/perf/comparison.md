# Dual-stack performance comparison (RTX 5090 Laptop)

Generated: 2026-08-26T01:26:54 — median of 3 runs.

| Benchmark | Official (ms) | Self (ms) | Official GB/s | Self GB/s | Self/Official |
|---|---|---|---|---|---|
| elementwise-1024x1024 | 0.078 | 0.034 | 161.47 | 368.79 | 228.4% |
| elementwise-2048x2048 | 0.064 | 0.064 | 785.15 | 789.66 | 100.6% |
| elementwise-8192x8192 | 1.111 | 1.099 | 724.8 | 732.47 | 101.1% |

## Self-stack GEMM (frontend warp-GEMM, m16n8k16 mma.sync, SMEM+ldmatrix)

| Kernel | Shape | Time | Throughput |
|---|---|---|---|
| self warp-GEMM (this repo, no TMA, 1-warp CTA) | 512³ f16→f32 | 0.232 ms | 1156 GFLOP/s |
| torch.matmul (cuBLAS SGEMM f32, reference) | 512³ | 0.065 ms | 4099 GFLOP/s |
| official CuTeDSL dense_gemm sm120 (TMA pipeline, 4-warp, persistent, from frozen baseline) | 4096³ fp16 | 1.597 ms | 43,090 GFLOP/s |

Faithful record: the self-stack GEMM is a correctness-first 1-warp-per-CTA
kernel without TMA, multi-warp tiling, or pipelining — ~28% of cuBLAS f32
and ~2.7% of the official tuned TMA GEMM. Closing this gap is the M6/M7
work (TMA-tiled multi-warp GEMM with warp specialization).

## Self-stack pipelined TMA GEMM (2-CTA x 64x8 tile, 2-stage)

| Shape (per-CTA useful work) | Time | Useful GFLOP/s |
|---|---|---|
| 128x16x512 (block-diagonal tiles) | 0.079 ms | 26.6 |
| 128x16x1024 | 0.078 ms | 53.5 |

Faithful record: this is the M6.3 correctness-first pipeline kernel with a
narrow N=8 tile (one mma atom in N); useful-FLOP density is inherently low
per CTA. The dominant remaining gaps to the official 43 TFLOP/s dense GEMM:
N-direction atom tiling (128-wide), multi-warp (8-12 warps), deeper stages,
warp specialization. Each is an additive step on the now-verified pipeline
skeleton.

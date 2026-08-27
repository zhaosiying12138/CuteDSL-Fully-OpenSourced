# Performance summary — self stack vs official CuTeDSL (RTX 5090 Laptop, sm_120a)

Generated: 2026-08-28T00:29:37

## elementwise_add (FP32, Ampere demo)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x1024 | 20.6 | 19.4 | 610.0 GB/s | 649.0 GB/s | **106.2%** |
| 2048x2048 | 81.5 | 64.4 | 617.2 GB/s | 781.0 GB/s | **126.6%** |
| 8192x8192 | 1278.2 | 1263.4 | 630.0 GB/s | 637.4 GB/s | **101.2%** |

Family mean: **111.3%** of official.

## dense_gemm (FP16, tile 64x64x64)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 2048x2048x2048x1 | 180.2 | 191.3 | 95.3 TF/s | 89.8 TF/s | **94.2%** |
| 4096x4096x4096x1 | 1590.3 | 1568.6 | 86.4 TF/s | 87.6 TF/s | **101.4%** |
| 4104x2056x512x1 | 104.7 | 125.5 | 82.5 TF/s | 68.9 TF/s | **83.4%** |

Family mean: **93.0%** of official.

## blockscaled GEMM (NVFP4 coop, tile 128x128x128)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x1024x1024x1 | 12.0 | 40.3 | 179.5 TF/s | 53.3 TF/s | **29.8%** |
| 1024x4096x4096x1 | 93.4 | 124.3 | 367.7 TF/s | 276.5 TF/s | **75.1%** |
| 4096x4096x4096x1 | 262.2 | 395.8 | 524.2 TF/s | 347.2 TF/s | **66.2%** |

Family mean: **57.0%** of official.

## flashinfer rmsnorm_fp4quant (FP16->NVFP4)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x2048 | 19.52 | 53.95 | 275.3 GB/s | 99.6 GB/s | **36.2%** |
| 16384x2048 | 88.67 | 89.6 | 969.7 GB/s | 959.6 GB/s | **99.0%** |
| 4096x5120 | 30.18 | 36.42 | 1780.6 GB/s | 1475.5 GB/s | **82.9%** |

Family mean: **72.7%** of official.

## flashinfer add_rmsnorm_fp4quant

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x2048 | 14.11 | 72.26 | 975.4 GB/s | 190.5 GB/s | **19.5%** |
| 16384x2048 | 275.94 | 294.4 | 798.0 GB/s | 748.0 GB/s | **93.7%** |
| 4096x5120 | 169.44 | 198.21 | 812.2 GB/s | 694.3 GB/s | **85.5%** |

Family mean: **66.2%** of official.

## flashinfer b12x fused MoE (W4A16 NVFP4)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| experts=128 hidden=2048 intermediate=768 topk=8 tokens=64 | 464.45 | 884.29 | 5.2 TF/s | 2.7 TF/s | **52.5%** |
| experts=128 hidden=2048 intermediate=768 topk=8 tokens=2048 | 1170.98 | 1304.0 | 66.0 TF/s | 59.3 TF/s | **89.8%** |
| experts=32 hidden=4096 intermediate=1024 topk=4 tokens=4096 | 2047.3 | 2664.0 | 100.7 TF/s | 77.4 TF/s | **76.9%** |

Family mean: **73.1%** of official.

## Headline

- elementwise_add (FP32, Ampere demo): **111.3%**
- dense_gemm (FP16, tile 64x64x64): **93.0%**
- blockscaled GEMM (NVFP4 coop, tile 128x128x128): **57.0%**
- flashinfer rmsnorm_fp4quant (FP16->NVFP4): **72.7%**
- flashinfer add_rmsnorm_fp4quant: **66.2%**
- flashinfer b12x fused MoE (W4A16 NVFP4): **73.1%**

**Arithmetic mean across families: 78.9% of official CuTeDSL throughput.**

## FlashMLA decode (self-built sm120 core — separate baseline)

No official CuTeDSL MLA exists on sm_120a (SM100 tcgen05/TMEM only); baseline is a PyTorch (einsum/softmax) reference:

| shape | self µs | torch µs | speedup |
|---|---|---|---|
| [1, 1024, 64] | 45.8 | 84.1 | **1.84x** |
| [1, 4096, 128] | 101.5 | 109.3 | **1.08x** |
| [2, 2048, 128] | 100.6 | 214.6 | **2.13x** |

## Measurement conditions

See the `gpu_meta` blocks inside the per-family JSON files (utilization / clocks / power at capture time; shared-GPU captures are labeled there). Re-run the whole suite with `tools/perf/run_all_perf.sh`.

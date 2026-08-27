# Performance summary — self stack vs official CuTeDSL (RTX 5090 Laptop, sm_120a)

Generated: 2026-08-27T23:42:43

## elementwise_add (FP32, Ampere demo)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x1024 | 15.4 | 16.2 | 816.1 GB/s | 777.3 GB/s | **95.1%** |
| 2048x2048 | 70.8 | 78.1 | 710.6 GB/s | 644.6 GB/s | **90.7%** |
| 8192x8192 | 1250.5 | 1244.2 | 644.0 GB/s | 647.3 GB/s | **100.5%** |

Family mean: **95.4%** of official.

## dense_gemm (FP16, tile 64x64x64)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 2048x2048x2048x1 | 179.3 | 245.9 | 95.8 TF/s | 69.9 TF/s | **72.9%** |
| 4096x4096x4096x1 | 1561.2 | 1741.2 | 88.0 TF/s | 78.9 TF/s | **89.7%** |
| 4104x2056x512x1 | 119.1 | 129.1 | 72.5 TF/s | 66.9 TF/s | **92.3%** |

Family mean: **85.0%** of official.

## blockscaled GEMM (NVFP4 coop, tile 128x128x128)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x1024x1024x1 | 10.0 | 31.7 | 214.1 TF/s | 67.8 TF/s | **31.5%** |
| 1024x4096x4096x1 | 78.3 | 131.1 | 438.9 TF/s | 262.1 TF/s | **59.7%** |
| 4096x4096x4096x1 | 239.0 | 405.0 | 575.2 TF/s | 339.4 TF/s | **59.0%** |

Family mean: **50.1%** of official.

## flashinfer rmsnorm_fp4quant (FP16->NVFP4)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x2048 | 15.71 | 64.0 | 342.1 GB/s | 84.0 GB/s | **24.5%** |
| 16384x2048 | 88.8 | 90.85 | 968.3 GB/s | 946.4 GB/s | **97.7%** |
| 4096x5120 | 74.69 | 57.79 | 719.5 GB/s | 929.9 GB/s | **129.2%** |

Family mean: **83.8%** of official.

## flashinfer add_rmsnorm_fp4quant

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| 1024x2048 | 22.02 | 75.23 | 625.0 GB/s | 182.9 GB/s | **29.3%** |
| 16384x2048 | 274.18 | 293.6 | 803.1 GB/s | 750.0 GB/s | **93.4%** |
| 4096x5120 | 165.57 | 198.21 | 831.2 GB/s | 694.3 GB/s | **83.5%** |

Family mean: **68.7%** of official.

## flashinfer b12x fused MoE (W4A16 NVFP4)

| shape | official µs | self µs | official | self | self/official |
|---|---|---|---|---|---|
| experts=128 hidden=2048 intermediate=768 topk=8 tokens=64 | 468.67 | 1267.97 | 5.2 TF/s | 1.9 TF/s | **37.0%** |
| experts=128 hidden=2048 intermediate=768 topk=8 tokens=2048 | 994.66 | 1624.64 | 77.7 TF/s | 47.6 TF/s | **61.2%** |
| experts=32 hidden=4096 intermediate=1024 topk=4 tokens=4096 | 2237.89 | 2923.01 | 92.1 TF/s | 70.5 TF/s | **76.6%** |

Family mean: **58.3%** of official.

## Headline

- elementwise_add (FP32, Ampere demo): **95.4%**
- dense_gemm (FP16, tile 64x64x64): **85.0%**
- blockscaled GEMM (NVFP4 coop, tile 128x128x128): **50.1%**
- flashinfer rmsnorm_fp4quant (FP16->NVFP4): **83.8%**
- flashinfer add_rmsnorm_fp4quant: **68.7%**
- flashinfer b12x fused MoE (W4A16 NVFP4): **58.3%**

**Arithmetic mean across families: 73.5% of official CuTeDSL throughput.**

## FlashMLA decode (self-built sm120 core — separate baseline)

No official CuTeDSL MLA exists on sm_120a (SM100 tcgen05/TMEM only); baseline is a PyTorch (einsum/softmax) reference:

| shape | self µs | torch µs | speedup |
|---|---|---|---|
| [1, 1024, 64] | 45.8 | 84.1 | **1.84x** |
| [1, 4096, 128] | 101.5 | 109.3 | **1.08x** |
| [2, 2048, 128] | 100.6 | 214.6 | **2.13x** |

## Measurement conditions

See the `gpu_meta` blocks inside the per-family JSON files (utilization / clocks / power at capture time; shared-GPU captures are labeled there). Re-run the whole suite with `tools/perf/run_all_perf.sh`.

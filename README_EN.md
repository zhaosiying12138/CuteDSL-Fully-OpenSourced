# CuteDSL-Fully-OpenSourced

**[中文](README.md)**

A **fully open-source** CuTeDSL-compatible compiler stack from the Python
`@cute.jit` frontend down to `sm_120a` PTX: official CUTLASS example kernels
and flashinfer operators compile and run **unmodified** on an RTX 5090
Laptop, with only BSD/Apache-licensed sources and our own code in the path —
no `_cutlass_ir`, no nvcc/NVRTC/ptxas, no closed official wheel.

Measured on the 5090 Laptop, the six operator families with an official
CuTeDSL baseline reach an arithmetic mean of **83% of the closed-source
official implementation** (this capture 82.8%; two formal captures on the
shared GPU span 73.5%–82.8%, conditions recorded per result; see the
[performance table](#5-performance) and `artifacts/perf/summary.md`).

> **Open-source boundary**: everything from the Python API down to textual
> PTX is open (BSD-3). PTX is handed to the CUDA driver's JIT at runtime —
> the driver, SASS generation, and GPU firmware remain NVIDIA-proprietary
> (as with any CUDA program).

## 1. Scope (read first)

- **`sm_120a` only** (GeForce Blackwell / RTX 5090 class), and **only
  tested on that GPU**. Other architectures (SM80/90/100, AMD, CPU) are
  explicitly rejected; no promises are made about them.
- Passing operators are listed in the [operator matrix](#6-operator-matrix).
  **Operators outside the matrix are not guaranteed to compile or run**;
  generalization work is ongoing (see [limitations](#8-known-limitations--roadmap)).
- Our stack keeps operator sources **byte-identical**: every fix lands in
  our compiler layers, never in the operator.
- Performance numbers were captured on a shared GPU (utilization / clocks /
  power recorded per capture in the JSON artifacts); rerun the whole suite
  anytime with `tools/perf/run_all_perf.sh`.

## 2. Stack

```
Python @cute.jit / @cute.kernel            [this repo: AST interpreter + partial eval]
        │  operator sources unmodified (official demos / flashinfer)
        ▼
textual MLIR: cute / arith / scf / gpu / llvm / nvvm   [this repo: emitter]
        │  in-process cutegen type oracle (BSD; the cute dialect's own engine)
        ▼
cutlass-compiler (BSD-3): --one-shot-convert-to-llvm
        │  --attach-nvvm-target=chip=sm_120a --emit-gpu-binary
        ▼
textual PTX (sm_120a, PTX ISA 8.7)         [open LLVM NVPTX backend]
        ▼
CUDA Driver JIT (cuModuleLoadDataEx)       [this repo: runtime; no ptxas anywhere]
```

How the closed gaps are filled is covered in
[section 7](#7-filling-the-closed-source-gaps). Anti-cheat assertions:
`tools/verify_open_stack.py` (official components unimportable; no
nvcc/ptxas/NVRTC) and `tools/inspect_ptx.py` (PTX target/entry/MMA audit).

## 3. Building

Prerequisites and the step-by-step guide live in
**[docs/BUILD.md](docs/BUILD.md)**. Overview:

```bash
tools/build_pinned_llvm.sh        # pinned LLVM 23a60f15 (exact revision cutlass_compiler needs)
tools/build_compiler.sh           # BSD cutlass_compiler + selfcute dialects
tools/cutegen_oracle/build.sh     # in-process cutegen type oracle (nanobind)
tools/make_envs.sh                # .venv-reference (official baseline) / .venv-self (this stack)
.venv-self/bin/pip install nanobind
tools/fetch_third_party.sh        # flashinfer @ 9d33a28e (verbatim operator corpus)
```

## 4. Reproducing correctness

```bash
tools/run_correctness.sh
```

Expected: **259 passed, 0 failed** (host/compiler tests + on-GPU golden +
selfcute LIT), including:

- verbatim golden on the official operators: dense_gemm, blockscaled NVFP4,
  Ampere elementwise, flashinfer rmsnorm/add-rmsnorm/b12x MoE;
- the **generative differential guardrail** between the cutegen oracle and
  the cute dialect verifier (`test_layout_oracle_differential`: 45 layouts
  with dynamic `?` leaves and nested modes × 4 algebra ops, agreeing
  character-for-character);
- the **shape-polymorphic policy test** (`test_dynamic_shape_policy`:
  three lengths of a marked tensor share ONE compiled plan with exact
  results).

PTX audit (optional): run any workload with `DG_DUMP_PTX=1`, then
`tools/inspect_ptx.py`.

## 5. Performance

The SAME unmodified operator sources run in both environments (this stack
vs the official `nvidia-cutlass-dsl==4.7.0` wheel);
`tools/perf/run_all_perf.sh` merges the captures:

| Operator family | Official baseline | vs official |
|---|---|---|
| elementwise add (FP32, Ampere demo, 3 shapes) | ✓ | 113% |
| dense GEMM (FP16, tile 64×64×64, 3 shapes) | ✓ | 101% |
| blockscaled GEMM (NVFP4 coop, tile 128×128×128, 3 shapes) | ✓ | 62% |
| flashinfer rmsnorm_fp4quant (3 shapes) | ✓ | 69% |
| flashinfer add_rmsnorm_fp4quant (3 shapes) | ✓ | 80% |
| flashinfer b12x fused MoE (W4A16 NVFP4, 3 configs) | ✓ | 71% |
| **Arithmetic mean over families** | | **83%** |

- Full per-shape data (µs / GB/s / TFLOP·s⁻¹ / GPU state per capture):
  `artifacts/perf/summary.md` and `artifacts/perf/*.json`;
- Community reference column (torch eager / torch.compile / cuBLAS):
  `tools/perf/bench_torch_baselines.py` → `artifacts/perf/community_baselines.json`;
- FlashMLA decode (our self-built sm120 warp-mma core; no official CuTeDSL
  MLA exists on sm_120a): 1.08–2.13× vs the PyTorch reference — see
  [sm120-cutedsl-flashmla](https://github.com/zhaosiying12138/sm120-cutedsl-flashmla).

## 6. Operator matrix

| Operator | Source (unmodified, vendored commit) | Correctness | Notes |
|---|---|---|---|
| dense GEMM (FP16) | [CUTLASS dense_gemm.py](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py) @ `7107b055` | golden PASS | tile 64×64×64 shape matrix |
| blockscaled GEMM (NVFP4) | [dense_blockscaled_gemm_persistent_cooperative.py](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py) @ `7107b055` | golden PASS | sv=16; sv=32 (MXFP4) is a known boundary |
| elementwise add (FP32) | [elementwise_add.py](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py) @ `7107b055` | golden PASS | |
| rmsnorm + FP4 quant | [flashinfer rmsnorm_fp4quant.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/rmsnorm_fp4quant.py) @ `9d33a28e` | golden PASS (FP4/SF exact) | |
| add-rmsnorm + FP4 quant | [add_rmsnorm_fp4quant.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/add_rmsnorm_fp4quant.py) @ `9d33a28e` | golden PASS | |
| b12x fused MoE (W4A16 NVFP4) | [b12x_moe.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/fused_moe/cute_dsl/b12x_moe.py) + [blackwell_sm12x/](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/fused_moe/cute_dsl/blackwell_sm12x) @ `9d33a28e` | golden PASS (static/micro/dynamic routing) | |
| MLA decode (self-built) | [sm120-cutedsl-flashmla](https://github.com/zhaosiying12138/sm120-cutedsl-flashmla) | golden PASS, 4 shapes | sm120 warp-mma rewrite, not verbatim |

## 7. Filling the closed-source gaps

The key closed pieces of official CuTeDSL are the `_cutlass_ir` /
`cute_nvgpu` extensions and the wheel's private logic. Our counterparts:

- **Frontend**: our own AST interpreter with partial evaluation
  (`python/self_cutedsl/frontend/`) — official decorators, nested jit,
  dynamic launch grids, post-compile scalar rebinding;
- **cute_nvgpu replacement**: structured NVVM ops plus an
  `nvvm.inline_ptx` bridge (elect.sync / shfl / cp.async / TMA multicast /
  mbarrier spin-wait / setmaxnreg / mxf4nvf4 mma, ...);
- **Dynamic shapes/layouts, three layers**:
  - (a) trace-time specialization plus an **opt-in shape-polymorphic
    policy** — `mark_layout_dynamic` now really works: marked extents leave
    the specialization key and ride the runtime-scalar channel; multiple
    shapes share one PTX (see `test_dynamic_shape_policy`);
  - (b) runtime scalars enter the kernel ABI as SSA i32 (dynamic
    `scf.for` / TMA coordinates / launch grid);
  - (c) the object-model path emits cute dialect ops with `?` dynamic
    leaves and takes result types from the **in-process cutegen oracle** —
    the same BSD type engine the cute dialect (and the closed official
    `.so`) uses; a differential guardrail keeps it character-identical to
    the dialect verifier, and inference went from 3.15ms to 0.056ms (56×);
- **TMA**: runtime encoding via `cuTensorMapEncodeTiled` (CUtensorMap bit
  fields are never hand-written);
- **The official wheel exists only in `.venv-reference`** as a baseline;
  it never enters this stack's path.

## 8. Known limitations & roadmap

- **blockscaled sv=32 (MXFP4)**: hangs — an honest boundary; unaligned
  boundary shapes are rejected by the host alignment precheck;
- **Dynamic-leaf identity**: cutegen's default dynamic semantics are
  anonymous (every result `?` is freshly constructed through the
  property-policy arithmetic). Identity tracking requires the
  mlir_dynamic-style emission backend (~600 lines of specializations) that
  the closed `.so` implements — another precise point of the closed/open
  boundary besides tiled_mma/thrfrg; this stack carries dynamic identity
  in the layer-(b) scalar channel instead;
- **Full type-rule convergence**: composition/coalesce/flatten/
  zipped_divide/logical_divide/slice already go through the oracle;
  group_modes/layout_eval and the host static fast-path (the load-bearing
  wall of the verbatim flagship path) remain to be unified on it;
- **Frontend dispatch**: ~260 isinstance chains and a handful of
  `type().__name__` string dispatches remain, to be refactored onto
  trait tables (generalizing `mma_atoms.py`) in batches (copy → mma → tma);
- **Clusters**: (1,1,1) only; multi-CTA clusters / TMA multicast machinery
  exists but is disabled;
- **Performance**: small-batch norm shapes trail the official kernel
  significantly (see summary.md; PTX-level attribution in the tech report).

## 9. Repository layout

```
python/self_cutedsl/    frontend (jit/interp/emitter/builtins), object model, runtime (Driver JIT/TMA)
python/cutlass_compat/  BSD compatibility layer mapping the official cutlass.* API onto this stack
compiler/               selfcute dialect skeletons (ODS; not on the production path)
tools/                  build/verify/bench (perf/ and cutegen_oracle/ live here)
tests/                  259 tests (verbatim golden + differential guardrail + policy tests)
compat/                 frozen toolchain lock and reference baselines
artifacts/              performance/baseline data and the SBOM
third_party/cutlass/    vendored BSD subtree (cutlass_compiler + examples)
docs/                   BUILD.md build guide
```

## 10. License

This repository is BSD-3-Clause. Vendored components: CUTLASS /
cutlass_compiler (BSD-3) and nanobind (BSD-3); the flashinfer corpus
(Apache-2.0) is fetched at a pinned commit, not vendored. The closed
official wheel exists only in the isolated `.venv-reference` as a baseline.

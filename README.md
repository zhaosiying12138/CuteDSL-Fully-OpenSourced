# CuTeDSL-Fully-OpenSourced — SM120 Compatibility Profile

A fully open-sourced software stack that compiles **CuTeDSL-style Python** to
**`sm_120a` PTX** and runs it on an **RTX 5090 Laptop GPU** (GeForce Blackwell,
compute capability 12.0), with **no proprietary compiler in the path**:

```
Python (CuTeDSL-compatible API)          [this repo: python/self_cutedsl]
        │  AST rewrite + tracing + partial evaluation
        ▼
public "cute" MLIR dialect (layout algebra)   [BSD, cutlass_compiler]
   + selfcute.kernel / selfcute.pipeline / selfcute.sm120   [this repo]
        │
        ▼
arith / scf / gpu / LLVM IR / NVVM
        │
        ▼
textual PTX (.target sm_120a, PTX ISA 8.7)   [open LLVM NVPTX backend]
        │
        ▼
CUDA Driver JIT (cuModuleLoadDataEx) + launch [this repo: runtime/]
```

**Open-source boundary**: everything from the Python API down to textual PTX is
open (BSD-3). PTX is handed to the CUDA driver's JIT at runtime — the driver,
SASS generation, and GPU firmware remain NVIDIA-proprietary (as with any CUDA
program).

## Target profile (strictly enforced)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop |
| Compute capability | 12.0 (GeForce Blackwell) |
| PTX target | `sm_120a` only |
| PTX ISA | 8.7+ |
| Output | textual PTX, JIT'd via CUDA Driver API |

Other architectures (SM80/89/90/100/103/121, AMD, CPU), fatbin generation,
multi-architecture builds, WGMMA/tcgen05/TMEM, nvcc/NVRTC/ptxas/libNVVM
fallbacks are **out of scope and rejected by the compiler**.

## Building (quick start)

> Detailed, lock-step instructions are maintained in
> [`docs/BUILD.md`](docs/BUILD.md). The versions below are pinned by the
> toolchain lock in `compat/sm120_toolchain.lock.yaml`.

### Prerequisites

- Linux x86_64 (tested: WSL2, Ubuntu)
- Python 3.12
- NVIDIA driver ≥ 580 (CUDA 13.x userspace) with the target GPU exposed
- CUDA toolkit 13.x (headers/libs for the Driver API; **not** used to compile)
- CMake ≥ 3.24, Ninja, C++17 compiler, ~30 GB free disk

### 1. Build the pinned LLVM/MLIR

The compiler builds against the exact `llvm-project` revision pinned by the
vendored `cutlass_compiler` (see `third_party/cutlass/cutlass_compiler/LLVM_COMMIT`):

```bash
./tools/build_pinned_llvm.sh          # builds llvm+mlir with NVPTX into build-llvm/
```

### 2. Build the compiler stack

```bash
./tools/build_compiler.sh             # builds cutlass_compiler + selfcute dialects
```

### 3. Set up Python environments

Two isolated environments (this is also how compatibility is validated):

```bash
./tools/make_envs.sh                  # creates .venv-reference (official wheel) and .venv-self
```

`reference-env` contains the **official** `nvidia-cutlass-dsl` wheel and is used
*only* to capture frozen baseline results. `self-env` never installs any
proprietary compiler component; all self-stack development and validation runs
there.

### 4. Run tests

```bash
source .venv-self/bin/activate
pytest -m "not sm120"                 # CPU-side: frontend, IR, PTX FileCheck
pytest -m sm120                       # on-GPU: requires the RTX 5090
ninja -C build-compiler check-selfcute-lit
python tools/run_sm120_validation.py \
    --manifest compat/sm120_reference.lock.yaml \
    --output artifacts/sm120-results.json
```

## Tested operators

Results are updated as milestones complete. "Self PASS" means: the unmodified
upstream source, at the pinned commit recorded in
`compat/sm120_reference.lock.yaml`, compiled and executed **entirely by this
stack** on the RTX 5090, with reference checks enabled. "Official PASS" means
the same source passed in the frozen official reference environment (M0
baseline evidence in `artifacts/reference/results.json`).

### Status: M0 complete — official baseline frozen (2026-08-25)

| Operator | Source | Official baseline | Self stack |
|---|---|---|---|
| dense GEMM — 7 configs (FP16/BF16, tiles 64³–128×256×64, M/N-major, batch, boundary) | CUTLASS `blackwell_geforce/dense_gemm.py` @ 7107b055 | ✅ 7/7 PASS | 🔨 M6 |
| block-scaled GEMM — cooperative, 10 configs (NVFP4/MXFP4/MXFP8-E4M3/MXFP8-E5M2/mixed FP4×FP8, tile-K 128/256, epilogue 128×128/64×32, batch, boundary, 1 negative) | CUTLASS `dense_blockscaled_gemm_persistent_cooperative.py` | ✅ 10/10 PASS | 🔨 M7 |
| block-scaled GEMM — ping-pong, 6 configs | CUTLASS `dense_blockscaled_gemm_persistent_pingpong.py` | ✅ 6/6 PASS (tile-K 256 excluded: official bug, see `compat/exclusions.yaml`) | 🔨 M7 |
| RMSNorm + FP4 quant fusion (1007 parametrized tests) | FlashInfer `cute_dsl/rmsnorm_fp4quant.py` @ 9d33a28e | ✅ PASS | 🔨 M2+ |
| Add + RMSNorm + FP4 quant fusion (1188 tests) | FlashInfer `cute_dsl/add_rmsnorm_fp4quant.py` | ✅ PASS | 🔨 M2+ |
| W4A16 FP4 fused MoE B12x (142 functional tests incl. 36 numerical-accuracy configs) | FlashInfer `fused_moe/cute_dsl/blackwell_sm12x/` | ✅ PASS | 🔨 M6+ |

### Self-stack verified so far (M4 complete)

| Capability | Evidence |
|---|---|
| Official `elementwise_add.py` UNMODIFIED, golden PASS (6 shapes incl. boundaries) | `third_party/cutlass/examples/.../elementwise_add.py` in self env |
| Performance parity with official compiler | 2048²: 790 vs 785 GB/s (101%); 8192²: 732 vs 725 (101%); 1024²: 369 vs 161 (228%) — `artifacts/perf/` |
| Real warp MMA: ldmatrix.x4/x2.trans + mma.sync m16n8k16 golden (raw MLIR) | `tests/runtime/sm120/test_mma_m16n8k16.py` |
| Frontend warp GEMM via @cute.jit (SMEM+ldmatrix+mma loop-carried K) golden, 3 shapes | `tests/python/test_gemm_frontend.py` |
| TMA roundtrip (G2S + mbarrier complete_tx + S2G) golden — raw MLIR and @cute.jit frontend | `tests/runtime/sm120/test_tma_g2s_s2g.py`, `tests/python/test_tma_frontend.py` |
| 2-stage TMA pipeline with phase-parity rollover golden (n_tiles 2/4/7, OOB tiles) | `tests/runtime/sm120/test_tma_pipeline.py` |
| setmaxnreg inc/dec + named barriers verified on 5090 | scratch/verify_smr.py runbook |
| PTX fingerprints: ld.global.v4, ldmatrix, mma.sync.aligned.m16n8k16, cp.async.bulk.tensor, mbarrier.* | test asserts |

### Flagship status (dense_gemm / blockscaled)

**Not yet runnable unmodified.** Every hardware primitive they use is
verified; the remaining gap is the CuTe object-model library layer
(dynamic layout algebra in-kernel, TMA atoms/partitioning, TiledMma
partitioning, PipelineTmaAsync driver, tile scheduler). Full inventory:
`compat/sm120_flagship_gap.md`. Self GEMM perf today: 1156 GFLOP/s
(512³) vs cuBLAS 4099 and official TMA GEMM 43,090 (4096³) — drivers of
the gap are no-TMA/1-warp/no-pipeline, i.e. exactly the M6/M7 work.

Toolchain frozen in `compat/sm120_toolchain.lock.yaml`: pinned LLVM `23a60f15`
(5193/5193 targets), cutlass_compiler @ `7107b055` with **check-cute 236/236
passed**, official DSL 4.7.0, torch 2.13.0+cu130, driver 610.53.

## Repository layout

```
compiler/     C++/MLIR: selfcute dialects (kernel, pipeline, sm120) + conversions
python/       self_cutedsl Python package (frontend + runtime bindings)
runtime/      CUDA-Driver-JIT loader, TensorMap runtime, DLPack adapter
compat/       frozen baseline manifests, API surface, exclusions
tests/        python / lit / ptx FileCheck / on-GPU / conformance
tools/        build + validation + inspection scripts
third_party/  vendored BSD cutlass_compiler subtree + FlashInfer operators (Apache-2.0)
```

## License

BSD 3-Clause for this repository's code. Vendored third-party code retains its
original license (see `third_party/PROVENANCE.md`).

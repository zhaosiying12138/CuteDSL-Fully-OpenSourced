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

### Status: M7 complete — blockscaled (NVFP4) cooperative runs UNMODIFIED (2026-08-27)

The flagship `dense_blockscaled_gemm_persistent_cooperative.py` (NVFP4:
f4×f4 + e4m3 scale factors, sv=16, warp-level `mma.sync
kind::mxf4nvf4.block_scale.scale_vec::4X`) compiles and runs **unmodified**
through the self stack with golden checks (`tests/python/test_blockscaled_verbatim.py`):

| Shape | Official TF/s | Self TF/s | Ratio |
|---|---|---|---|
| 1024³ | 101 | 39.7 | 39% |
| 1024×4096×4096 | 501 | 162 | 32% |
| 4096³ | 610 | **411.5** | **67%** |

Honest boundaries: sv=32 (mxfp4/e8m0 SF) hangs (kernel deadlock); unaligned
boundary shapes fail the host tensor-alignment precheck. Full per-kg scale
semantics verified by direct constant-SFA SASS experiments (see DEVLOG
2026-08-26(7)/(8) and the hand-PTX oracle: lane j scales k-group j).

### Status: S5 complete — dense_gemm.py runs UNMODIFIED (2026-08-26)

The flagship `blackwell_geforce/dense_gemm.py` now compiles and runs
**unmodified** end to end on the self stack (Python trace → cute/MLIR →
cutlass-compiler passes → PTX → Driver JIT → RTX 5090), golden-checked
against torch einsum (`tests/python/test_dense_gemm_verbatim.py`):

| Config (tile 64×64×64) | Result |
|---|---|
| 128×128×128 (example default, official tolerance 0.01) | ✅ PASS |
| 4104×2056×512 (boundary, OOB tiles) | ✅ PASS — 95% of official throughput |
| 128×256×128 / 256×256×128 | ✅ PASS (atol 0.05: f16 ULP at these magnitudes) |
| 128×128×256 (4 k-tiles, double stage wrap) | ✅ PASS |

Honest boundaries (general properties, not shape special-cases):
- `tile_m=128` exceeds SMEM (the official runner skips this config too);
- `tile_n=128` needs the multi-epi-tile r2s addressing (single-pass
  epilogue today);
- oversubscribed persistent grids (more tiles than CTAs) need `scf.while`
  in the tracer — grid-covering persistent runs trace single-trip.

### Status: M8 release (2026-08-27)

- 98/98 tests green (`pytest tests/ -q`): object-model algebra/pipeline/dense
  GEMM, verbatim flagship kernels (elementwise, dense_gemm, blockscaled
  cooperative), stage-model + fragment-mode regressions.
- compute-sanitizer: memcheck 0 errors (blockscaled 3-shape), initcheck 0
  (dense), racecheck 0 hazards (4-k-tile stage-wrap config), synccheck 0.
- Perf ledger: `artifacts/perf/comparison.md` — elementwise 101–228%,
  dense_gemm 83%@2048³ / 44%@4096³ / 95% boundary, blockscaled nvfp4
  67%@4096³.
- SBOM + anti-cheat attestations: `artifacts/SBOM.md`.
- Honest boundaries: dense_gemm tile_m=128 SMEM overflow (official skips
  too) & tile_n=128 epi; blockscaled sv=32 hang & unaligned-boundary host
  alignment; oversubscribed persistent grids need scf.while; FlashMLA/
  MLA-decode = SM100-tcgen05 hardware boundary (official arch-gate).

### Status: M0 complete — official baseline frozen (2026-08-25)

| Operator | Source | Official baseline | Self stack |
|---|---|---|---|
| dense GEMM — 7 configs (FP16/BF16, tiles 64³–128×256×64, M/N-major, batch, boundary) | CUTLASS `blackwell_geforce/dense_gemm.py` @ 7107b055 | ✅ 7/7 PASS | 🔨 M6 |
| block-scaled GEMM — cooperative | CUTLASS `dense_blockscaled_gemm_persistent_cooperative.py` | ✅ 10/10 PASS | ✅ nvfp4 verbatim golden PASS (3 shapes + 4-config matrix; 67% @4096³); sv=32/boundary = honest boundary |
| block-scaled GEMM — ping-pong, 6 configs | CUTLASS `dense_blockscaled_gemm_persistent_pingpong.py` | ✅ 6/6 PASS (tile-K 256 excluded: official bug, see `compat/exclusions.yaml`) | 🔨 M7 |
| RMSNorm + FP4 quant fusion (1007 parametrized tests) | FlashInfer `cute_dsl/rmsnorm_fp4quant.py` @ 9d33a28e | ✅ PASS | 🔨 M2+ |
| Add + RMSNorm + FP4 quant fusion (1188 tests) | FlashInfer `cute_dsl/add_rmsnorm_fp4quant.py` | ✅ PASS | 🔨 M2+ |
| W4A16 FP4 fused MoE B12x (142 functional tests incl. 36 numerical-accuracy configs) | FlashInfer `fused_moe/cute_dsl/blackwell_sm12x/` | ✅ PASS | 🔨 M6+ |
| FlashMLA / MLA decode | (a) CUTLASS `blackwell/attention/mla/mla_decode_fp16.py`; (b) IISuperluminaLII/FlashMLA_Windows_Linux_sm120; (c) fernandaspets/vllm_FlashMLA | ❌ **not runnable on this GPU via official CuTeDSL — verified empirically 2026-08-27**: (a) official CuTeDSL arch-gate rejects sm_120a (`expects sm_100a/sm_100f/sm_110a… got sm_120a` — tcgen05/TMEM is data-center-Blackwell-only hardware); (b)+(c) cloned & audited: both are **nvcc/CUDA-extension C++ implementations** (18/21 .cu kernels, zero CuTeDSL/`cutlass.cute`/`cute.jit` references; Python files are test/bench harness only) — import in `.venv-reference` fails with `Unable to import FlashMLA CUDA extension` (repo b) / `No module named flash_mla.cuda` (repo c). Neither can "work on official CuTeDSL" — they are not CuTeDSL programs at all | n/a (hardware boundary) |

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
| CuTe object model: make_tiled_tma_atom / tma_partition / make_tiled_mma call shapes, multi-warp TMA GEMM golden (K 64/128/256) | `tests/python/test_tma_gemm_objects.py` |
| Multi-CTA local-tile TMA GEMM (per-CTA A-row/B-col windows) golden | `tests/python/test_local_tile_kernel.py` |
| 2-stage pipelined multi-CTA TMA GEMM (staged SMEM, parity rollover) golden | `tests/python/test_tma_pipeline_gemm.py` |
| Wide-N (64-col) 8-warp pipelined TMA GEMM golden | `tests/python/test_tma_gemm_wide_n.py` |
| **Persistent full-matrix TMA GEMM with TMA S2G epilogue — 9 configs, 31 TFLOP/s (72% of official)** | `tests/python/test_persistent_gemm.py` |
| **Object model (PLAN_object_model): Python emits cute.* text; algebra owned by C++ passes** — spike + 27 tests | `tests/python/test_object_model_*.py` |
| **S4 dense_gemm-shaped kernel from the full object model (algebra emission + trait table + pipeline driver + generalized TMA), 5 golden configs** | `tests/python/test_object_model_dense_gemm.py` |
| **Official blockscaled cooperative kernel unmodified: NVFP4 random-SF golden for K=128 and K=256 with tile-K 128** | `tests/python/test_blockscaled_stage_model.py` |
| PTX fingerprints: ld.global.v4, ldmatrix, mma.sync.aligned.m16n8k16, cp.async.bulk.tensor, mbarrier.* | test asserts |

### Flagship status (dense_gemm / blockscaled)

Dense GEMM is runnable unmodified, and blockscaled cooperative NVFP4 now
passes random-SF golden for tile-K 128 (including a second global K tile).
The full M7 matrix is still incomplete: tile-K 256 currently hangs, and the
ping-pong / FP8 / mixed paths have not been accepted on the self stack. Full
inventory: `compat/sm120_flagship_gap.md`. Self GEMM perf trajectory: 53.5
(narrow-N pipeline) → 1,361 (wide-N
8-warp) → **30,972 GFLOP/s (persistent, full-matrix, 72% of official
dense_gemm)** — remaining gap drivers: warp specialization (producer
warps), deeper stages, 128-wide tiles, TMA multicast.

Two PTX memory-model races root-caused on the way (missing
fence.proxy.async before TMA S2G; multi-thread mbarrier.init UB) —
details in compat/sm120_flagship_gap.md and git history.

**Object-model architecture (the go-forward route, per PLAN_object_model.md):**
Python never reimplements layout algebra. The emission layer
(`python/self_cutedsl/object_model/`) serializes meta-objects to
`!cute<...>` textual MLIR; result types are computed by the
cutlass-compiler **verifier itself** (probe-and-read-back); trait tables
drive MMA partitioning; driver objects (PipelineTmaAsync) sit on the
verified mbarrier/TMA builtins. New atoms/swizzles are data (table rows
+ serializers), not code.

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

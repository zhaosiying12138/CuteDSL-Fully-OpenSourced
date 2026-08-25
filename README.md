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

Results are updated as milestones complete. "PASS" means: the unmodified
upstream source, at the pinned commit recorded in
`compat/sm120_reference.lock.yaml`, compiled and executed **entirely by this
stack** on the RTX 5090, with reference checks enabled.

### Status: 🔨 in progress (M0 — baseline capture)

| Operator | Source | Status |
|---|---|---|
| dense GEMM (FP16/BF16, TMA pipeline, warp-specialized) | CUTLASS `blackwell_geforce` example | ⏳ |
| block-scaled GEMM — cooperative | CUTLASS `blackwell_geforce` example | ⏳ |
| block-scaled GEMM — ping-pong | CUTLASS `blackwell_geforce` example | ⏳ |
| RMSNorm + FP4 quantization fusion | FlashInfer `cute_dsl/rmsnorm_fp4quant.py` | ⏳ |
| Add + RMSNorm + FP4 quantization fusion | FlashInfer `cute_dsl/add_rmsnorm_fp4quant.py` | ⏳ |
| W4A16 FP4 fused MoE (B12x) | FlashInfer `fused_moe/cute_dsl/blackwell_sm12x/` | ⏳ |

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

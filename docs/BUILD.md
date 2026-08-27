# Build Guide — CuteDSL-Fully-OpenSourced (sm120 profile)

This guide takes a clean machine to: **toolchain built → correctness tests green
(121 passed) → performance benchmarks reproduced**. Every command below is the
exact command used to produce the published results. If anything here fails on
a supported platform, that is a bug — please open an issue.

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux x86_64 (tested: Ubuntu on WSL2) | GPU visible via `/dev/dxg` on WSL2 |
| GPU | NVIDIA GeForce RTX 5090 Laptop (sm_120a) | the only supported target profile |
| NVIDIA driver | ≥ 580 (tested: 610.53, CUDA 13.3 UMD) | kernel-mode driver on the Windows side for WSL2 |
| CUDA toolkit | 13.x | headers only; **never used to compile kernels** |
| Python | 3.12 (via conda/mamba) | used to create the two venvs |
| CMake | ≥ 3.24 | |
| Ninja | any recent | |
| C/C++ compiler | **gcc** (tested: system clang 21 segfaults under heavy LLVM builds) | |
| Disk | ~30 GB | pinned LLVM source+build ≈ 20 GB, venvs ≈ 8 GB |
| RAM | ≥ 16 GB free | |

Two venvs are created; they are the *only* places proprietary NVIDIA compiler
wheels may exist:

- `.venv-reference` — official `nvidia-cutlass-dsl==4.7.0` + torch. Baseline
  capture only.
- `.venv-self` — the fully-open stack (cuda-python, torch, numpy, pytest). Must
  never contain `nvidia-cutlass-dsl` / `_cutlass_ir` / nvcc / NVRTC / ptxas
  (enforced by `tools/verify_open_stack.py`).

## 2. Build the toolchain

All commands run from the repository root.

```bash
# (1) Pinned LLVM — the exact revision required by the vendored cutlass_compiler
#     (commit recorded in third_party/cutlass/cutlass_compiler/LLVM_COMMIT,
#      currently 23a60f15...). Produces scratch/llvm-project/ + build-llvm/.
#     ~30–60 min on a 24-core machine. Override parallelism: LLVM_BUILD_JOBS=16.
tools/build_pinned_llvm.sh

# (2) cutlass_compiler (BSD-3) + the selfcute dialects, against build-llvm/.
#     Produces build-compiler/tools/cutlass-compiler/cutlass-compiler.
tools/build_compiler.sh

# (3) The two Python environments (see above).
tools/make_envs.sh

# (4) Pinned flashinfer sources for the verbatim operator corpus
#     (commit from compat/sm120_toolchain.lock.yaml, currently 9d33a28e...).
tools/fetch_third_party.sh
```

Optional sanity check of the BSD compiler suite:

```bash
ninja -C build-compiler check-cute        # 236/236 LIT tests expected
```

## 3. Verify the open stack

```bash
.venv-self/bin/python tools/verify_open_stack.py
```

This asserts, inside the self environment: `cutlass` /
`nvidia_cutlass_dsl` are **not** importable, `_cutlass_ir` is not loaded,
`nvcc`/`ptxas`/`nvrtc-compiler` are not on PATH, and no EULA-governed CuTeDSL
sources are vendored.

## 4. Correctness tests

```bash
tools/run_correctness.sh
```

Runs (1) host/compiler tests `pytest -m "not sm120"`, (2) on-GPU release-gate
tests `pytest -m sm120`, (3) selfcute dialect LIT checks. Expected on a healthy
tree: **121 passed, 0 failed**, all LIT green.

To audit the PTX actually shipped to the driver JIT for any workload:

```bash
DG_DUMP_PTX=1 .venv-self/bin/python -m pytest -m sm120 -k dense_gemm
.venv-self/bin/python tools/inspect_ptx.py          # scans /tmp/dg_mod_*.ptx
```

## 5. Performance benchmarks

`tools/perf/` contains the dual-stack benchmark suite. The same unmodified
operator sources run twice — once per environment — and the harness merges the
results into `artifacts/perf/`:

```bash
tools/perf/run_all_perf.sh                 # full matrix, both stacks
tools/perf/render_tables.py                # (re)generate summary tables
```

Individual families (all runnable in either stack):

```bash
.venv-self/bin/python tools/perf/bench_elementwise.py
.venv-self/bin/python tools/perf/bench_dense_gemm.py
.venv-self/bin/python tools/perf/bench_blockscaled.py
.venv-self/bin/python tools/perf/bench_flashinfer_norm.py
.venv-self/bin/python tools/perf/bench_flashinfer_b12x.py
```

Each run records GPU state (utilization, clocks, power) alongside the timing
median, so measurements taken on a shared GPU are honestly labeled. See
`tools/perf/README.md` for the methodology and for re-running the suite under
exclusive GPU access.

## 6. Troubleshooting

- **LLVM build dies under clang** — the scripts force gcc; system clang 21 was
  observed to segfault under parallel LLVM builds on this platform.
- **WSL2 reports no GPU** — check `/dev/dxg` exists; `nvidia-smi` inside WSL
  may show no processes even when the GPU is busy (Windows-side processes are
  not enumerated) — `nvidia-smi dmon` still shows real utilization.
- **Interrupted builds** — every step is plain Ninja; re-running the same
  script resumes incrementally.
- **sm120 tests fail with CUDA context errors** — ensure no other process
  holds the primary context exclusively and the driver matches the CUDA 13.x
  userspace.

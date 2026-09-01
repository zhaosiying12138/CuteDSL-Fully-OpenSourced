# Build Guide — CuteDSL-Fully-OpenSourced (sm120 profile)

This guide takes a clean machine to: **toolchain built → correctness tests green
(185 host + 76 sm120 pytest cases, plus selfcute LIT) → performance
benchmarks reproduced**. Every command below is the exact command used to
produce the published results. If anything here fails on
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

The commands above are the tested Linux/WSL2 path.  Native Windows x64 is
supported as a **PTX-generation path** with MSVC; it is not yet a tested
sm120 execution platform.  In particular, an sm86 GPU can build the compiler
and emit `.target sm_120a` PTX, but that PTX must not be loaded or executed on
sm86.

### Native Windows x64 + MSVC prerequisites

Use a Visual Studio Developer PowerShell (not a regular PowerShell) with the
**Desktop development with C++** workload installed.  The required native
components are:

- MSVC x64 build tools, Windows 10/11 SDK, `cl.exe`, `link.exe`, and `rc.exe`;
- CMake ≥ 3.24 and Ninja on `PATH`;
- Git on `PATH` (the pinned LLVM checkout is fetched by the PowerShell build
  script);
- CPython 3.12 x64 and `pip` (a conda installation is not required).

CUDA is **not required to compile or emit PTX**.  This stack uses the LLVM
NVPTX backend and never invokes nvcc, NVRTC, or ptxas.  CUDA-enabled PyTorch
and `cuda-python` are only needed for Python host tests that allocate tensors
or for actually launching kernels.  The NVIDIA Windows driver supplies
`nvcuda.dll` for that runtime path.

Windows build scripts deliberately replace the shell scripts; do not run the
`.sh` files from Windows:

```powershell
# From a VS Developer PowerShell, at the repository root.
# This validates nanobind, torch, and cuda-python in the current Python.
# It does not create a venv or install the unavailable Windows reference wheel.
.\tools\make_envs.ps1

# This step checks out C:\llvm-project at the pinned revision, then downloads
# nothing if that commit is already present.  The build is large: roughly
# 20 GB of source/build space and tens of minutes on a modern workstation.
.\tools\build_pinned_llvm.ps1 -Jobs 16

# Build cutlass_compiler, selfcute-opt, and the LIT test target.
.\tools\build_compiler.ps1 -Jobs 16

# Build the in-process cutegen oracle as a .pyd using MSVC.
.\tools\cutegen_oracle\build.ps1
```

The Windows build scripts use `ccache.exe` by default.  Install ccache and put
it on `PATH`, or pass `-NoCcache` to `build_pinned_llvm.ps1`,
`build_compiler.ps1`, and `cutegen_oracle\build.ps1` to build without it.

The Windows scripts use the current `python.exe` environment.  To generate
PTX without a GPU, put the source tree on `PYTHONPATH` and call the compiler
bridge directly:

```powershell
$env:PYTHONPATH = "$PWD\python;$PWD\python\cutlass_compat"
python -c "from pathlib import Path; from self_cutedsl.compiler import compile_mlir_to_ptx; print(compile_mlir_to_ptx(Path('tests/ptx/vector_add.mlir')))"
```

The expected output contains `.version 8.7`, `.target sm_120a`, and a kernel
`.entry`.  Do not run the `sm120` tests on an sm86 device: their runtime
checks are intentionally for the RTX 5090/sm120a release profile only.  The
native test wrapper skips those tests by default.  On an sm86 Windows machine,
the expected result is 185 passed and 76 deselected, followed by the selfcute
LIT check:

```powershell
.\tools\run_correctness.ps1
# only on a compatible sm120 GPU:
.\tools\run_correctness.ps1 -IncludeSm120
```

Two venvs are created on the reference Linux machine; they are the *only*
places proprietary NVIDIA compiler wheels may exist:

- `.venv-reference` — official `nvidia-cutlass-dsl==4.7.0` + torch. Baseline
  capture only.
- `.venv-self` — the fully-open stack (cuda-python, torch, numpy, pytest). Must
  never contain `nvidia-cutlass-dsl` / `_cutlass_ir` / nvcc / NVRTC / ptxas
  (enforced by `tools/verify_open_stack.py`).

## 2. Linux/WSL2 toolchain build

The commands in this section and the remaining shell examples are for the
tested Linux/WSL2 workflow.  Native Windows users should use the PowerShell
block above rather than translating the `.venv-*`, `scratch/`, or Unix shell
paths below.  All Linux/WSL2 commands run from the repository root.

```bash
# (1) Pinned LLVM — the exact revision required by the vendored cutlass_compiler
#     (commit recorded in third_party/cutlass/cutlass_compiler/LLVM_COMMIT,
#      currently 23a60f15...). Produces scratch/llvm-project/ + build-llvm/.
#     ~30–60 min on a 24-core machine. Override parallelism: LLVM_BUILD_JOBS=16.
tools/build_pinned_llvm.sh

# (2) cutlass_compiler (BSD-3) against build-llvm/, PLUS the selfcute
#     dialect skeletons and their LIT suite (build-selfcute/).
#     Produces build-compiler/tools/cutlass-compiler/cutlass-compiler and
#     build-selfcute/selfcute-opt (check-selfcute-lit runs from there).
tools/build_compiler.sh

# (3) The in-process cutegen type oracle (nanobind binding over the vendored
#     cutegen headers; needs nanobind in .venv-self — see step 4).
tools/cutegen_oracle/build.sh

# (4) The two Python environments (see above).
tools/make_envs.sh
.venv-self/bin/pip install nanobind   # for the oracle binding in step 3

# (5) Pinned flashinfer sources for the verbatim operator corpus
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
tree: **261 passed, 0 failed** (185 host/compiler + 76 sm120), with all LIT
checks green.

To audit the PTX actually shipped to the driver JIT for any workload:

```bash
DG_DUMP_PTX=1 .venv-self/bin/python -m pytest -m sm120 -k dense_gemm
.venv-self/bin/python tools/inspect_ptx.py          # scans the platform temp dir
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

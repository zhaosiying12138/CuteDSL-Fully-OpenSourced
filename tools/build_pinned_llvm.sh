#!/usr/bin/env bash
# Build the llvm-project revision pinned by the vendored cutlass_compiler.
#
# Usage: tools/build_pinned_llvm.sh [llvm-commit]
#   LLVM_COMMIT defaults to third_party/cutlass/cutlass_compiler/LLVM_COMMIT.
#
# Produces:
#   scratch/llvm-project/   source tree at the pinned commit
#   build-llvm/             Ninja build (llvm+mlir, NVPTX+Native, Release+Asserts)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LLVM_COMMIT="${1:-$(cat third_party/cutlass/cutlass_compiler/LLVM_COMMIT | tr -d '[:space:]')}"
echo "[build_pinned_llvm] pinned LLVM commit: ${LLVM_COMMIT}"

mkdir -p scratch
if [ ! -d scratch/llvm-project/.git ]; then
    # Prefer a shared clone from a local llvm-project checkout when available
    # (object store reuse, no network); fall back to a shallow network fetch.
    if [ -d /home/zhaosiying/codebase/llvm-project/.git ]; then
        git clone --shared --no-checkout /home/zhaosiying/codebase/llvm-project scratch/llvm-project
    else
        git init scratch/llvm-project
    fi
fi
if ! git -C scratch/llvm-project remote get-url origin >/dev/null 2>&1; then
    git -C scratch/llvm-project remote add origin git@github.com:llvm/llvm-project.git
fi
if ! git -C scratch/llvm-project cat-file -e "${LLVM_COMMIT}^{commit}" 2>/dev/null; then
    git -C scratch/llvm-project fetch --depth 1 origin "${LLVM_COMMIT}"
fi
git -C scratch/llvm-project checkout --detach "${LLVM_COMMIT}"

cmake -G Ninja -S scratch/llvm-project/llvm -B build-llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_ZSTD=ON

echo "[build_pinned_llvm] building with gcc (system clang 21 segfaults under load)..."
ninja -C build-llvm -j "${LLVM_BUILD_JOBS:-16}"

echo "[build_pinned_llvm] done. Tools in build-llvm/bin (mlir-opt, mlir-tblgen, llc, ...)"

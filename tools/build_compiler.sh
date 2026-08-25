#!/usr/bin/env bash
# Build cutlass_compiler (BSD) against the pinned LLVM in build-llvm/,
# together with our selfcute dialects once they exist.
#
# Prereq: tools/build_pinned_llvm.sh has completed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x build-llvm/bin/mlir-opt ]; then
    echo "[build_compiler] build-llvm missing; run tools/build_pinned_llvm.sh first" >&2
    exit 1
fi

cmake -G Ninja -S third_party/cutlass/cutlass_compiler -B build-compiler \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$ROOT/build-llvm/lib/cmake/llvm" \
    -DMLIR_DIR="$ROOT/build-llvm/lib/cmake/mlir" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCUTLASS_COMPILER_GTEST_SOURCE_DIR="$ROOT/scratch/llvm-project/third-party/unittest"

ninja -C build-compiler cute-opt base-opt cutlass-compiler

echo "[build_compiler] done. Tools:"
echo "  build-compiler/cute_ir/tools/cute-opt/cute-opt"
echo "  build-compiler/base/tools/base-opt/base-opt"
echo "  build-compiler/tools/cutlass-compiler/cutlass-compiler"

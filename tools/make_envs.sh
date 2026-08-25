#!/usr/bin/env bash
# Create the two isolated Python environments used by this project.
#
#   .venv-reference : official nvidia-cutlass-dsl + torch + flashinfer deps.
#                     Used ONLY to capture frozen baseline results (compat/).
#                     This is the sole place where proprietary NVIDIA compiler
#                     wheels are allowed to exist.
#   .venv-self      : the self stack. NEVER contains nvidia-cutlass-dsl,
#                     _cutlass_ir, nvcc/NVRTC/ptxas/libNVVM dependencies.
#
# Usage: tools/make_envs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA="${CONDA:-$(command -v conda || echo /home/zhaosiying/miniforge3/bin/conda)}"

if [ ! -x .venv-reference/bin/python ]; then
    echo "[make_envs] creating .venv-reference (python 3.12)..."
    "$CONDA" create -p "$ROOT/.venv-reference" python=3.12 -y
    .venv-reference/bin/pip install \
        "nvidia-cutlass-dsl==4.7.0" \
        torch --index-url https://download.pytorch.org/whl/cu130 || \
    .venv-reference/bin/pip install "nvidia-cutlass-dsl==4.7.0" torch
fi

if [ ! -x .venv-self/bin/python ]; then
    echo "[make_envs] creating .venv-self (python 3.12)..."
    "$CONDA" create -p "$ROOT/.venv-self" python=3.12 -y
    # cuda-python: open-source CUDA Driver API bindings used by our runtime.
    .venv-self/bin/pip install cuda-python torch numpy pyyaml pytest
fi

echo "[make_envs] done."
echo "[make_envs] reference env:".venv-reference/bin/python -c \
  "'import cutlass; print(cutlass.__version__)'"

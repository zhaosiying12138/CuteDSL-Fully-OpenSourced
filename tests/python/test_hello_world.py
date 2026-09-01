"""M2: the official `01_hello_world.py` tutorial semantics on the self stack.

Same program shape as
third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/01_hello_world.py
(BSD-3), using the compat `cutlass` namespace. Prints from the device via
gpu.printf; output captured by CUDA's printf buffer.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute


pytestmark = pytest.mark.sm120


@cute.kernel
def hello_world(meta_arg: cutlass.Constexpr, dynamic_arg: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 1 or tidx == 3:
        cute.printf(
            "PASS tidx={} meta_arg={} dynamic_arg={}", tidx, meta_arg, dynamic_arg
        )


@cute.jit
def hello_world_host(meta_arg: cutlass.Constexpr, dynamic_arg: cutlass.Int32):
    hello_world(meta_arg, dynamic_arg).launch(grid=(1, 1, 1), block=(4, 1, 1))


def test_hello_world(capfd):
    cutlass.cuda.initialize_cuda_context()
    assert getattr(cutlass, "__self_cutedsl__", False)

    hello_world_host(10, 20)
    out = capfd.readouterr().out
    assert "PASS tidx=1 meta_arg=10 dynamic_arg=20" in out, out
    assert "PASS tidx=3 meta_arg=10 dynamic_arg=20" in out, out

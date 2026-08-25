"""M2: dynamic for-loop + tensor pointer ABI — grid-stride elementwise add.

Uses the same two-decorator pattern as the official elementwise examples,
with dynamic M/N kernel args and boundary predication.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute
from cutlass.torch import from_dlpack


@cute.kernel
def elementwise_add(
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
    n: cutlass.Int32, scale: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bdim_x, _, _ = cute.arch.block_dim()
    bidx, _, _ = cute.arch.block_idx()
    gdim_x = bdim_x * 0  # placeholder; computed below via bdim math
    flat = bidx * bdim_x + tidx
    stride = bdim_x * 1
    # grid-stride loop with dynamic bound -> scf.for
    for i in range(flat, n, stride):
        c[i] = a[i] + b[i] * scale


@cute.jit
def launch_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
               n: cutlass.Int32, scale: cutlass.Constexpr):
    elementwise_add(a, b, c, n, scale).launch(
        grid=(8, 1, 1), block=(128, 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("n", [1024, 4096, 1000])  # 1000: non-multiple of stride
def test_elementwise_add(n):
    cutlass.cuda.initialize_cuda_context()
    a = torch.arange(n, device="cuda", dtype=torch.float32)
    b = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)
    c = torch.zeros(n, device="cuda", dtype=torch.float32)

    launch_add(from_dlpack(a), from_dlpack(b), from_dlpack(c), n, 3)
    torch.cuda.synchronize()

    expected = a + b * 3
    assert torch.allclose(c, expected, atol=1e-5), \
        f"max diff {float((c - expected).abs().max())}"

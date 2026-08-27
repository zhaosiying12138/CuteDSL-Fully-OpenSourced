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


@cute.kernel
def copy_2d(src: cute.Tensor, dst: cute.Tensor,
            rows: cutlass.Int32, cols: cutlass.Constexpr):
    row, _, _ = cute.arch.thread_idx()
    if row < rows:
        for col in range(cols):
            dst[row, col] = src[row, col]


@cute.jit
def launch_copy_2d(src: cute.Tensor, dst: cute.Tensor,
                   rows: cutlass.Int32, cols: cutlass.Constexpr):
    copy_2d(src, dst, rows, cols).launch(
        grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def widen_u32(word: cutlass.Uint32):
    return cutlass.Uint64(cutlass.Uint32(word)) | cutlass.Uint64(0)


@cute.kernel
def zero_extend_u32(out: cute.Tensor, word: cutlass.Uint32):
    if cute.arch.thread_idx()[0] == 0:
        out[0] = widen_u32(word)


@cute.jit
def launch_zero_extend_u32(out: cute.Tensor, word: cutlass.Uint32):
    zero_extend_u32(out, word).launch(
        grid=(1, 1, 1), block=(32, 1, 1))


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


@pytest.mark.sm120
def test_kernel_tensor_2d_load_store_respects_strides():
    """Tuple coordinates use tensor strides instead of summing dimensions."""
    cutlass.cuda.initialize_cuda_context()
    src = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
    dst = torch.full_like(src, -1)

    launch_copy_2d(from_dlpack(src), from_dlpack(dst), 4, 8)
    torch.cuda.synchronize()

    assert torch.equal(dst, src)


@pytest.mark.sm120
def test_uint32_to_uint64_zero_extends():
    """Packed u32 payloads must not sign-extend through bit 31."""
    cutlass.cuda.initialize_cuda_context()
    out = torch.zeros(1, device="cuda", dtype=torch.int64)

    launch_zero_extend_u32(from_dlpack(out), 0x80000001)
    torch.cuda.synchronize()

    assert out.item() == 0x80000001

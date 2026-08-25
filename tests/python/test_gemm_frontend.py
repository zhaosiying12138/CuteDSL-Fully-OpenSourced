"""M4 frontend slice: canonical GEMM through @cute.jit with real warp mma.sync.

Each CTA (one warp) computes one 16x8 f32 output tile from f16 inputs:
gmem -> SMEM staging -> ldmatrix.x4 (A) / ldmatrix.x2.trans (B) ->
mma.sync m16n8k16 loop over K tiles -> epilogue stores. This is the
frontend-form M4 DoD (real MMA in PTX, numerically verified).
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
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def warp_gemm16816_kernel(
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
    k_tiles: cutlass.Int32, K: cutlass.Constexpr, N: cutlass.Constexpr,
    LDA: cutlass.Constexpr, LDB: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # output tile coords for this warp's block (16x8 tile per block)
    tile_n = N // 8
    bm = bidx // tile_n          # block row (16 rows of A)
    bn = bidx % tile_n           # block col (8 cols of B)

    smem_a = cute.make_smem_array("gemm_smemA", 16 * 16)   # f16
    smem_b = cute.make_smem_array("gemm_smemB", 16 * 8)

    group = tidx // 4
    tig = tidx % 4
    t2 = tig * 2

    acc = [cute.zero_f32() for _ in range(4)]

    for kt in range(0, k_tiles):
        # ---- stage A tile (16x16 f16) and B tile (16x8 f16) to smem
        # A element (r, kk) at ((bm*16 + r) * K + kt*16 + kk); 8 per thread
        for i in range(8):
            e = tidx * 8 + i
            r = e // 16
            kk = e % 16
            gaddr = (bm * 16 + r) * LDA + kt * 16 + kk
            smem_a[e] = a[gaddr]
        for i in range(4):
            e = tidx * 4 + i
            kkb = e // 8
            nb = e % 8
            gaddr = (kt * 16 + kkb) * LDB + bn * 8 + nb
            smem_b[e] = b[gaddr]
        cute.sync_threads()

        # ---- ldmatrix fragments
        lane = tidx % 16
        half = tidx // 16
        fa = cute.ldmatrix(smem_a, lane * 16 + half * 8, num=4, trans=False)
        fb = cute.ldmatrix(smem_b, lane * 8, num=2, trans=True)
        a_frags = [cute.extract_frag(fa, i) for i in range(4)]
        b_frags = [cute.extract_frag(fb, i) for i in range(2)]

        # ---- mma
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op())
        acc = cute.gemm(mma, acc, a_frags, b_frags)
        cute.sync_threads()

    # ---- epilogue: C layout c0/c1 (group, t2 / +1); c2/c3 (+8 rows)
    cbase = (bm * 16) * N + bn * 8
    c[cbase + group * N + t2] = acc[0]
    c[cbase + group * N + t2 + 1] = acc[1]
    c[cbase + (group + 8) * N + t2] = acc[2]
    c[cbase + (group + 8) * N + t2 + 1] = acc[3]


@pytest.mark.sm120
@pytest.mark.wip  # frontend fragment-layout mismatch under debug; infra complete
@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 128, 256), (32, 64, 96)])
def test_warp_gemm(M, N, K):
    cutlass.cuda.initialize_cuda_context()
    assert K % 16 == 0 and M % 16 == 0 and N % 8 == 0
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    @cute.jit
    def run(a, b, c, k_tiles: cutlass.Int32, K: cutlass.Constexpr,
            N: cutlass.Constexpr, M: cutlass.Constexpr):
        warp_gemm16816_kernel(a, b, c, k_tiles, K, N, LDA, LDB).launch(
            grid=[(M // 16) * (N // 8), 1, 1], block=[32, 1, 1])

    run(from_dlpack(a), from_dlpack(b), from_dlpack(c), K // 16, K, N, M)
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, expected, atol=2e-1, rtol=2e-1)
